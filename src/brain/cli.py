"""``brain`` command line. Model access goes through ``make_llm`` so tests can swap in a fake."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from brain.config import Settings, load_settings
from brain.index.db import connect, open_index, read_meta
from brain.index.reindex import ReindexStats, reindex
from brain.llm import LLM
from brain.vault import gitops
from brain.vault.generated import refresh_generated
from brain.vault.init import init_vault
from brain.vault.store import Vault
from brain.vault.writer import Review, ReviewNotFound, approve, pending_reviews, reject

app = typer.Typer(
    name="brain",
    help="Digital Self: a local second brain (markdown vault, local model, read-only chat).",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


# --- factories (monkeypatched in tests) --------------------------------------------------------


def get_settings() -> Settings:
    return load_settings()


def make_llm(settings: Settings) -> Any:
    return LLM(settings.llm)


def make_http() -> Any:
    """HTTP client for ``brain ingest <url>`` (tests swap in an ``httpx.MockTransport``)."""
    from brain.ingest.readers.web import make_client

    return make_client()


# --- helpers -----------------------------------------------------------------------------------


def _fail(message: str) -> typer.Exit:
    console.print(f"[red]error:[/] {escape(message)}")
    return typer.Exit(1)


def _vault(settings: Settings) -> Vault:
    if not (settings.vault / "SCHEMA.md").exists():
        raise _fail(f"no vault at {settings.vault}; run `brain init-vault` first")
    return Vault(settings.vault)


def _model(settings: Settings) -> Any:
    llm = make_llm(settings)
    if not llm.health():
        raise _fail(f"model server not reachable at {settings.llm.base_url}")
    return llm


def _open_db(settings: Settings, llm: Any) -> sqlite3.Connection:
    return open_index(settings.db_path, embed_model=llm.embed_model, dim=llm.embed_dim)


def _print_reindex(stats: ReindexStats) -> None:
    console.print(
        f"index: {stats.items} items · +{stats.added} new · {stats.changed} changed · "
        f"{stats.deleted} deleted · {stats.embedded} embedded ({stats.seconds:.1f}s)"
    )


def _refresh(vault: Vault, settings: Settings, llm: Any | None) -> None:
    """After memory changed: regenerate index.md / core.md, then reindex if the model is up."""
    changed = refresh_generated(vault, max_tokens=settings.core_max_tokens)
    if changed:
        console.print("regenerated: " + ", ".join(p.name for p in changed))
    if llm is None or not llm.health():
        console.print("[yellow]model server down: run `brain reindex` later[/]")
        return
    db = _open_db(settings, llm)
    try:
        _print_reindex(reindex(vault, db, llm))
    finally:
        db.close()


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# --- commands ----------------------------------------------------------------------------------


@app.command("init-vault")
def init_vault_cmd(
    path: Annotated[Path | None, typer.Option(help="Vault folder (default: config).")] = None,
    update_template: Annotated[
        bool, typer.Option(help="Overwrite workspace template files with the repo's.")
    ] = False,
) -> None:
    """Create (or complete) the vault: layout, root files, workspace template, git repo."""
    settings = get_settings()
    root = (path or settings.vault).expanduser()
    try:
        result = init_vault(root, update_template=update_template)
    except ValueError as err:
        raise _fail(str(err)) from err
    vault = Vault(result.root)
    refresh_generated(vault, max_tokens=settings.core_max_tokens)
    console.print(f"vault: {result.root}")
    console.print(
        f"created {len(result.created)} · updated {len(result.updated)} · "
        f"kept {len(result.kept)} files"
    )
    for warning in result.warnings:
        console.print(f"[yellow]warning:[/] {escape(warning)}")


@app.command()
def status() -> None:
    """Model server health, vault contents, pending reviews and index stats."""
    settings = get_settings()
    table = Table(show_header=False, box=None, pad_edge=False)
    llm = make_llm(settings)
    healthy = llm.health()
    table.add_row(
        "model server",
        ("[green]up[/]" if healthy else "[red]down[/]") + f" · {settings.llm.base_url}",
    )
    table.add_row("chat model", settings.llm.chat_model)
    if not (settings.vault / "SCHEMA.md").exists():
        table.add_row("vault", f"[red]missing[/] · {settings.vault} (run `brain init-vault`)")
        console.print(table)
        return
    vault = Vault(settings.vault)
    pages = vault.pages()
    current = sum(len(p.facts) for p in pages)
    history = sum(len(p.history) for p in pages)
    private = sum(p.privacy == "private" for p in pages)
    raw = vault.raw_dir
    docs = len([p for p in (raw / "docs").glob("*") if p.is_file()])
    convs = len(list((raw / "conversations").rglob("*.md")))
    table.add_row("vault", str(vault.root))
    table.add_row(
        "memory",
        f"{len(pages)} pages ({private} private) · {current} facts (+{history} history) · "
        f"{len(vault.episodes())} episodes · {len(vault.observations())} observations",
    )
    table.add_row("raw", f"{docs} documents · {convs} conversations")
    pending = pending_reviews(vault)
    ops = sum(len(r.ops) for r in pending)
    table.add_row("reviews", f"{len(pending)} pending ({ops} operations)")
    table.add_row("index", _index_status(settings.db_path))
    head = gitops.head(vault.root)
    table.add_row("vault git", head[:10] if head else "no commits")
    console.print(table)


def _index_status(path: Path) -> str:
    if not path.exists():
        return "not built (run `brain reindex`)"
    conn = connect(path)
    try:
        meta = read_meta(conn)
        if meta is None:
            return "empty (run `brain reindex`)"
        items = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        vectors = conn.execute("SELECT COUNT(*) FROM vec_items").fetchone()[0]
        entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    except sqlite3.Error as err:
        return f"unreadable ({err})"
    finally:
        conn.close()
    return f"{items} items · {vectors} vectors · {entities} entities · {meta.embed_model}"


@app.command("reindex")
def reindex_cmd(
    full: Annotated[bool, typer.Option("--full", help="Re-embed every item.")] = False,
) -> None:
    """Sync the SQLite index with the vault (markdown is the source of truth)."""
    settings = get_settings()
    vault = _vault(settings)
    llm = _model(settings)
    db = _open_db(settings, llm)
    try:
        _print_reindex(reindex(vault, db, llm, full=full))
    finally:
        db.close()


@app.command("ingest")
def ingest_cmd(
    source: Annotated[str, typer.Argument(help="A file (html, pdf, md, txt) or an http(s) URL.")],
    observed: Annotated[
        bool,
        typer.Option(
            "--observed", help="Written by someone else: facts are 'observed', not 'stated'."
        ),
    ] = False,
    follow: Annotated[
        int,
        typer.Option("--follow", min=0, help="Also ingest up to N same-origin pages it links to."),
    ] = 0,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Re-run the model even for excerpts it already read."),
    ] = False,
) -> None:
    """Import a document or a web page into memory."""
    from brain.ingest import FetchError, UnsupportedDocument, ingest, ingest_url, is_url
    from brain.ingest.pipeline import ChunkReport

    url = is_url(source)
    if not url:
        path = Path(source)
        if not path.is_file():
            raise _fail(f"{source}: no such file")
        if follow:
            raise _fail("--follow needs a URL")
    settings = get_settings()
    vault = _vault(settings)
    llm = _model(settings)
    if not no_cache:
        from brain.llm.cache import StructuredCache

        llm = StructuredCache(llm, vault.brain_dir / "cache" / "llm")
    kind = "observed" if observed else "stated"

    def progress(item: ChunkReport, total: int) -> None:
        console.print(
            f"  chunk {item.chunk.index + 1}/{total} [dim]{escape(item.chunk.heading)}[/]: "
            f"{item.entities} entities, {item.facts} facts ({item.seconds:.1f}s)"
        )

    def page(page_url: str, n: int, total: int) -> None:
        console.print(f"[bold]page {n}/{total}[/] {escape(page_url)}")

    db = _open_db(settings, llm)
    try:
        if url:
            http = make_http()
            try:
                web = ingest_url(
                    source,
                    vault,
                    llm,
                    follow=follow,
                    http=http,
                    db=db,
                    kind=kind,
                    core_max_tokens=settings.core_max_tokens,
                    on_page=page,
                    on_chunk=progress,
                )
            finally:
                http.close()
            reports = web.pages
        else:
            reports = [
                ingest(
                    path,
                    vault,
                    llm,
                    db=db,
                    kind=kind,
                    core_max_tokens=settings.core_max_tokens,
                    on_chunk=progress,
                )
            ]
    except (UnsupportedDocument, FetchError) as err:
        raise _fail(str(err)) from err
    finally:
        db.close()

    for report in reports:
        _print_ingest(report)
    if url:
        for _, reason in web.failed:  # the reason starts with the page's URL
            console.print(f"[yellow]skipped linked page {escape(reason)}[/]")


def _print_ingest(report: Any) -> None:
    raw_note = "" if report.raw_created else " (identical copy already there)"
    console.print(f"raw copy: {report.raw_path}{raw_note}")
    for chunk, reason in report.failed:
        where = escape(chunk.heading or f"chunk {chunk.index}")
        console.print(f"[yellow]skipped part of the document ({where}): {escape(reason)}[/]")
    result = report.applied
    assert result is not None and report.normalised is not None
    console.print(
        f"applied: {len(result.created)} new pages, {result.facts_added} facts · "
        f"queued: {len(result.queued)} · skipped: {len(result.skipped)}"
    )
    for page_id in result.created:
        console.print(f"  + {page_id}")
    if result.review_id:
        console.print(f"review: {result.review_id} (run `brain review`)")
        for op, reason in result.queued:
            console.print(f"  ? {op.op}: {escape(reason)}")
    for op, reason in result.skipped:
        console.print(f"  - skipped {op.op}: {escape(reason)}")
    for note in report.normalised.notes:
        console.print(f"  [dim]· {escape(note)}[/]")
    if report.generated:
        console.print("regenerated: " + ", ".join(p.name for p in report.generated))
    if report.reindexed:
        _print_reindex(report.reindexed)


def _review_changes(review: Review) -> str:
    text = review.path.read_text(encoding="utf-8")
    start = text.find("## Changes")
    end = text.find("## Operations")
    body = text[start : end if end != -1 else None] if start != -1 else text
    return body.strip()


@app.command("review")
def review_cmd(
    list_only: Annotated[bool, typer.Option("--list", help="List pending reviews, exit.")] = False,
) -> None:
    """Approve, reject or skip pending changes."""
    settings = get_settings()
    vault = _vault(settings)
    pending = pending_reviews(vault)
    if not pending:
        console.print("No pending reviews.")
        return
    if list_only:
        table = Table("id", "origin", "created", "ops", "source")
        for r in pending:
            created = f"{r.created:%Y-%m-%d %H:%M}"
            table.add_row(r.id, r.origin, created, str(len(r.ops)), escape(", ".join(r.src)))
        console.print(table)
        return

    changed = False
    for n, rev in enumerate(pending, 1):
        console.rule(f"{rev.id} · {rev.origin} · {n}/{len(pending)}")
        console.print(f"source: {escape(', '.join(rev.src))}")
        if rev.rationale:
            console.print(escape(rev.rationale))
        console.print(_review_changes(rev), markup=False, highlight=False)
        choice = typer.prompt("[a]pprove / [r]eject / [s]kip / [q]uit", default="s").strip()
        choice = choice[:1].casefold()
        try:
            if choice == "a":
                result = approve(rev.id, vault)
                console.print(
                    f"approved: +{result.facts_added} facts, {len(result.created)} pages, "
                    f"{len(result.skipped)} skipped"
                )
                for op, reason in result.skipped:
                    console.print(f"  - skipped {op.op}: {escape(reason)}")
                changed = True
            elif choice == "r":
                reason = typer.prompt("reason", default="rejected in review")
                reject(rev.id, reason, vault)
                console.print("rejected")
            elif choice == "q":
                break
        except ReviewNotFound:
            console.print(f"[yellow]{rev.id} is no longer pending[/]")
    if changed:
        _refresh(vault, settings, make_llm(settings))


@app.command("chat")
def chat_cmd(
    message: Annotated[
        list[str] | None,
        typer.Option("--message", "-m", help="Send this message and exit (repeatable)."),
    ] = None,
) -> None:
    """Chat with your brain (read-only in Phase 1; the transcript is saved to raw/)."""
    from brain.chat import ChatSession, run_repl

    settings = get_settings()
    vault = _vault(settings)
    llm = _model(settings)
    db = _open_db(settings, llm)
    try:
        reindex(vault, db, llm)
        session = ChatSession(vault, llm, db, settings)
        if message:
            for text in message:
                console.print(f"[bold cyan]you ›[/] {escape(text)}")
                console.print("[bold green]brain ›[/] ", end="")
                for delta in session.turn(text):
                    console.print(delta, end="", markup=False, highlight=False)
                console.print()
        else:
            run_repl(session, console)
        if session.close() and session.transcript:
            console.print(f"[dim]transcript: {session.transcript.rel}[/]")
    finally:
        db.close()
