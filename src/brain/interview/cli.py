"""``brain interview``: let the brain ask (registered in ``brain.cli``).

Model access and settings go through ``brain.cli``'s factories (``get_settings``, ``make_llm``),
so the CLI tests' fakes apply here too; ``make_process`` is this module's seam for the memory
pipeline.
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Callable, Iterator, Sequence
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from brain.interview.coverage import CoverageReport, coverage_report
from brain.interview.planner import (
    PlanItem,
    RegionError,
    build_plan,
    choose_region,
    region_scores,
)
from brain.interview.questions import GapItem, read_questions, write_questions
from brain.interview.session import InterviewSession, ProcessSession, Reply, default_process

console = Console()


def make_process() -> ProcessSession:
    """The memory pipeline entry point (tests swap in a fake)."""
    return default_process()


# --- rendering ---------------------------------------------------------------------------------


def coverage_table(report: CoverageReport, items: Sequence[GapItem] = ()) -> Table:
    """Coverage per region, best interview candidate first (usable by ``brain status``)."""
    table = Table("region", "coverage", "filled", "partial", "stale", "empty", "score", "")
    scores = sorted(region_scores(report, items), key=lambda s: -s.score)
    for s in scores:
        region = report.regions[s.id]
        c = region.counts()
        entities = f" · {len(region.entities)} {region.per_entity}s" if region.per_entity else ""
        table.add_row(
            escape(s.id) + f"[dim]{escape(entities)}[/]",
            f"{s.coverage:.0%}",
            str(c["filled"]),
            str(c["partial"]),
            str(c["stale"]),
            str(c["empty"]),
            f"{s.score:.2f}" + (f" [dim](+{s.boost:.2f})[/]" if s.boost else ""),
            "[yellow]private[/]" if s.private else "",
        )
    return table


def coverage_summary(report: CoverageReport, *, lowest: int = 3) -> str:
    """One line for ``brain status``: overall coverage and the least covered regions."""
    rows = sorted((r for r in report.regions.values() if not r.private), key=lambda r: r.coverage)
    low = ", ".join(f"{r.id} {r.coverage:.0%}" for r in rows[:lowest])
    return f"{report.overall(include_private=False):.0%} (least: {low})"


def describe_plan(plan: Sequence[PlanItem]) -> list[str]:
    return [f"{n}. [{p.mode}] {p.topic}: {p.hint}" for n, p in enumerate(plan, 1)]


def _describe_report(report: Any) -> str:
    if report is None:
        return ""
    summary = getattr(report, "summary", None)
    if callable(summary):
        return str(summary())
    if isinstance(summary, str):
        return summary
    parts = [str(getattr(report, "status", type(report).__name__))]
    if isinstance(episodes := getattr(report, "episodes", None), list):
        parts.append(f"{len(episodes)} episodes")
    if error := getattr(report, "error", None):
        parts.append(f"error: {error}")
    return " · ".join(parts)


# --- the loop ----------------------------------------------------------------------------------


def scripted(lines: Sequence[str]) -> Callable[[], str]:
    """A reader that returns ``lines`` one by one, then raises EOFError."""
    it: Iterator[str] = iter(lines)

    def read() -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return read


def _say(out: Console, reply: Reply) -> None:
    if reply.message:
        out.print("[bold green]brain ›[/] " + escape(reply.message), highlight=False)


def run_interview(
    session: InterviewSession,
    out: Console,
    *,
    read: Callable[[], str] | None = None,
    echo: bool = False,
) -> None:
    """Ask until the session ends, the owner says stop, or input ends (EOF / Ctrl-C)."""
    read = read or (lambda: out.input("[bold cyan]you ›[/] "))
    out.print(
        f"[bold]Interview: {escape(session.label)}[/] [dim]· about {session.minutes:g} min · "
        "say skip, private (keeps an answer private) or stop[/]"
    )
    reply = session.opening()
    _say(out, reply)
    while not reply.done:
        try:
            text = read()
        except (EOFError, KeyboardInterrupt):
            out.print()
            break
        if echo:
            out.print("[bold cyan]you ›[/] " + escape(text), highlight=False)
        reply = session.reply(text)
        _say(out, reply)


# --- the command -------------------------------------------------------------------------------


def _reader(answers: Path) -> Callable[[], str]:
    text = sys.stdin.read() if str(answers) == "-" else answers.read_text(encoding="utf-8")
    return scripted([line for line in text.splitlines() if line.strip()])


def interview(
    region: Annotated[
        str | None,
        typer.Argument(
            help="Region to ask about: id, label or last part (projects, friends). "
            "Private regions only when named. Default: the least covered region."
        ),
    ] = None,
    minutes: Annotated[
        float, typer.Option("--minutes", min=1, help="Wrap up after about this long.")
    ] = 12,
    answers: Annotated[
        Path | None,
        typer.Option(
            "--answers",
            help="Answer from this file, one line per answer ('-' = stdin); non-interactive.",
        ),
    ] = None,
    show_coverage: Annotated[
        bool, typer.Option("--coverage", help="Show coverage per region, then exit.")
    ] = False,
    plan_only: Annotated[
        bool, typer.Option("--plan", help="Show what would be asked, then exit (no model call).")
    ] = False,
) -> None:
    """Let the brain interview you to fill gaps in what it knows (one question at a time)."""
    from brain import cli as main
    from brain.index.db import connect
    from brain.index.reindex import reindex
    from brain.vault import gitops

    settings = main.get_settings()
    vault = main._vault(settings)
    read_only = show_coverage or plan_only
    llm = main.make_llm(settings)
    healthy = llm.health()
    if not healthy and not read_only:
        raise main._fail(f"model server not reachable at {settings.llm.base_url}")
    db: sqlite3.Connection
    if healthy:
        db = main._open_db(settings, llm)
    elif settings.db_path.exists():
        db = connect(settings.db_path)
    else:
        raise main._fail("no index yet: run `brain reindex` first")
    try:
        if healthy:
            reindex(vault, db, llm)
        today = date.today()
        report = coverage_report(db, vault.lifemap, today)
        if not read_only and write_questions(vault, report):
            gitops.commit_all(vault.root, "brain: regenerate questions.md")
        items = read_questions(vault)
        try:
            choice = choose_region(report, items, region)
        except RegionError as err:
            raise main._fail(str(err)) from err
        if show_coverage:
            console.print(coverage_table(report, items))
            console.print(f"overall (non-private): {coverage_summary(report)}")
            if choice:
                console.print(f"next interview: [bold]{escape(choice.id)}[/]")
            return
        if choice is None:
            console.print("Nothing left to ask right now: every non-private region is covered.")
            return
        plan = build_plan(report, choice.id, items)
        if plan_only:
            console.print(f"[bold]{escape(choice.id)}[/] (coverage {choice.coverage:.0%})")
            for line in describe_plan(plan):
                console.print(escape(line), highlight=False)
            return
        session = InterviewSession(vault, llm, report, choice.id, plan, settings, minutes=minutes)
        read = _reader(answers) if answers else None
        run_interview(session, console, read=read, echo=answers is not None)
        if session.answers:
            console.print("[dim]saving to memory…[/]")
        result = session.close(process=_process_or_none(), db=db)
    finally:
        db.close()
    if result.transcript:
        console.print(f"[dim]transcript: {result.transcript}[/]")
    if result.error:
        console.print(
            f"[yellow]memory pipeline failed ({escape(result.error)}); the transcript is "
            "saved and can be processed later[/]"
        )
    elif result.report is not None:
        text = _describe_report(result.report)
        console.print(escape(text if text.startswith("memory:") else f"memory: {text}"))
    if result.answered:
        console.print(f"answered questions.md items: {', '.join(result.answered)}")


def _process_or_none() -> ProcessSession:
    try:
        return make_process()
    except ImportError:
        return _missing_pipeline


def _missing_pipeline(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("the memory pipeline (brain.memory.process_session) is not available")
