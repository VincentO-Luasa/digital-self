"""``brain ingest``: load → raw copy → chunk → extract per chunk → one change set → apply →
regenerate index.md / core.md → reindex.

A URL is fetched first; the fetched bytes become the raw copy and are read from there, so a web
page goes through exactly the same path as a file. ``follow`` ingests same-origin linked pages,
each as its own raw document and change set."""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Protocol

import httpx

from brain.index.reindex import ReindexStats, reindex
from brain.ingest.chunking import CHUNK_TOKENS, Chunk, chunk_markdown, halve
from brain.ingest.extract import (
    Extraction,
    KnownEntity,
    StructuredModel,
    extract_chunk,
    known_entities,
    owner_names,
    relevant_known,
)
from brain.ingest.normalise import LITERAL_TYPES, TYPE_DIRS, Normalised, build_changeset
from brain.ingest.readers import Document, UnsupportedDocument, load_document
from brain.ingest.readers.web import Fetched, FetchError, fetch, make_client, same_origin_links
from brain.llm.base import OutputTruncated, RetrievalModel, StructuredOutputError
from brain.vault import gitops
from brain.vault.generated import refresh_generated
from brain.vault.models import Kind
from brain.vault.raw import rel_path, save_doc, save_doc_bytes, slugify
from brain.vault.store import Vault
from brain.vault.writer import ApplyResult, apply

log = logging.getLogger(__name__)

MAX_SPLITS = 2  # a truncated chunk is halved at most twice (quarters)


class IngestModel(StructuredModel, RetrievalModel, Protocol):
    """Structured output (extraction) + embeddings (reindex)."""


@dataclass
class ChunkReport:
    chunk: Chunk
    seconds: float
    entities: int
    facts: int


@dataclass
class IngestReport:
    document: Document
    raw_path: str
    raw_created: bool
    source: str = ""  # the file name or URL the document came from
    chunks: list[ChunkReport] = field(default_factory=list)
    failed: list[tuple[Chunk, str]] = field(default_factory=list)  # skipped chunks + reason
    extractions: list[Extraction] = field(default_factory=list)
    normalised: Normalised | None = None
    applied: ApplyResult | None = None
    generated: list[Path] = field(default_factory=list)
    reindexed: ReindexStats | None = None


def remember_entities(known: dict[str, KnownEntity], extraction: Extraction) -> list[str]:
    """Add the chunk's entities to ``known`` (new ones under the id the normaliser will give
    them) so later chunks of the same document reuse them; return their ids."""
    ids = []
    for entity in extraction.entities:
        if not entity.name.strip() or entity.type in LITERAL_TYPES:
            continue
        entity_id = entity.id if entity.id in known else ""
        entity_id = entity_id or f"{TYPE_DIRS[entity.type]}/{slugify(entity.name)}"
        names = list(known[entity_id].names) if entity_id in known else []
        for name in (entity.name, *entity.aliases):
            if name.strip() and name.strip().casefold() not in map(str.casefold, names):
                names.append(name.strip())
        known[entity_id] = KnownEntity(entity_id, tuple(names))
        ids.append(entity_id)
    return ids


@dataclass
class WebReport:
    pages: list[IngestReport] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (url, reason) of linked pages


def ingest(
    path: Path,
    vault: Vault,
    llm: IngestModel,
    *,
    db: sqlite3.Connection | None = None,
    kind: Kind = "stated",
    today: date | None = None,
    core_max_tokens: int = 1500,
    chunk_tokens: int = CHUNK_TOKENS,
    on_chunk: Callable[[ChunkReport, int], None] | None = None,
) -> IngestReport:
    """Ingest one document. ``kind`` is ``stated`` for the owner's own documents (CV, homepage)
    and ``observed`` for documents written by others."""
    today = today or date.today()
    doc = load_document(path)
    raw, created = save_doc(vault, Path(path), today=today)
    return _ingest_document(
        doc,
        raw,
        created,
        Path(path).name,
        vault,
        llm,
        db=db,
        kind=kind,
        today=today,
        core_max_tokens=core_max_tokens,
        chunk_tokens=chunk_tokens,
        on_chunk=on_chunk,
    )


def ingest_url(
    url: str,
    vault: Vault,
    llm: IngestModel,
    *,
    follow: int = 0,
    http: httpx.Client | None = None,
    db: sqlite3.Connection | None = None,
    kind: Kind = "stated",
    today: date | None = None,
    core_max_tokens: int = 1500,
    chunk_tokens: int = CHUNK_TOKENS,
    on_page: Callable[[str, int, int], None] | None = None,
    on_chunk: Callable[[ChunkReport, int], None] | None = None,
) -> WebReport:
    """Fetch and ingest a web page, then up to ``follow`` same-origin pages it links to.

    A failure to fetch the first page raises ``FetchError``; a linked page that cannot be
    fetched or read is recorded in ``WebReport.failed`` and skipped.
    """
    today = today or date.today()
    client = http or make_client()
    report = WebReport()
    try:
        first = fetch(url, client)
        links = same_origin_links(first.data, first.url)[:follow] if first.suffix == ".html" else []
        total = 1 + len(links)
        queue: list[tuple[str, Fetched | None]] = [(first.url, first), *((u, None) for u in links)]
        for n, (page_url, fetched) in enumerate(queue, 1):
            if on_page:
                on_page(page_url, n, total)
            try:
                fetched = fetched or fetch(page_url, client)
                raw, created = save_doc_bytes(
                    vault, fetched.data, fetched.name, fetched.suffix, today=today, commit=False
                )
                if created:  # the commit message is where the raw copy's URL is recorded
                    message = f"brain: raw doc {rel_path(vault, raw)} from {fetched.url}"
                    gitops.commit_all(vault.root, message)
                doc = load_document(raw)
            except (FetchError, UnsupportedDocument) as err:
                if fetched is first:
                    raise
                report.failed.append((page_url, str(err)))
                continue
            report.pages.append(
                _ingest_document(
                    doc,
                    raw,
                    created,
                    fetched.url,
                    vault,
                    llm,
                    db=db,
                    kind=kind,
                    today=today,
                    core_max_tokens=core_max_tokens,
                    chunk_tokens=chunk_tokens,
                    on_chunk=on_chunk,
                )
            )
    finally:
        if http is None:
            client.close()
    return report


def _ingest_document(
    doc: Document,
    raw: Path,
    created: bool,
    source: str,
    vault: Vault,
    llm: IngestModel,
    *,
    db: sqlite3.Connection | None,
    kind: Kind,
    today: date,
    core_max_tokens: int,
    chunk_tokens: int,
    on_chunk: Callable[[ChunkReport, int], None] | None,
) -> IngestReport:
    report = IngestReport(doc, rel_path(vault, raw), created, source)
    chunks = chunk_markdown(doc.text, chunk_tokens)
    pages = vault.pages()
    known = known_entities(pages)
    owner = owner_names(pages)
    seen: set[str] = set()  # entities named in earlier chunks of this document
    queue: list[tuple[Chunk, int]] = [(chunk, 0) for chunk in chunks]
    while queue:
        chunk, splits = queue.pop(0)
        start = time.perf_counter()
        try:
            extraction = extract_chunk(
                llm,
                chunk,
                doc_title=doc.title,
                known=relevant_known(known, chunk.text, seen),
                owner_names=owner,
                today=today,
            )
        except OutputTruncated as err:
            halves = halve(chunk) if splits < MAX_SPLITS else []
            log.info("ingest chunk %d truncated; split into %d", chunk.index, len(halves))
            if halves:
                queue[:0] = [(half, splits + 1) for half in halves]
            else:
                report.failed.append((chunk, str(err)))
            continue
        except StructuredOutputError as err:
            log.warning("ingest chunk %d skipped: %s", chunk.index, err)
            report.failed.append((chunk, str(err)))
            continue
        seconds = time.perf_counter() - start
        for name in extraction.owner_names:
            if name.strip() and name.strip().casefold() not in map(str.casefold, owner):
                owner.append(name.strip())
        seen.update(remember_entities(known, extraction))
        item = ChunkReport(chunk, seconds, len(extraction.entities), len(extraction.facts))
        log.info("ingest %s chunk %d: %.1fs", doc.path.name, chunk.index, seconds)
        report.chunks.append(item)
        report.extractions.append(extraction)
        if on_chunk:
            on_chunk(item, len(chunks))

    noun = "chunk" if len(chunks) == 1 else "chunks"
    report.normalised = build_changeset(
        report.extractions,
        vault,
        src=report.raw_path,
        kind=kind,
        rationale=f"Imported from {source} ({len(chunks)} {noun}).",
        today=today,
    )
    report.applied = apply(report.normalised.changeset, vault)
    report.generated = refresh_generated(vault, max_tokens=core_max_tokens, today=today)
    if db is not None:
        report.reindexed = reindex(vault, db, llm)
    return report
