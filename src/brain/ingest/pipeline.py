"""``brain ingest``: load → raw copy → chunk → extract per chunk → one change set → apply →
regenerate index.md / core.md → reindex."""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Protocol

from brain.index.reindex import ReindexStats, reindex
from brain.ingest.chunking import CHUNK_TOKENS, Chunk, chunk_markdown
from brain.ingest.extract import (
    Extraction,
    KnownEntity,
    StructuredModel,
    extract_chunk,
    known_entities,
    owner_names,
    relevant_known,
)
from brain.ingest.loaders import Document, load_document
from brain.ingest.normalise import TYPE_DIRS, Normalised, build_changeset
from brain.llm.base import RetrievalModel
from brain.vault.generated import refresh_generated
from brain.vault.models import Kind
from brain.vault.raw import rel_path, save_doc, slugify
from brain.vault.store import Vault
from brain.vault.writer import ApplyResult, apply

log = logging.getLogger(__name__)


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
    chunks: list[ChunkReport] = field(default_factory=list)
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
        if not entity.name.strip():
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
    report = IngestReport(doc, rel_path(vault, raw), created)

    chunks = chunk_markdown(doc.text, chunk_tokens)
    pages = vault.pages()
    known = known_entities(pages)
    owner = owner_names(pages)
    seen: set[str] = set()  # entities named in earlier chunks of this document
    for chunk in chunks:
        start = time.perf_counter()
        extraction = extract_chunk(
            llm,
            chunk,
            doc_title=doc.title,
            known=relevant_known(known, chunk.text, seen),
            owner_names=owner,
            today=today,
        )
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
        rationale=f"Imported from {doc.path.name} ({len(chunks)} {noun}).",
    )
    report.applied = apply(report.normalised.changeset, vault)
    report.generated = refresh_generated(vault, max_tokens=core_max_tokens, today=today)
    if db is not None:
        report.reindexed = reindex(vault, db, llm)
    return report
