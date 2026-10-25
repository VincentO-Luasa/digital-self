"""Hybrid search over indexed items: aliases + FTS5 BM25 + vectors, fused by RRF, then reranked.

Phase 1 retrieval (``docs/retrieval.md``, build order). Pipeline:

1. alias matches in the query (casefold, word boundaries) → items of those entities form an
   extra ranked list (ordered by vector similarity), which boosts them in the fusion;
2. FTS5 BM25 top-N and sqlite-vec top-N under the same filters;
3. reciprocal rank fusion of the three lists;
4. the top ``rerank_top`` fused candidates are reordered by the reranker.

Filters are applied inside SQL: ``include_private=False`` keeps only ``privacy = 'normal'``
items (what the MCP server must use); ``as_of`` keeps facts valid at that (partial) date and
admits superseded history facts; without ``as_of`` only current facts are returned.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

import numpy as np

from brain.config import RetrievalSettings
from brain.index.db import vector_blob
from brain.index.temporal import period_start
from brain.llm.base import RetrievalModel

log = logging.getLogger(__name__)

RRF_K = 60
MAX_KNN = 4096
STOPWORDS = frozenset(
    (  # noqa: SIM905
        "a an and are as at be but by did do does for from had has have he her his how i if in is "
        "it its me my of on or our she so that the their them they this to was we were what when "
        "where which who why will with you your"
    ).split()
)


@dataclass(frozen=True)
class Hit:
    id: str
    kind: str  # fact | summary | episode | observation
    entity_id: str | None
    text: str
    score: float  # rerank relevance when reranked, else the fused RRF score
    privacy: str
    sources: tuple[str, ...]
    date: str | None = None
    importance: int | None = None
    current: bool = True  # False for superseded (history) facts, only seen with as_of
    rrf: float = 0.0
    rerank: float | None = None


# -- aliases -------------------------------------------------------------------------------------


class AliasMatcher:
    """Finds entity aliases in text (casefolded, whole words, longest alias wins)."""

    def __init__(self, aliases: Iterable[tuple[str, str]]) -> None:
        self.ids: dict[str, list[str]] = {}
        for alias, entity_id in aliases:
            if alias.strip():
                self.ids.setdefault(alias.casefold(), []).append(entity_id)
        names = sorted(self.ids, key=len, reverse=True)
        body = "|".join(map(re.escape, names))
        self.pattern = re.compile(rf"(?<!\w)(?:{body})(?!\w)") if names else None

    @classmethod
    def from_db(cls, db: sqlite3.Connection) -> AliasMatcher:
        return cls(db.execute("SELECT alias_cf, entity_id FROM aliases"))

    def match(self, text: str) -> list[str]:
        if self.pattern is None:
            return []
        found: dict[str, None] = {}
        for m in self.pattern.finditer(text.casefold()):
            found.update(dict.fromkeys(self.ids[m.group(0)]))
        return list(found)


def matched_entities(db: sqlite3.Connection, text: str) -> list[str]:
    """Entity ids whose aliases (or titles) occur in ``text``, in order of first mention."""
    return AliasMatcher.from_db(db).match(text)


# -- filters -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Filters:
    include_private: bool = True
    as_of: str | None = None
    kinds: tuple[str, ...] | None = None

    def where(self) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if not self.include_private:
            clauses.append("i.privacy = 'normal'")
        if self.kinds:
            clauses.append(f"i.kind IN ({', '.join('?' * len(self.kinds))})")
            params.extend(self.kinds)
        if self.as_of is None:
            clauses.append("(i.kind != 'fact' OR f.current = 1)")
        else:
            clauses.append("(i.kind != 'fact' OR brain_valid_at(f.valid_from, f.valid_to, ?))")
            params.append(self.as_of)
        return " AND ".join(clauses), params


FACT_JOIN = "LEFT JOIN facts f ON f.id = i.id AND i.kind = 'fact'"
FROM_ITEMS = f"items i {FACT_JOIN}"


# -- candidate lists -----------------------------------------------------------------------------


def fts_query(text: str) -> str | None:
    terms = [t for t in re.findall(r"\w+", text.casefold()) if t not in STOPWORDS and len(t) > 1]
    unique = list(dict.fromkeys(terms))
    return " OR ".join(f'"{t}"' for t in unique) or None


def fts_ranked(db: sqlite3.Connection, query: str, filters: Filters, limit: int) -> list[int]:
    match = fts_query(query)
    if match is None:
        return []
    where, params = filters.where()
    rows = db.execute(
        f"SELECT i.pk FROM items_fts JOIN items i ON i.pk = items_fts.rowid {FACT_JOIN} "
        f"WHERE items_fts MATCH ? AND {where} ORDER BY bm25(items_fts) LIMIT ?",
        [match, *params, limit],
    )
    return [r[0] for r in rows]


def vector_ranked(
    db: sqlite3.Connection, vector: np.ndarray, filters: Filters, limit: int
) -> list[int]:
    where, params = filters.where()
    k = min(MAX_KNN, max(limit * 4, 100))
    rows = db.execute(
        "WITH knn AS (SELECT rowid AS pk, distance FROM vec_items"
        " WHERE embedding MATCH ? AND k = ?)"
        f" SELECT i.pk FROM knn JOIN items i ON i.pk = knn.pk {FACT_JOIN} WHERE {where}"
        " ORDER BY knn.distance LIMIT ?",
        [vector_blob(vector), k, *params, limit],
    )
    return [r[0] for r in rows]


def boostable(db: sqlite3.Connection, entity_ids: Sequence[str]) -> list[str]:
    """Drop ``self`` pages: nearly everything is about the owner, so they carry no signal."""
    if not entity_ids:
        return []
    marks = ", ".join("?" * len(entity_ids))
    selfish = {
        r[0]
        for r in db.execute(
            f"SELECT id FROM entities WHERE id IN ({marks}) AND type = 'self'", list(entity_ids)
        )
    }
    return [e for e in entity_ids if e not in selfish and e != "self"]


def entity_ranked(
    db: sqlite3.Connection,
    entity_ids: Sequence[str],
    vector: np.ndarray,
    filters: Filters,
    limit: int,
) -> list[int]:
    """Items about, pointing at, or mentioning the entities, by similarity to the query."""
    if not entity_ids:
        return []
    where, params = filters.where()
    marks = ", ".join("?" * len(entity_ids))
    ids = list(entity_ids)
    pks = [
        r[0]
        for r in db.execute(
            f"SELECT i.pk FROM {FROM_ITEMS} WHERE {where} AND (i.entity_id IN ({marks})"
            f" OR f.subject IN ({marks}) OR f.o_entity IN ({marks})"
            f" OR i.id IN (SELECT src_id FROM links WHERE dst_id IN ({marks})))",
            [*params, *ids, *ids, *ids, *ids],
        )
    ]
    if not pks:
        return []
    rows = db.execute(
        f"SELECT rowid, embedding FROM vec_items WHERE rowid IN ({', '.join('?' * len(pks))})",
        pks,
    ).fetchall()
    sims = {r[0]: float(np.frombuffer(r[1], dtype=np.float32) @ vector) for r in rows}
    return sorted(pks, key=lambda pk: sims.get(pk, -1.0), reverse=True)[:limit]


def rrf(lists: Iterable[Sequence[int]], k: int = RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranked in lists:
        for rank, pk in enumerate(ranked):
            scores[pk] = scores.get(pk, 0.0) + 1.0 / (k + rank + 1)
    return scores


# -- assembly ------------------------------------------------------------------------------------


def load_hits(db: sqlite3.Connection, scores: dict[int, float], pks: Sequence[int]) -> list[Hit]:
    if not pks:
        return []
    rows = db.execute(
        "SELECT i.pk, i.id, i.kind, i.entity_id, i.text, i.privacy, i.src_json, i.date,"
        f" i.importance, f.current FROM {FROM_ITEMS}"
        f" WHERE i.pk IN ({', '.join('?' * len(pks))})",
        list(pks),
    )
    by_pk = {r["pk"]: r for r in rows}
    return [_hit(by_pk[pk], scores[pk]) for pk in pks if pk in by_pk]


def _hit(row: sqlite3.Row, score: float) -> Hit:
    return Hit(
        id=row["id"],
        kind=row["kind"],
        entity_id=row["entity_id"],
        text=row["text"],
        score=score,
        privacy=row["privacy"],
        sources=tuple(json.loads(row["src_json"])),
        date=row["date"],
        importance=row["importance"],
        current=row["current"] != 0,
        rrf=score,
    )


def apply_rerank(llm: RetrievalModel, query: str, hits: list[Hit], top: int) -> list[Hit]:
    head, tail = hits[:top], hits[top:]
    if not head:
        return hits
    try:
        ranked = llm.rerank(query, [h.text for h in head], top_n=len(head))
    except Exception as err:  # reranking is an optimisation; never fail the search over it
        log.warning("rerank failed, keeping fused order: %s", err)
        return hits
    seen = {i for i, _ in ranked}
    reranked = [_with_rerank(head[i], s) for i, s in ranked]
    reranked += [h for i, h in enumerate(head) if i not in seen]
    return reranked + tail


def _with_rerank(hit: Hit, score: float) -> Hit:
    return replace(hit, score=score, rerank=score)


def search(
    db: sqlite3.Connection,
    llm: RetrievalModel,
    query: str,
    *,
    k: int = 12,
    include_private: bool = True,
    as_of: str | date | None = None,
    kinds: Collection[str] | None = None,
    settings: RetrievalSettings | None = None,
    rerank: bool = True,
    query_vector: np.ndarray | None = None,
) -> list[Hit]:
    """Top ``k`` items for ``query``. ``query_vector`` overrides embedding the query text."""
    settings = settings or RetrievalSettings()
    as_of_s = as_of.isoformat() if isinstance(as_of, date) else as_of
    if as_of_s is not None:
        period_start(as_of_s)  # validate early
    filters = Filters(include_private, as_of_s, tuple(kinds) if kinds else None)
    n = max(settings.candidates, k)

    vector = query_vector if query_vector is not None else llm.embed([query], query=True)[0]
    entities = boostable(db, matched_entities(db, query))
    lists = [
        fts_ranked(db, query, filters, n),
        vector_ranked(db, vector, filters, n),
        entity_ranked(db, entities, vector, filters, n),
    ]
    scores = rrf(lists)
    order = sorted(scores, key=lambda pk: scores[pk], reverse=True)
    hits = load_hits(db, scores, order[: max(k, settings.rerank_top)])
    if rerank:
        hits = apply_rerank(llm, query, hits, settings.rerank_top)
    return hits[:k]
