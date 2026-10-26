"""Rebuild the SQLite index from the vault (markdown is the source of truth).

Graph tables (entities, aliases, facts, links) are cheap and rewritten on every run. Retrievable
``items`` are synced incrementally: an item is re-embedded only when its text hash changes, it is
new, or its vector is missing; metadata changes (privacy, importance, ...) update in place.
Embedding happens before the write transaction, so a server failure leaves the index untouched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from brain.index.db import ensure_schema, transaction, vector_blob
from brain.llm.base import RetrievalModel
from brain.vault.lifemap import load_lifemap
from brain.vault.models import (
    WIKILINK,
    Episode,
    Fact,
    Intention,
    Observation,
    Page,
    fact_private,
    link_target,
)

log = logging.getLogger(__name__)

PRIVATE = "private"
NORMAL = "normal"


class VaultSource(Protocol):
    """The subset of ``brain.vault.store.Vault`` the indexer reads."""

    def pages(self) -> Iterable[Page]: ...
    def episodes(self) -> Iterable[Episode]: ...
    def observations(self) -> Iterable[Observation]: ...
    def intentions(self) -> Iterable[Intention]: ...
    def alias_index(self) -> dict[str, list[str]]: ...


@dataclass
class ReindexStats:
    pages: int = 0
    episodes: int = 0
    observations: int = 0
    facts: int = 0
    items: int = 0
    added: int = 0
    changed: int = 0  # text changed (re-embedded)
    updated: int = 0  # metadata only (no re-embed)
    embedded: int = 0
    deleted: int = 0
    vectors_reset: bool = False
    seconds: float = 0.0


@dataclass(frozen=True)
class Item:
    id: str
    kind: str
    entity_id: str | None
    text: str
    privacy: str
    importance: int | None
    date: str | None
    src: tuple[str, ...]

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    def meta(self) -> tuple[Any, ...]:
        return (self.kind, self.entity_id, self.privacy, self.importance, self.date, self.src_json)

    @property
    def src_json(self) -> str:
        return json.dumps(list(self.src))


@dataclass
class Snapshot:
    """Everything derived from one pass over the vault."""

    entities: list[tuple[Any, ...]] = field(default_factory=list)
    aliases: set[tuple[str, str]] = field(default_factory=set)
    facts: list[tuple[Any, ...]] = field(default_factory=list)
    links: set[tuple[str, str, str]] = field(default_factory=set)
    items: dict[str, Item] = field(default_factory=dict)

    def add_item(self, item: Item) -> None:
        if not item.text.strip():
            return
        if item.id in self.items:
            log.warning("duplicate item id %s; keeping the last one", item.id)
        self.items[item.id] = item


# -- privacy -------------------------------------------------------------------------------------


def load_private_regions(vault_root: Path | None = None) -> frozenset[str]:
    """Regions marked ``privacy: private`` in the life map (a vault-local copy wins)."""
    regions = load_lifemap(vault_root).regions
    return frozenset(rid for rid, region in regions.items() if region.privacy == PRIVATE)


def region_is_private(region: str, private_regions: Collection[str]) -> bool:
    return any(region == p or region.startswith(p + ".") for p in private_regions)


# -- entity resolution ---------------------------------------------------------------------------


def entity_names(page: Page) -> set[str]:
    """Casefolded title and aliases; like ``Vault.alias_index`` they name the page subject."""
    return {key for name in (page.title, *page.aliases) if (key := name.strip().casefold())}


class Resolver:
    def __init__(self, pages: dict[str, Page], alias_index: dict[str, list[str]]) -> None:
        self.pages = pages
        self.aliases: dict[str, set[str]] = {}
        for alias, ids in alias_index.items():
            self.aliases.setdefault(alias.casefold(), set()).update(ids)
        for page in pages.values():
            for name in entity_names(page):
                self.aliases.setdefault(name, set()).add(page.subject)

    def resolve(self, target: str) -> str | None:
        """Entity id for a wikilink target (page id, path or unambiguous alias), else None."""
        t = target.strip().removeprefix("wiki/").removesuffix(".md")
        if t in self.pages:
            return t
        ids = self.aliases.get(t.casefold(), set())
        return next(iter(ids)) if len(ids) == 1 else None

    def link(self, target: str) -> str:
        return self.resolve(target) or target.strip()

    def privacy(self, entity_id: str | None) -> str:
        page = self.pages.get(entity_id or "")
        return page.privacy if page else NORMAL

    def plain(self, text: str) -> str:
        """Render wikilinks as readable names: ``[[x|Sam]]`` → Sam, ``[[people/x]]`` → title."""

        def repl(m: Any) -> str:
            raw = m.group(0)
            if "|" in raw:
                return raw[raw.index("|") + 1 : -2].strip()
            page = self.pages.get(self.resolve(m.group(1)) or "")
            return page.title if page else m.group(1).strip()

        return WIKILINK.sub(repl, text)


def wikilink_targets(text: str) -> list[str]:
    return [m.group(1).strip() for m in WIKILINK.finditer(text)]


def ref_target(value: str) -> str:
    """Episode entity / src values may be bare ids or ``[[wikilinks]]``."""
    return link_target(value) or value.strip()


# -- builders ------------------------------------------------------------------------------------


def validity_suffix(fact: Fact) -> str:
    if fact.valid_from and fact.valid_to:
        return f" ({fact.valid_from} to {fact.valid_to})"
    if fact.valid_from:
        return f" (since {fact.valid_from})"
    if fact.valid_to:
        return f" (until {fact.valid_to})"
    return ""


def add_page(snap: Snapshot, page: Page, res: Resolver, root: Path | None) -> None:
    path = f"wiki/{page.id}.md"
    mtime = (root / path).stat().st_mtime if root and (root / path).exists() else None
    snap.entities.append(
        (page.id, page.type, page.subject, page.title, page.privacy, json.dumps(page.regions))
        + (path, mtime)
    )
    snap.aliases.update((name, page.subject) for name in entity_names(page))

    label = "" if page.type == "self" else f"{page.title}: "
    for text in (page.summary, page.notes):
        for target in wikilink_targets(text):
            snap.links.add((page.id, res.link(target), "wikilink"))
    if page.summary.strip():
        snap.add_item(
            Item(
                id=f"summary:{page.id}",
                kind="summary",
                entity_id=page.id,
                text=label + res.plain(page.summary.strip()),
                privacy=page.privacy,
                importance=None,
                date=page.updated.isoformat(),
                src=(page.id,),
            )
        )
    for current, facts in ((1, page.facts), (0, page.history)):
        for fact in facts:
            add_fact(snap, page, fact, current, label, res)


def add_fact(
    snap: Snapshot, page: Page, fact: Fact, current: int, label: str, res: Resolver
) -> None:
    o_entity = res.link(fact.object_id) if fact.object_id else None
    src = tuple(ref_target(s) for s in fact.src)
    snap.facts.append(
        (fact.id, page.subject, fact.p, fact.o, o_entity, fact.text, fact.valid_from)
        + (fact.valid_to, fact.rec.isoformat(), fact.kind, json.dumps(list(src)), current, page.id)
    )
    if current:
        if o_entity:
            snap.links.add((page.subject, o_entity, "fact"))
        for target in wikilink_targets(fact.text):
            snap.links.add((page.id, res.link(target), "wikilink"))
    snap.add_item(
        Item(
            id=fact.id,
            kind="fact",
            entity_id=page.id,
            text=label + res.plain(fact.text) + validity_suffix(fact),
            privacy=PRIVATE if fact_private(fact) else page.privacy,
            importance=None,
            date=fact.rec.isoformat(),
            src=src,
        )
    )


def episode_privacy(
    ep: Episode, entities: list[str], res: Resolver, private: Collection[str]
) -> str:
    if ep.privacy == PRIVATE:
        return PRIVATE
    if any(res.privacy(e) == PRIVATE for e in entities):
        return PRIVATE
    return PRIVATE if any(region_is_private(r, private) for r in ep.regions) else NORMAL


def add_episode(snap: Snapshot, ep: Episode, res: Resolver, private: Collection[str]) -> str:
    entities = [res.link(ref_target(e)) for e in ep.entities]
    for entity in entities:
        snap.links.add((ep.id, entity, "mention"))
    privacy = episode_privacy(ep, entities, res, private)
    snap.add_item(
        Item(
            id=ep.id,
            kind="episode",
            entity_id=entities[0] if entities else None,
            text=res.plain("\n".join(t for t in (ep.gist.strip(), ep.summary.strip()) if t)),
            privacy=privacy,
            importance=ep.importance,
            date=ep.started.isoformat(timespec="minutes"),
            src=(ep.session,),
        )
    )
    return privacy


def add_observation(
    snap: Snapshot, obs: Observation, res: Resolver, episode_privacy: dict[str, str]
) -> None:
    src = tuple(ref_target(s) for s in obs.src)
    mentioned = [res.link(t) for t in wikilink_targets(obs.text)]
    for entity in mentioned:
        snap.links.add((obs.id, entity, "mention"))
    for s in src:
        snap.links.add((obs.id, res.link(s), "source"))
    refs = [*mentioned, *(res.link(s) for s in src)]
    private = any(episode_privacy.get(r) == PRIVATE or res.privacy(r) == PRIVATE for r in refs)
    snap.add_item(
        Item(
            id=obs.id,
            kind="observation",
            entity_id=next((m for m in mentioned if m in res.pages), None),
            text=res.plain(obs.text.strip()),
            privacy=PRIVATE if private else NORMAL,
            importance=obs.importance,
            date=obs.day.isoformat(),
            src=src,
        )
    )


def snapshot(
    vault: VaultSource, private_regions: Collection[str], root: Path | None = None
) -> tuple[Snapshot, ReindexStats]:
    stats = ReindexStats()
    pages = {p.id: p for p in vault.pages()}
    alias_index = vault.alias_index()
    res = Resolver(pages, alias_index)
    snap = Snapshot()
    for page in pages.values():
        add_page(snap, page, res, root)
    ep_privacy: dict[str, str] = {}
    for ep in vault.episodes():
        ep_privacy[ep.id] = add_episode(snap, ep, res, private_regions)
    for obs in vault.observations():
        add_observation(snap, obs, res, ep_privacy)
        stats.observations += 1
    for alias, ids in alias_index.items():
        snap.aliases.update((alias.casefold(), i) for i in ids)
    stats.pages, stats.episodes, stats.facts = len(pages), len(ep_privacy), len(snap.facts)
    stats.items = len(snap.items)
    return snap, stats


# -- sync ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Existing:
    pk: int
    hash: str
    meta: tuple[Any, ...]


def existing_items(db: sqlite3.Connection) -> dict[str, Existing]:
    rows = db.execute(
        "SELECT pk, id, hash, kind, entity_id, privacy, importance, date, src_json FROM items"
    )
    return {r["id"]: Existing(r["pk"], r["hash"], tuple(r)[3:]) for r in rows}


def write_graph(db: sqlite3.Connection, snap: Snapshot) -> None:
    for table in ("entities", "aliases", "facts", "links"):
        db.execute(f"DELETE FROM {table}")
    db.executemany("INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?, ?)", snap.entities)
    db.executemany("INSERT OR IGNORE INTO aliases VALUES (?, ?)", sorted(snap.aliases))
    db.executemany(
        "INSERT OR REPLACE INTO facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", snap.facts
    )
    db.executemany("INSERT OR IGNORE INTO links VALUES (?, ?, ?)", sorted(snap.links))


def write_vector(db: sqlite3.Connection, pk: int, vector: np.ndarray) -> None:
    db.execute("DELETE FROM vec_items WHERE rowid = ?", (pk,))
    db.execute("INSERT INTO vec_items (rowid, embedding) VALUES (?, ?)", (pk, vector_blob(vector)))


def sync_items(
    db: sqlite3.Connection,
    items: dict[str, Item],
    old: dict[str, Existing],
    vectors: dict[str, np.ndarray],
    stats: ReindexStats,
) -> None:
    for item_id in old.keys() - items.keys():
        pk = old[item_id].pk
        db.execute("DELETE FROM items WHERE pk = ?", (pk,))
        db.execute("DELETE FROM vec_items WHERE rowid = ?", (pk,))
        stats.deleted += 1
    for item in items.values():
        values = (item.kind, item.entity_id, item.privacy, item.importance, item.date)
        prev = old.get(item.id)
        if prev is None:
            cur = db.execute(
                "INSERT INTO items (id, kind, entity_id, privacy, importance, date, src_json, "
                "text, hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (item.id, *values, item.src_json, item.text, item.hash),
            )
            pk = int(cur.lastrowid or 0)
            stats.added += 1
        else:
            pk = prev.pk
            if prev.hash != item.hash:
                db.execute(
                    "UPDATE items SET text = ?, hash = ? WHERE pk = ?", (item.text, item.hash, pk)
                )
                stats.changed += 1
            if prev.meta != item.meta():
                db.execute(
                    "UPDATE items SET kind = ?, entity_id = ?, privacy = ?, importance = ?, "
                    "date = ?, src_json = ? WHERE pk = ?",
                    (*values, item.src_json, pk),
                )
                if prev.hash == item.hash:
                    stats.updated += 1
        if item.id in vectors:
            write_vector(db, pk, vectors[item.id])


def reindex(
    vault: VaultSource,
    db: sqlite3.Connection,
    llm: RetrievalModel,
    *,
    full: bool = False,
    private_regions: Collection[str] | None = None,
) -> ReindexStats:
    """Sync the index with the vault. ``full=True`` re-embeds every item."""
    start = time.perf_counter()
    root = getattr(vault, "root", None)
    root = Path(root) if root is not None else None
    if private_regions is None:
        private_regions = load_private_regions(root)

    reset = ensure_schema(db, embed_model=llm.embed_model, dim=llm.embed_dim)
    snap, stats = snapshot(vault, private_regions, root)
    stats.vectors_reset = reset

    old = existing_items(db)
    with_vector = {r[0] for r in db.execute("SELECT rowid FROM vec_items")}
    todo = [
        item
        for item in snap.items.values()
        if full
        or item.id not in old
        or old[item.id].hash != item.hash
        or old[item.id].pk not in with_vector
    ]
    vectors: dict[str, np.ndarray] = {}
    if todo:
        embedded = llm.embed([item.text for item in todo])
        vectors = {item.id: embedded[i] for i, item in enumerate(todo)}
    stats.embedded = len(todo)

    with transaction(db):
        write_graph(db, snap)
        sync_items(db, snap.items, old, vectors, stats)

    stats.seconds = time.perf_counter() - start
    log.info("reindex: %s", stats)
    return stats
