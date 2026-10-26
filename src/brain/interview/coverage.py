"""Coverage of the life map: how much the brain knows about each region (docs/architecture.md,
Flow 2). Computed from the index's ``facts`` and ``entities`` tables, no model call.

A slot is filled by facts with its predicate about its subject (``self``, or one entity of a
per-entity region). Two kinds of slots:

- **present** slots (with ``stale_days``) describe the owner's life now: only current facts
  valid today count, and a fact not recorded or confirmed within ``stale_days`` is *stale*;
- **timeless** slots (no ``stale_days``: hometown, schools, past jobs, how you met someone):
  history counts too, since a fact that ended is still known.

"Confirmed" is the latest of the fact's ``rec::`` and the dates in its sources (an episode
``ep-YYYYMMDD-…`` that confirmed the fact is added to ``src::``).

Slot score: ``min(fresh distinct values / needed, 1)`` (needed = ``min`` for multi slots, else
1), so a multi slot half-filled scores 0.5; stale facts score 0. Region coverage is the mean slot
score; a per-entity region averages its entities (each scored over its slots) and is 0 when it
has no entity yet (the interview then asks who or what there is).
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from brain.index.temporal import valid_at
from brain.vault.generated import DONE_STATUS
from brain.vault.lifemap import LifeMap, Region, Slot
from brain.vault.models import RELATIONSHIP_PREDICATES, SELF_ID

Status = Literal["filled", "partial", "stale", "empty"]

_EPISODE_DATE = re.compile(r"\b(?:ep|conv|r)-(\d{4})(\d{2})(\d{2})\b")
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


@dataclass(frozen=True)
class KnownFact:
    id: str
    subject: str
    p: str
    o: str
    text: str
    valid_from: str | None
    valid_to: str | None
    rec: date
    confirmed: date
    current: bool

    def line(self) -> str:
        """``Atlas is in the testing phase (since 2026-10; noted 2026-10-25)`` for prompts."""
        when = []
        if self.valid_from and self.valid_to:
            when.append(f"{self.valid_from} to {self.valid_to}")
        elif self.valid_from:
            when.append(f"since {self.valid_from}")
        elif self.valid_to:
            when.append(f"until {self.valid_to}")
        when.append(f"noted {self.confirmed.isoformat()}")
        return f"{self.text} ({'; '.join(when)})"


@dataclass(frozen=True)
class SlotCoverage:
    region: str
    slot: Slot
    status: Status
    score: float
    facts: tuple[KnownFact, ...] = ()  # the facts considered (fresh and stale)
    entity: str | None = None
    entity_name: str | None = None

    @property
    def key(self) -> str:
        """``projects/status/projects/atlas-migration``: stable id of this gap."""
        return "/".join([self.region, self.slot.id] + ([self.entity] if self.entity else []))

    @property
    def ask(self) -> str:
        return self.slot.ask.replace("{name}", self.entity_name or "them")

    @property
    def needed(self) -> int:
        return max(self.slot.min, 1) if self.slot.multi else 1


@dataclass(frozen=True)
class EntityCoverage:
    id: str
    name: str
    coverage: float
    slots: tuple[SlotCoverage, ...]
    mentions: int = 0  # facts about or pointing at the entity (salience for ordering)
    private: bool = False


@dataclass(frozen=True)
class RegionCoverage:
    region: Region
    coverage: float
    slots: tuple[SlotCoverage, ...] = ()  # every slot (per-entity regions: of every entity)
    entities: tuple[EntityCoverage, ...] = ()

    @property
    def id(self) -> str:
        return self.region.id

    @property
    def private(self) -> bool:
        return self.region.privacy == "private"

    @property
    def per_entity(self) -> str | None:
        return self.region.per_entity

    @property
    def gaps(self) -> list[SlotCoverage]:
        return [s for s in self.slots if s.status != "filled"]

    def counts(self) -> dict[Status, int]:
        out: dict[Status, int] = {"filled": 0, "partial": 0, "stale": 0, "empty": 0}
        for s in self.slots:
            out[s.status] += 1
        return out


@dataclass(frozen=True)
class Unplaced:
    """An entity of a per-entity type (a person) that no region claims yet."""

    id: str
    name: str
    type: str
    private: bool = False


@dataclass
class CoverageReport:
    today: date
    lifemap: LifeMap
    regions: dict[str, RegionCoverage] = field(default_factory=dict)
    unplaced: list[Unplaced] = field(default_factory=list)

    def region(self, region_id: str) -> RegionCoverage | None:
        return self.regions.get(region_id)

    def overall(self, *, include_private: bool = True) -> float:
        """Weighted mean coverage over regions."""
        rows = [r for r in self.regions.values() if include_private or not r.private]
        total = sum(r.region.weight for r in rows)
        return sum(r.region.weight * r.coverage for r in rows) / total if total else 0.0


# --- reading the index -------------------------------------------------------------------------


def confirmed_date(rec: date, sources: Iterable[str], today: date) -> date:
    """Latest of ``rec`` and the dates named by the sources, never after today."""
    best = rec
    for src in sources:
        for m in (*_EPISODE_DATE.finditer(src), *_ISO_DATE.finditer(src)):
            try:
                d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                continue
            if best < d <= today:
                best = d
    return best


def _parse_date(text: str) -> date:
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return date.min


def load_facts(db: sqlite3.Connection, today: date) -> list[KnownFact]:
    rows = db.execute(
        "SELECT id, subject, p, o, o_entity, text, valid_from, valid_to, rec, src_json, current "
        "FROM facts ORDER BY page_id, id"
    ).fetchall()
    out = []
    for r in rows:
        rec = _parse_date(r["rec"])
        try:
            sources = [str(s) for s in json.loads(r["src_json"] or "[]")]
        except ValueError:
            sources = []
        out.append(
            KnownFact(
                id=r["id"],
                subject=r["subject"],
                p=r["p"],
                o=r["o_entity"] or r["o"],
                text=" ".join(str(r["text"]).split()),
                valid_from=r["valid_from"],
                valid_to=r["valid_to"],
                rec=rec,
                confirmed=confirmed_date(rec, sources, today),
                current=bool(r["current"]),
            )
        )
    return out


@dataclass(frozen=True)
class _Entity:
    id: str
    type: str
    title: str
    regions: tuple[str, ...]
    private: bool = False


def load_entities(db: sqlite3.Connection) -> list[_Entity]:
    out = []
    query = "SELECT subject, type, title, privacy, regions_json FROM entities ORDER BY id"
    for r in db.execute(query):
        try:
            regions = tuple(str(x) for x in json.loads(r["regions_json"] or "[]"))
        except ValueError:
            regions = ()
        if r["subject"] == SELF_ID:
            continue
        private = r["privacy"] == "private"
        out.append(_Entity(r["subject"], r["type"], r["title"], regions, private))
    # one entity may have several pages (same subject): keep the first title, merge regions
    merged: dict[str, _Entity] = {}
    for e in out:
        if e.id in merged:
            old = merged[e.id]
            merged[e.id] = _Entity(
                old.id, old.type, old.title, old.regions + e.regions, old.private or e.private
            )
        else:
            merged[e.id] = e
    return list(merged.values())


# --- scoring -----------------------------------------------------------------------------------


class _Facts:
    def __init__(self, facts: list[KnownFact]) -> None:
        self.by_subject: dict[tuple[str, str], list[KnownFact]] = {}
        self.by_object: dict[tuple[str, str], list[KnownFact]] = {}
        self.mentions: dict[str, int] = {}
        for f in facts:
            self.by_subject.setdefault((f.subject, f.p), []).append(f)
            self.by_object.setdefault((f.o, f.p), []).append(f)
            self.mentions[f.subject] = self.mentions.get(f.subject, 0) + 1
            if f.o != f.subject:
                self.mentions[f.o] = self.mentions.get(f.o, 0) + 1

    def about(self, subject: str, p: str) -> list[KnownFact]:
        """Facts with predicate ``p`` about ``subject``. A relationship stated on the owner's
        page (``self friend_of [[people/x]]``) counts for the person as well."""
        out = list(self.by_subject.get((subject, p), []))
        if subject != SELF_ID and p in RELATIONSHIP_PREDICATES:
            out += [f for f in self.by_object.get((subject, p), []) if f.subject == SELF_ID]
        return out


def score_slot(
    region: str,
    slot: Slot,
    facts: list[KnownFact],
    today: date,
    *,
    entity: str | None = None,
    entity_name: str | None = None,
) -> SlotCoverage:
    if slot.stale_days is not None:
        considered = [f for f in facts if f.current and valid_at(f.valid_from, f.valid_to, today)]
        fresh = [f for f in considered if (today - f.confirmed).days <= slot.stale_days]
    else:
        considered = list(facts)
        fresh = considered
    needed = max(slot.min, 1) if slot.multi else 1
    values = {f.o.casefold() for f in fresh}
    score = min(len(values) / needed, 1.0)
    if len(values) >= needed:
        status: Status = "filled"
    elif values:
        status = "partial"
    elif considered:
        status = "stale"
    else:
        status = "empty"
    considered.sort(key=lambda f: (not f.current, -f.confirmed.toordinal(), f.id))
    return SlotCoverage(
        region, slot, status, score, tuple(considered), entity=entity, entity_name=entity_name
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _closed(entity: str, facts: _Facts) -> bool:
    """A project (or anything) whose current status says it is over is not asked about."""
    return any(
        f.current and f.o.strip().casefold() in DONE_STATUS for f in facts.about(entity, "status")
    )


def _member(entity: _Entity, region_id: str) -> bool:
    return any(r == region_id or r.startswith(region_id + ".") for r in entity.regions)


def place_entities(
    lifemap: LifeMap, entities: list[_Entity], facts: _Facts
) -> tuple[dict[str, list[_Entity]], list[Unplaced]]:
    """Per-entity region → its entities. An entity belongs to the regions its page lists; else
    to the region whose slot predicates it has facts for; else to the only region of its type;
    otherwise it is unplaced (e.g. a person whose relation to the owner is unknown)."""
    per_type: dict[str, list[Region]] = {}
    for region in lifemap.regions.values():
        if region.per_entity:
            per_type.setdefault(region.per_entity, []).append(region)
    placed: dict[str, list[_Entity]] = {r.id: [] for rs in per_type.values() for r in rs}
    unplaced: list[Unplaced] = []
    for e in entities:
        candidates = per_type.get(e.type, [])
        if not candidates:
            continue
        home = [r for r in candidates if _member(e, r.id)]
        if not home:
            home = [r for r in candidates if any(facts.about(e.id, s.p) for s in r.slots)][:1]
        if not home and len(candidates) == 1:
            home = candidates
        if not home:
            unplaced.append(Unplaced(e.id, e.title, e.type, e.private))
        for r in home:
            placed[r.id].append(e)
    return placed, unplaced


def coverage_report(
    db: sqlite3.Connection, lifemap: LifeMap, today: date | None = None
) -> CoverageReport:
    today = today or date.today()
    facts = _Facts(load_facts(db, today))
    placed, unplaced = place_entities(lifemap, load_entities(db), facts)
    report = CoverageReport(today, lifemap, unplaced=unplaced)
    for region in lifemap.regions.values():
        if not region.per_entity:
            slots = tuple(
                score_slot(region.id, s, facts.about(SELF_ID, s.p), today) for s in region.slots
            )
            report.regions[region.id] = RegionCoverage(
                region, _mean([s.score for s in slots]), slots
            )
            continue
        scored = []
        for e in placed.get(region.id, []):
            if _closed(e.id, facts):
                continue
            slots = tuple(
                score_slot(
                    region.id, s, facts.about(e.id, s.p), today, entity=e.id, entity_name=e.title
                )
                for s in region.slots
            )
            cov = _mean([s.score for s in slots])
            mentions = facts.mentions.get(e.id, 0)
            scored.append(EntityCoverage(e.id, e.title, cov, slots, mentions, e.private))
        # least covered first; among equals the most mentioned (most salient) first
        scored.sort(key=lambda e: (e.coverage, -e.mentions, e.id))
        report.regions[region.id] = RegionCoverage(
            region,
            _mean([e.coverage for e in scored]),
            tuple(s for e in scored for s in e.slots),
            tuple(scored),
        )
    return report


def region_facts(report: CoverageReport, region_id: str, *, limit: int = 4) -> list[KnownFact]:
    """Fresh facts already known in a region (context for opening questions), newest first."""
    region = report.regions.get(region_id)
    if not region:
        return []
    seen: dict[str, KnownFact] = {}
    for s in region.slots:
        for f in s.facts:
            if s.status in ("filled", "partial") and f.current:
                seen.setdefault(f.id, f)
    facts = sorted(seen.values(), key=lambda f: (f.confirmed, f.id), reverse=True)
    return facts[:limit]
