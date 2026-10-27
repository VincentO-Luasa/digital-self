"""Which region to interview about, and what to ask there, in order (no model call).

Region choice: ``argmax(weight × (1 − coverage) + boost)``, where the boost counts open
questions.md items added for that region by other components or the owner (coverage items are
already in the coverage term): ``BOOST_PER_ITEM × priority / 5`` each, capped at ``BOOST_CAP``.
Private regions (life map ``privacy: private``) are chosen only when the owner names one
explicitly (``brain interview relationships.friends`` or ``friends``); naming a parent
(``relationships``) picks among its non-private children (docs/decisions.md).

The plan for a region: questions.md items the owner or other components added first, then the
slot gaps, stale ones first (confirming what is known is the easiest opener), then empty, then
partial ones; per-entity regions take the least covered entities first (at most
``MAX_ENTITIES``). It ends with an open "anything else?" step, which finds new people or projects
in per-entity regions.

Lint doubts (items with a ``doubt::`` key) are grouped by entity: all of an entity's doubts form
one block, and its ``type`` and ``part_of`` doubts become ONE question ("What is stream
processing to you: a project (at Fabrikam Analytics, or on your own), or a topic or tool you
work with?"), built from ``EntityInfo`` (``brain.interview.doubts``); without it each item's own
text is asked. At most ``MAX_DOUBT_BLOCKS`` blocks per session, interleaved one for one with the
coverage steps, so an interview after ``brain lint`` still learns new things.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from brain.interview.coverage import (
    CoverageReport,
    KnownFact,
    RegionCoverage,
    SlotCoverage,
    region_facts,
)
from brain.interview.doubts import Doubt, EntityInfo, combined_hint, merge_hint
from brain.interview.questions import COVERAGE, GapItem, discover_hint, doubt_key
from brain.vault.lifemap import LifeMap

BOOST_PER_ITEM = 0.1
BOOST_CAP = 0.3
MAX_ENTITIES = 3
MAX_KNOWN = 3
MAX_DOUBT_BLOCKS = 4  # entities (blocks of lint doubts) asked about per session
STATUS_ORDER = {"stale": 0, "empty": 1, "partial": 2}

Mode = Literal["item", "confirm", "ask", "more", "discover"]


class RegionError(ValueError):
    """The requested region is unknown, or names only private regions implicitly."""


@dataclass(frozen=True)
class RegionScore:
    id: str
    label: str
    weight: float
    coverage: float
    boost: float
    private: bool

    @property
    def score(self) -> float:
        return self.weight * (1.0 - self.coverage) + self.boost


@dataclass(frozen=True)
class PlanItem:
    region: str
    mode: Mode
    hint: str
    topic: str
    known: tuple[str, ...] = ()  # fact lines about exactly this (prompt and grounding)
    cite: str | None = None  # the fact a confirm question cites (fallback template)
    entity: str | None = None
    slot: str | None = None
    question_ids: tuple[str, ...] = ()  # questions.md items this asks (ticked when answered)
    doubts: tuple[str, ...] = ()  # lint finding keys it asks about (settled at close)
    brief: tuple[str, ...] = ()  # what the vault has on its entities, with sources
    block: str | None = None  # the entity block of lint doubts it belongs to

    @property
    def key(self) -> str:
        return "/".join(p for p in (self.region, self.slot or self.mode, self.entity) if p)

    @property
    def question_id(self) -> str | None:
        return self.question_ids[0] if self.question_ids else None

    @property
    def is_doubt(self) -> bool:
        return bool(self.doubts)


# --- regions -----------------------------------------------------------------------------------


def region_scores(report: CoverageReport, items: Sequence[GapItem]) -> list[RegionScore]:
    boosts: dict[str, float] = {}
    for item in items:
        if item.open and item.origin != COVERAGE and item.region in report.regions:
            boosts[item.region] = boosts.get(item.region, 0.0) + BOOST_PER_ITEM * item.priority / 5
    return [
        RegionScore(
            id=r.id,
            label=r.region.label or r.id,
            weight=r.region.weight,
            coverage=r.coverage,
            boost=min(BOOST_CAP, boosts.get(r.id, 0.0)),
            private=r.private,
        )
        for r in report.regions.values()
    ]


def resolve_region(lifemap: LifeMap, name: str) -> list[str]:
    """Region ids named by ``name``: an id, a label, a unique last part (``friends``) or a
    parent (``relationships`` → its children)."""
    want = name.strip().casefold()
    ids = list(lifemap.regions)
    exact = [r for r in ids if r.casefold() == want]
    if exact:
        return exact
    by_label = [r for r in ids if (lifemap.regions[r].label or "").casefold() == want]
    if len(by_label) == 1:
        return by_label
    by_tail = [r for r in ids if r.casefold().rsplit(".", 1)[-1] == want]
    if len(by_tail) == 1:
        return by_tail
    children = [r for r in ids if r.casefold().startswith(want + ".")]
    if children:
        return children
    raise RegionError(f"unknown region {name!r}; regions: {', '.join(ids)}")


def choose_region(
    report: CoverageReport, items: Sequence[GapItem], requested: str | None = None
) -> RegionScore | None:
    """The region to interview about, or None when nothing (non-private) is left to ask."""
    scores = region_scores(report, items)
    by_id = {s.id: s for s in scores}
    if requested:
        ids = resolve_region(report.lifemap, requested)
        if len(ids) == 1:
            return by_id[ids[0]]  # named explicitly: private regions allowed
        candidates = [by_id[r] for r in ids if not by_id[r].private]
        if not candidates:
            raise RegionError(f"{requested!r} has only private regions; name one: {', '.join(ids)}")
    else:
        candidates = [s for s in scores if not s.private]
    best = max(candidates, key=lambda s: s.score, default=None)  # first wins ties
    if best is None or (not requested and best.score <= 0):
        return None
    return best


# --- plan --------------------------------------------------------------------------------------


def _coverage_id(region: str, slot: str, entity: str | None) -> str:
    return GapItem(text="", region=region, slot=slot, about=entity, origin=COVERAGE).id


def _lines(facts: tuple[KnownFact, ...] | list[KnownFact]) -> tuple[str, ...]:
    return tuple(f.line() for f in list(facts)[:MAX_KNOWN])


def _slot_step(s: SlotCoverage, topic: str) -> PlanItem:
    mode: Mode = {"stale": "confirm", "partial": "more"}.get(s.status, "ask")  # type: ignore[assignment]
    hint = s.ask
    if mode == "more":  # ask for what is missing, not to rank what is known
        hint = f"{s.ask.rstrip('?. ')}, besides what I already know?"
    return PlanItem(
        region=s.region,
        mode=mode,
        hint=hint,
        topic=topic,
        known=_lines(s.facts),
        cite=s.facts[0].text if s.facts else None,
        entity=s.entity,
        slot=s.slot.id,
    )


def _gaps(slots: tuple[SlotCoverage, ...], dismissed: set[str]) -> list[SlotCoverage]:
    gaps = [
        s
        for s in slots
        if s.status != "filled" and _coverage_id(s.region, s.slot.id, s.entity) not in dismissed
    ]
    return sorted(gaps, key=lambda s: STATUS_ORDER[s.status])  # stable: life-map slot order


def _entity_facts(
    report: CoverageReport, entity: str, *, fresh_only: bool = False
) -> list[KnownFact]:
    """Facts on the entity's slots (``fresh_only``: of filled or partial slots: stale ones
    are asked about in their own step and would pull a question towards them)."""
    seen: dict[str, KnownFact] = {}
    for region in report.regions.values():
        for s in region.slots:
            if s.entity == entity and (not fresh_only or s.status in ("filled", "partial")):
                for f in s.facts:
                    seen.setdefault(f.id, f)
    return list(seen.values())


def _added_step(
    report: CoverageReport, region_id: str, label: str, items: Sequence[GapItem], hint: str
) -> PlanItem:
    about = next((i.about for i in items if i.about), None)
    facts = _entity_facts(report, about) if about else []
    return PlanItem(
        region=region_id,
        mode="item",
        hint=hint,
        topic=label,
        known=_lines(facts),
        entity=about,
        slot=f"q/{items[0].id}",
        question_ids=tuple(i.id for i in items),
    )


def _by_entity(items: Sequence[GapItem]) -> list[list[GapItem]]:
    """Items in order, those about one entity together (at the place of the first)."""
    blocks: dict[str, list[GapItem]] = {}
    for item in items:
        blocks.setdefault(item.about or f"#{item.id}", []).append(item)
    return list(blocks.values())


def _doubt_blocks(items: Sequence[GapItem]) -> list[list[GapItem]]:
    """Lint items grouped by the entities their doubts name (a merge joins both pages' doubts),
    highest priority first."""
    blocks: list[tuple[set[str], list[GapItem]]] = []
    for item in sorted(items, key=lambda i: -i.priority):
        doubt = Doubt.parse(doubt_key(item))
        ents = set(doubt.entities if doubt else ()) | ({item.about} if item.about else set())
        joined = [b for b in blocks if ents and b[0] & ents]
        if not joined:
            blocks.append((ents, [item]))
            continue
        first = joined[0]
        first[0].update(ents)
        first[1].append(item)
        for other in joined[1:]:
            first[0].update(other[0])
            first[1].extend(other[1])
            blocks.remove(other)
    return [b[1] for b in blocks]


def _doubt_steps(
    report: CoverageReport,
    region_id: str,
    label: str,
    block: list[GapItem],
    infos: Mapping[str, EntityInfo],
) -> list[PlanItem]:
    """One block's questions: an entity's type and part_of doubts as one combined question, a
    merge as one question naming both pages, anything else as written."""
    parsed = [(i, Doubt.parse(doubt_key(i))) for i in block]
    ents = list(dict.fromkeys(e for _, d in parsed if d for e in d.entities))
    name = ents[0] if ents else block[0].about or block[0].id
    brief = tuple(infos[e].brief() for e in ents if e in infos)
    steps: list[PlanItem] = []

    def step(items: list[GapItem], hint: str, entity: str | None) -> None:
        base = _added_step(report, region_id, label, items, hint)
        keys = tuple(k for i in items if (k := doubt_key(i)))
        steps.append(
            PlanItem(
                **{
                    **base.__dict__,
                    "entity": entity or base.entity,
                    "doubts": keys,
                    "brief": brief,
                    "block": name,
                }
            )
        )

    done: set[str] = set()
    for entity in ents:  # type + part_of on one page: one question
        mine = [i for i, d in parsed if d and d.kind in ("type", "part_of") and d.ids[0] == entity]
        info = infos.get(entity)
        hint = combined_hint(info, {d.kind for i, d in parsed if i in mine and d}) if info else None
        if mine and hint:
            step(mine, hint, entity)
            done |= {i.id for i in mine}
    for item, doubt in parsed:
        if item.id in done:
            continue
        hint = item.text
        if doubt and doubt.kind == "merge" and all(e in infos for e in doubt.ids[:2]):
            hint = merge_hint(infos[doubt.ids[0]], infos[doubt.ids[1]])
        entity = doubt.entities[0] if doubt and doubt.entities else None
        step([item], hint, entity or item.about)
    return steps


def _interleave(doubts: list[list[PlanItem]], coverage: list[list[PlanItem]]) -> list[PlanItem]:
    out: list[PlanItem] = []
    for n in range(max(len(doubts), len(coverage))):
        for blocks in (doubts, coverage):
            if n < len(blocks):
                out += blocks[n]
    return out


def build_plan(
    report: CoverageReport,
    region_id: str,
    items: Sequence[GapItem] = (),
    *,
    max_entities: int = MAX_ENTITIES,
    infos: Mapping[str, EntityInfo] | None = None,
    max_doubts: int = MAX_DOUBT_BLOCKS,
) -> list[PlanItem]:
    region: RegionCoverage = report.regions[region_id]
    label = region.region.label or region_id
    dismissed = {i.id for i in items if i.origin == COVERAGE and i.done}
    infos = infos or {}
    plan: list[PlanItem] = []

    added = [i for i in items if i.open and i.origin != COVERAGE and i.region == region_id]
    plain = sorted((i for i in added if not doubt_key(i)), key=lambda i: -i.priority)
    for group in _by_entity(plain):
        plan += [_added_step(report, region_id, label, [i], i.text) for i in group]
    chosen = _doubt_blocks([i for i in added if doubt_key(i)])[:max_doubts]
    doubt_blocks = [_doubt_steps(report, region_id, label, b, infos) for b in chosen]
    # a page that may be demoted, retyped or merged away gets no coverage questions this time
    held = {
        e
        for block in chosen
        for i in block
        if (d := Doubt.parse(doubt_key(i))) and d.kind in ("type", "merge")
        for e in d.entities
    }

    coverage: list[list[PlanItem]] = []
    if not region.per_entity:
        coverage = [[_slot_step(s, label)] for s in _gaps(region.slots, dismissed)]
        hint = f"Is there anything else about your {label.casefold()} you'd like me to remember?"
        last = PlanItem(region_id, "discover", hint, label, slot="more")
        return plan + _interleave(doubt_blocks, coverage) + [last]

    names = tuple(e.name for e in region.entities)
    if not region.entities and _coverage_id(region_id, "discover", None) not in dismissed:
        hint = discover_hint(region_id, label, region.per_entity, more=False)
        coverage.append([PlanItem(region_id, "discover", hint, label, slot="discover")])
    asked = 0
    for entity in region.entities:
        gaps = _gaps(entity.slots, dismissed)
        if not gaps or entity.id in held:
            continue
        if asked >= max_entities:
            break
        asked += 1
        coverage.append([_slot_step(s, f"{label}: {entity.name}") for s in gaps])
    plan += _interleave(doubt_blocks, coverage)
    if region.entities:
        hint = discover_hint(region_id, label, region.per_entity, more=True)
        known = (f"Already known: {', '.join(names[:8])}",)
        plan.append(PlanItem(region_id, "discover", hint, label, known=known, slot="more"))
    return plan


def context_lines(report: CoverageReport, item: PlanItem, *, limit: int = 3) -> tuple[str, ...]:
    """A few other facts (of the entity, else of the region), so a question can start from
    what is known."""
    known = set(item.known)
    if item.entity:
        facts = [f for f in _entity_facts(report, item.entity, fresh_only=True) if f.current]
        facts.sort(key=lambda f: (f.confirmed, f.id), reverse=True)
    else:
        facts = region_facts(report, item.region, limit=limit + len(known))
    return tuple(line for line in (f.line() for f in facts) if line not in known)[:limit]
