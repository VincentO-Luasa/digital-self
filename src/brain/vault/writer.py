"""The single writer: validate operations, apply the safe ones, queue the rest for review,
append to log.md and commit the vault (one commit per change set).

Owner corrections: a ``supersede_fact`` whose fact id the caller lists in ``corrections`` (the
memory pipeline, after a typed decision that the owner explicitly corrected that fact) is applied
instead of queued, but only when ``correction_problem`` finds nothing: origin ``extract``, a fact
about the owner, no relationship predicate, no private page or region, a ``stated`` replacement
with the same predicate. Everything else stays in the review queue (docs/vault-schema.md,
Reviews).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from pydantic import TypeAdapter

from brain.vault import gitops
from brain.vault.models import (
    PREDICATES,
    RELATIONSHIP_PREDICATES,
    SELF_ID,
    Episode,
    Fact,
    Intention,
    Observation,
    Page,
    Privacy,
    fact_private,
    is_multi_valued,
    link_target,
    new_id,
)
from brain.vault.ops import (
    AddAlias,
    AddEpisode,
    AddFact,
    AddIntention,
    AddObservation,
    ChangeSet,
    ConfirmFact,
    CreateEntity,
    FactSpec,
    MergeEntities,
    Operation,
    Origin,
    SupersedeFact,
    UpdateEpisode,
    UpdateSummary,
)
from brain.vault.parse import (
    DIR_TYPES,
    parse_fact_line,
    parse_intention_line,
    parse_observation_line,
    split_frontmatter,
)
from brain.vault.render import (
    render_episode,
    render_fact,
    render_frontmatter,
    render_intention,
    render_journal,
    render_observation,
    render_page,
)
from brain.vault.store import SELF_PAGE, Vault

TYPE_DIRS = {t: d for d, t in DIR_TYPES.items()}
ENTITY_ID = re.compile(r"^[a-z]+/[a-z0-9][a-z0-9._-]*$")
EPISODE_ID = re.compile(r"^ep-\d{8}-\d{4}-[a-z0-9][a-z0-9-]*$")
EPISODE_OPS = (AddEpisode, UpdateEpisode)  # written by the memory pipeline, never proposed
SECTION_HEADING = re.compile(r"^##\s+\S")  # as parse._parse_page_body sees it
REVIEW_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
SELF_REGION_PAGES = {"self.values_goals": "self/goals"}  # other self.* regions → self/<name>
_OPS = TypeAdapter(list[Operation])


class ReviewNotFound(LookupError):
    pass


class _Skip(Exception):  # noqa: N818 - control flow, not an error
    pass


class _Depends:
    """Validation outcome: the op targets an entity whose creation was queued."""


DEPENDS = _Depends()


@dataclass
class ApplyResult:
    applied: list[Operation] = field(default_factory=list)
    queued: list[tuple[Operation, str]] = field(default_factory=list)
    skipped: list[tuple[Operation, str]] = field(default_factory=list)
    review_id: str | None = None
    facts_added: int = 0
    superseded: int = 0
    confirmed: int = 0  # facts whose src:: gained this change set's sources
    corrected: list[str] = field(default_factory=list)  # fact ids superseded as owner corrections
    episodes: list[str] = field(default_factory=list)  # episode ids written or updated
    created: list[str] = field(default_factory=list)  # new page ids
    new_ids: list[str] = field(default_factory=list)  # new fact / observation / intention ids
    changed: list[Path] = field(default_factory=list)
    commit: str | None = None


@dataclass
class Review:
    id: str
    origin: Origin
    created: datetime
    src: list[str]
    rationale: str
    status: str
    ops: list[Operation]
    path: Path


# --- working state -----------------------------------------------------------------------------


class _State:
    """A mutable copy of the parts of the vault a change set may touch."""

    def __init__(self, vault: Vault, now: datetime) -> None:
        self.vault = vault
        self.now = now
        self.today = now.date()
        self.lifemap = vault.lifemap
        self.pages: dict[str, Page] = {p.id: p.model_copy(deep=True) for p in vault.pages()}
        self.dirty: set[str] = set()
        self.deleted: set[str] = set()
        self.queued_entities: set[str] = set()
        self.used_ids = {f.id for p in self.pages.values() for f in (*p.facts, *p.history)}
        self.used_ids |= {o.id for o in vault.observations()}
        self.used_ids |= {i.id for i in vault.intentions()}
        self.intention_lines: list[str] = []
        self.journal_lines: dict[str, list[str]] = {}
        self.text_files: dict[Path, str] = {}
        self.episodes: dict[str, Episode] = {
            e.id: e.model_copy(deep=True) for e in vault.episodes()
        }
        self.episode_paths: dict[str, Path] = {
            e.id: vault.episode_path(e) for e in self.episodes.values()
        }
        self.dirty_episodes: set[str] = set()

    def new_id(self, prefix: str) -> str:
        while (candidate := new_id(prefix)) in self.used_ids:
            pass
        self.used_ids.add(candidate)
        return candidate

    def page(self, entity: str) -> Page | None:
        return self.pages.get(Vault.page_id_for(_entity(entity)))

    def touch(self, page: Page) -> None:
        page.updated = self.today
        self.dirty.add(page.id)

    def facts_about(self, subject: str) -> Iterator[Fact]:
        for page in self.pages.values():
            if page.subject == subject:
                yield from page.facts

    def find_fact(self, fact_id: str) -> tuple[Page, Fact, bool] | None:
        for page in self.pages.values():
            for in_history, facts in ((False, page.facts), (True, page.history)):
                for fact in facts:
                    if fact.id == fact_id:
                        return page, fact, in_history
        return None

    def read_text(self, path: Path) -> str | None:
        if path in self.text_files:
            return self.text_files[path]
        return path.read_text(encoding="utf-8") if path.exists() else None


def _entity(value: str) -> str:
    return link_target(value) or value.strip()


def _clean(text: str) -> str:
    return " ".join(text.split()).replace("::", ":")


def _norm(value: str) -> str:
    return " ".join(_entity(value).casefold().split())


def _self_page_id(state: _State, p: str) -> tuple[str, str | None]:
    """Facts about the self land on the self page of the predicate's life-map region."""
    region = state.lifemap.region_for(p, SELF_ID)
    if region and region.startswith("self."):
        default = "self/" + region.removeprefix("self.").replace("_", "-")
        return SELF_REGION_PAGES.get(region, default), region
    return SELF_PAGE, region


def _fact_page_id(state: _State, op: AddFact) -> tuple[str, str | None]:
    subject = _entity(op.subject)
    if subject in (SELF_ID, SELF_PAGE):
        return _self_page_id(state, op.p)
    return subject, None


def _page_privacy(state: _State, page_id: str, region: str | None = None) -> Privacy:
    if page := state.pages.get(page_id):
        return page.privacy
    return state.lifemap.privacy_for([region] if region else [])


# --- validation and policy ---------------------------------------------------------------------


def _lo(d: str | None) -> str:
    return d or "0000"


def _hi(d: str | None) -> str:
    return (d + "~") if d else "9999~"  # "~" sorts after digits and "-": "2024~" > "2024-12-31"


def _ends_by(end: str | None, start: str | None) -> bool:
    """True if validity ending at ``end`` is over by ``start``. Equal dates are a handover
    (supersession writes the old ``to::`` as the new ``from::``), not an overlap."""
    return bool(end and start) and (end == start or _hi(end) < _lo(start))


def _overlaps(a: Fact | FactSpec, b: Fact | FactSpec) -> bool:
    return not _ends_by(a.valid_to, b.valid_from) and not _ends_by(b.valid_to, a.valid_from)


def overlaps(a: Fact | FactSpec, b: Fact | FactSpec) -> bool:
    """Valid times overlap (back-to-back periods hand over and do not)."""
    return _overlaps(a, b)


def _date_problem(*dates: str | None) -> str | None:
    """Partial dates must be real calendar dates (the models only check the shape)."""
    for d in dates:
        if not d:
            continue
        parts = [int(x) for x in d.split("-")]
        try:
            date(parts[0], parts[1] if len(parts) > 1 else 1, parts[2] if len(parts) > 2 else 1)
        except ValueError:
            return f"{d} is not a valid date"
    return None


def _order_problem(valid_from: str | None, valid_to: str | None) -> str | None:
    if valid_from and valid_to and _hi(valid_to) < _lo(valid_from):
        return f"valid time ends before it starts ({valid_from} → {valid_to})"
    return None


def _window_problem(valid_from: str | None, valid_to: str | None) -> str | None:
    return _date_problem(valid_from, valid_to) or _order_problem(valid_from, valid_to)


def _subject(op: AddFact) -> str:
    subject = _entity(op.subject)
    return SELF_ID if subject == SELF_PAGE else subject


def _adds_nothing(new: str | None, old: str | None) -> bool:
    """A bound says nothing new if it is unknown, equal, or a coarser form of the old one."""
    return new is None or (old is not None and old.startswith(new))


def duplicate_in(facts: Iterable[Fact], spec: FactSpec) -> Fact | None:
    """Of ``facts`` (one subject's), one with the same p and o that already says everything
    ``spec`` says (its dates are missing, equal or coarser)."""
    for f in facts:
        if f.p != spec.p or _norm(f.o) != _norm(spec.o):
            continue
        if _adds_nothing(spec.valid_from, f.valid_from) and _adds_nothing(
            spec.valid_to, f.valid_to
        ):
            return f
    return None


def conflicts_in(facts: Iterable[Fact], subject: str, spec: FactSpec) -> list[Fact]:
    """Scoped contradiction check over ``facts`` (one subject's current facts): same p (+ o for
    multi-valued predicates), with overlapping valid time. Run after ``duplicate_in``: a same-o
    hit here carries new dates."""
    multi = is_multi_valued(spec.p, subject)
    return [
        f
        for f in facts
        if f.p == spec.p and (not multi or _norm(f.o) == _norm(spec.o)) and _overlaps(f, spec)
    ]


def _duplicate(state: _State, subject: str, spec: FactSpec) -> Fact | None:
    """A current fact with the same p and o that already says everything ``spec`` says."""
    return duplicate_in(state.facts_about(subject), spec)


def _conflicts(state: _State, subject: str, spec: FactSpec) -> list[Fact]:
    return conflicts_in(state.facts_about(subject), subject, spec)


def _conflict(state: _State, subject: str, spec: FactSpec) -> Fact | None:
    return next(iter(_conflicts(state, subject, spec)), None)


def _heading_problem(summary: str) -> str | None:
    """A ``## `` line in a summary would open another section on the next parse."""
    if any(SECTION_HEADING.match(line) for line in summary.splitlines()):
        return "a summary may not contain ## section headings"
    return None


def _require_page(state: _State, entity: str) -> _Depends | str | None:
    entity = _entity(entity)
    if state.page(entity):
        return None
    if entity in state.queued_entities:
        return DEPENDS
    return f"unknown entity {entity}"


def _validate(
    op: Operation, state: _State, src: list[str], origin: Origin = "extract"
) -> _Depends | str | None:
    """None if the op can be applied or queued, DEPENDS, or the reason to skip it."""
    if isinstance(op, EPISODE_OPS) and origin == "external":
        return "episodes are written by the memory pipeline only"
    match op:
        case CreateEntity():
            if not ENTITY_ID.match(op.id):
                return f"bad entity id {op.id!r}"
            if op.id.split("/", 1)[0] != TYPE_DIRS[op.type]:
                return f"{op.type} pages live under {TYPE_DIRS[op.type]}/"
            if op.id in state.pages or op.id in state.queued_entities:
                return f"{op.id} already exists"
            return _heading_problem(op.summary)
        case AddFact():
            if not src:
                return "no source, no fact"
            if op.p not in PREDICATES:
                return f"unknown predicate {op.p!r}"
            if problem := _window_problem(op.valid_from, op.valid_to):
                return problem
            if _subject(op) != SELF_ID and (need := _require_page(state, op.subject)):
                return need
            if dup := _duplicate(state, _subject(op), op):
                return f"duplicate of ^{dup.id}"
        case SupersedeFact():
            found = state.find_fact(op.fact_id)
            if not found:
                return f"unknown fact ^{op.fact_id}"
            if found[2] and (found[1].by or not op.new):
                # an ended fact (History, no by::) can still be corrected by a replacement
                return f"^{op.fact_id} is already superseded"
            if problem := _date_problem(op.to) or _order_problem(found[1].valid_from, op.to):
                return problem
            if op.new and op.new.p not in PREDICATES:
                return f"unknown predicate {op.new.p!r}"
            if op.new and not src:
                return "no source, no fact"
            if op.new and (problem := _window_problem(op.new.valid_from, op.new.valid_to)):
                return problem
        case MergeEntities():
            if _entity(op.keep) == _entity(op.merge):
                return "cannot merge an entity into itself"
            for entity in (op.keep, op.merge):
                if need := _require_page(state, entity):
                    return need
                if _entity(entity) == SELF_ID or entity.startswith("self/"):
                    return "self pages cannot be merged"
        case UpdateSummary():
            return _require_page(state, op.id) or _heading_problem(op.summary)
        case AddAlias():
            if not _clean(op.alias):
                return "empty alias"
            if need := _require_page(state, op.id):
                return need
            page = state.page(op.id)
            assert page
            if _clean(op.alias).casefold() in {n.casefold() for n in (page.title, *page.aliases)}:
                return f"{op.alias!r} is already a name of {op.id}"
        case AddIntention():
            if not op.text.strip():
                return "empty intention"
        case AddObservation():
            if not src:
                return "no source for the observation"
        case ConfirmFact():
            found = state.find_fact(op.fact_id)
            if not found:
                return f"unknown fact ^{op.fact_id}"
            if found[2] and found[1].by:
                return f"^{op.fact_id} is superseded"
            if not src or set(src) <= set(found[1].src):
                return f"^{op.fact_id} already cites {', '.join(src) or 'nothing'}"
        case AddEpisode():
            if not EPISODE_ID.match(op.id):
                return f"bad episode id {op.id!r}"
            if op.id in state.episodes:
                return f"episode {op.id} already exists"
            if not op.session.strip() or not op.gist.strip():
                return "an episode needs a session and a gist"
            if op.turns[0] < 1 or op.turns[1] < op.turns[0]:
                return f"bad turn range {list(op.turns)}"
            return _heading_problem(op.summary)
        case UpdateEpisode():
            if op.id not in state.episodes:
                return f"unknown episode {op.id}"
    return None


def _private_target(op: Operation, state: _State) -> str | None:
    """The private page an op touches, if any."""
    targets: list[tuple[str, str | None]] = []
    match op:
        case CreateEntity():
            privacy = op.privacy or state.lifemap.privacy_for(op.regions)
            return op.id if privacy == "private" else None
        case AddFact():
            targets.append(_fact_page_id(state, op))
            if obj := link_target(op.o):
                targets.append((Vault.page_id_for(obj), None))
        case SupersedeFact():
            if found := state.find_fact(op.fact_id):
                targets.append((found[0].id, None))
        case MergeEntities():
            targets += [(Vault.page_id_for(_entity(e)), None) for e in (op.keep, op.merge)]
        case UpdateSummary() | AddAlias():
            targets.append((Vault.page_id_for(_entity(op.id)), None))
        case AddIntention():
            targets += [(Vault.page_id_for(_entity(e)), None) for e in op.when]
    for page_id, region in targets:
        if _page_privacy(state, page_id, region) == "private":
            return page_id
    return None


def _region_private(state: _State, page: Page) -> bool:
    return state.lifemap.privacy_for(page.regions) == "private"


def _correction_problem(op: SupersedeFact, state: _State, origin: Origin) -> str | None:
    """Why a supersession the caller marked as an owner correction may not be auto-applied."""
    if origin != "extract":
        return f"origin {origin}"
    found = state.find_fact(op.fact_id)
    if not found:
        return f"unknown fact ^{op.fact_id}"
    page, old, _ = found
    if page.subject != SELF_ID:
        return f"a fact about {page.subject}, not the owner"
    if fact_private(old) or (op.new is not None and op.new.extra.get("privacy") == "private"):
        return "from an answer the owner marked private"
    if old.p in RELATIONSHIP_PREDICATES:
        return f"relationship predicate {old.p}"
    if page.privacy == "private" or _region_private(state, page):
        return f"private page {page.id}"
    if (region := state.lifemap.region_for(old.p, SELF_ID)) and state.lifemap.privacy_for(
        [region]
    ) == "private":
        return f"private region {region}"
    if op.new is not None:
        if op.new.p != old.p:
            return f"replacement changes the predicate ({old.p} → {op.new.p})"
        if op.new.kind != "stated":
            return f"{op.new.kind} replacement"
        if (obj := link_target(op.new.o)) and (
            _page_privacy(state, Vault.page_id_for(obj)) == "private"
        ):
            return f"points at private page {obj}"
    return None


def _policy(
    op: Operation, state: _State, origin: Origin, corrections: Collection[str] = ()
) -> str | None:
    if origin == "external" and (private := _private_target(op, state)):
        return f"external change to private page {private}"
    match op:
        case AddFact():
            if op.kind == "inferred":
                return "inferred fact"
            if op.p in RELATIONSHIP_PREDICATES:
                return f"relationship predicate {op.p}"
            if clash := _conflict(state, _subject(op), op):
                return f"contradicts current fact ^{clash.id} ({clash.p}:: {clash.o})"
        case SupersedeFact():
            if op.fact_id not in corrections:
                return "supersession"
            if problem := _correction_problem(op, state, origin):
                return f"supersession (not auto-applied as an owner correction: {problem})"
        case MergeEntities():
            return "entity merge"
    return None


def needs_review(
    op: Operation,
    vault: Vault,
    *,
    origin: Origin = "extract",
    corrections: Collection[str] = (),
) -> str | None:
    """Why ``op`` would be queued for review against the current vault, or None."""
    return _policy(op, _State(vault, datetime.now()), origin, corrections)


def correction_problem(
    op: SupersedeFact, vault: Vault, *, origin: Origin = "extract"
) -> str | None:
    """Why ``op`` could not be auto-applied as an owner correction (None: it could)."""
    return _correction_problem(op, _State(vault, datetime.now()), origin)


# --- applying ----------------------------------------------------------------------------------


def _build_fact(state: _State, spec: FactSpec, src: list[str]) -> Fact:
    o = _clean(spec.o)
    if not link_target(o) and (o in state.pages or o == SELF_ID):
        o = f"[[{o}]]"
    fact = Fact(
        id=state.new_id("f"),
        text=_clean(spec.text),
        p=spec.p,
        o=o,
        valid_from=spec.valid_from,
        valid_to=spec.valid_to,
        rec=state.today,
        src=list(src),
        kind=spec.kind,
        evidence=[_clean(e).replace('"', "'") for e in spec.evidence],  # as render_links does
        extra={_clean(k): _clean(v) for k, v in spec.extra.items()},
    )
    try:
        roundtrip = parse_fact_line(render_fact(fact))
    except ValueError as exc:
        raise _Skip(f"cannot be written as a fact line: {exc}") from exc
    if roundtrip != fact:
        raise _Skip("cannot be written as a fact line without loss")
    return fact


def _ensure_fact_page(state: _State, op: AddFact, result: ApplyResult) -> Page:
    page_id, region = _fact_page_id(state, op)
    if page := state.pages.get(page_id):
        return page
    if not page_id.startswith("self/"):
        raise _Skip(f"unknown entity {page_id}")
    label = state.lifemap.regions[region].label if region in state.lifemap.regions else ""
    page = Page(
        id=page_id,
        type="self",
        subject=SELF_ID,
        title=label or page_id.removeprefix("self/").replace("-", " ").title(),
        regions=[region] if region else [],
        privacy=_page_privacy(state, page_id, region),
        created=state.today,
        updated=state.today,
    )
    state.pages[page_id] = page
    state.deleted.discard(page_id)
    result.created.append(page_id)
    return page


def _apply_create(op: CreateEntity, state: _State, result: ApplyResult) -> None:
    aliases: list[str] = []
    for alias in map(_clean, op.aliases):
        if alias and alias.casefold() not in {a.casefold() for a in [op.title, *aliases]}:
            aliases.append(alias)
    page = Page(
        id=op.id,
        type=op.type,
        subject=SELF_ID if op.type == "self" else op.id,
        title=_clean(op.title),
        aliases=aliases,
        regions=list(op.regions),
        privacy=op.privacy or state.lifemap.privacy_for(op.regions),
        created=state.today,
        updated=state.today,
        summary=op.summary.strip(),
    )
    state.pages[op.id] = page
    state.deleted.discard(op.id)  # re-created after a merge earlier in the change set
    state.touch(page)
    result.created.append(op.id)


def _start_after(old: Fact, start: str | None) -> str | None:
    """The replacement's ``from::`` as the old fact's end, if it starts after the old one."""
    return start if start and not _ends_by(start, old.valid_from) else None


def _retire(
    state: _State, page: Page, old: Fact, end: str | None, by: str | None, result: ApplyResult
) -> None:
    """Supersede, never delete: move ``old`` to History with ``to::`` (and ``by::``).
    Without an end date it ended when the brain recorded the change."""
    end = end or old.valid_to or state.today.isoformat()
    page.facts.remove(old)
    page.history.append(old.model_copy(update={"valid_to": end, "by": by}))
    state.touch(page)
    result.superseded += 1


def _apply_add_fact(op: AddFact, state: _State, src: list[str], result: ApplyResult) -> None:
    fact = _build_fact(state, op, src)
    clashes = _conflicts(state, _subject(op), op)
    page = _ensure_fact_page(state, op, result)
    for old in clashes:  # only when applied despite the policy (review=False / approve)
        found = state.find_fact(old.id)
        assert found
        _retire(state, found[0], old, _start_after(old, op.valid_from), fact.id, result)
    page.facts.append(fact)
    state.touch(page)
    result.facts_added += 1
    result.new_ids.append(fact.id)


def _apply_supersede(op: SupersedeFact, state: _State, src: list[str], result: ApplyResult) -> None:
    found = state.find_fact(op.fact_id)
    assert found
    page, old, in_history = found
    new = _build_fact(state, op.new, src) if op.new else None
    if in_history:  # an ended fact corrected: it stays in History, now replaced
        assert new is not None
        index = page.history.index(old)
        page.history[index] = old.model_copy(update={"by": new.id})
        state.touch(page)
        result.superseded += 1
    else:
        end = op.to or old.valid_to or (_start_after(old, op.new.valid_from) if op.new else None)
        _retire(state, page, old, end, new.id if new else None, result)
    if new:
        page.facts.append(new)
        result.facts_added += 1
        result.new_ids.append(new.id)


def _link_pattern(old: str) -> re.Pattern[str]:
    return re.compile(r"\[\[" + re.escape(old) + r"(?=[\]|#^])")


def _repoint_page(page: Page, old: str, new: str) -> bool:
    pattern = _link_pattern(old)

    def sub(text: str) -> str:
        return pattern.sub("[[" + new, text)

    before = page.model_dump()
    for fact in (*page.facts, *page.history):
        fact.o = sub(fact.o)
        fact.extra = {k: sub(v) for k, v in fact.extra.items()}  # at:: [[orgs/…]]
        fact.src = [new if s == old else s for s in fact.src]
        fact.evidence = [new if e == old else e for e in fact.evidence]
    page.summary, page.notes = sub(page.summary), sub(page.notes)
    page.unparsed = {k: [sub(line) for line in v] for k, v in page.unparsed.items()}
    return page.model_dump() != before


def _repoint_files(state: _State, old: str, new: str) -> None:
    vault = state.vault
    pattern = _link_pattern(old)
    paths = [vault.intentions_path, vault.questions_path, vault.core_path, vault.index_path]
    for base in (vault.episodes_dir, vault.journal_dir):
        paths += sorted(base.rglob("*.md")) if base.is_dir() else []
    for path in paths:
        text = state.read_text(path)
        if text is not None and (updated := pattern.sub("[[" + new, text)) != text:
            state.text_files[path] = updated
    state.intention_lines = [pattern.sub("[[" + new, x) for x in state.intention_lines]
    for month, lines in state.journal_lines.items():
        state.journal_lines[month] = [pattern.sub("[[" + new, x) for x in lines]


def _apply_merge(op: MergeEntities, state: _State) -> None:
    keep, gone = state.page(op.keep), state.page(op.merge)
    assert keep and gone
    keep.facts += gone.facts
    keep.history += gone.history
    for section, lines in gone.unparsed.items():
        keep.unparsed.setdefault(section, []).extend(lines)
    for name in (gone.title, *gone.aliases):
        if name.casefold() not in {n.casefold() for n in (keep.title, *keep.aliases)}:
            keep.aliases.append(name)
    keep.regions += [r for r in gone.regions if r not in keep.regions]
    if gone.privacy == "private":
        keep.privacy = "private"
    keep.summary = keep.summary or gone.summary
    keep.notes = "\n\n".join(n for n in (keep.notes, gone.notes) if n)
    keep.meta = {**gone.meta, **keep.meta}
    del state.pages[gone.id]
    state.deleted.add(gone.id)
    state.dirty.discard(gone.id)
    state.touch(keep)
    for page in state.pages.values():
        if _repoint_page(page, gone.subject, keep.subject):
            state.touch(page)
    _repoint_files(state, gone.subject, keep.subject)


def _apply_intention(op: AddIntention, state: _State, src: list[str]) -> str:
    intention = Intention(
        id=state.new_id("i"),
        text=_clean(op.text),
        when=[_entity(w) for w in op.when],
        after=op.after,
        src=list(src),
    )
    line = render_intention(intention)
    try:
        if parse_intention_line(line)[0] != intention:
            raise _Skip("cannot be written as an intention line without loss")
    except ValueError as exc:
        raise _Skip(f"cannot be written as an intention line: {exc}") from exc
    state.intention_lines.append(line)
    return intention.id


def _apply_observation(op: AddObservation, state: _State, src: list[str], created: datetime) -> str:
    obs = Observation(
        id=state.new_id("o"),
        day=created.date(),
        importance=op.importance,
        text=_clean(op.text),
        src=list(src),
    )
    line = render_observation(obs)
    try:
        if parse_observation_line(line) != obs:
            raise _Skip("cannot be written as a journal line without loss")
    except ValueError as exc:
        raise _Skip(f"cannot be written as a journal line: {exc}") from exc
    state.journal_lines.setdefault(obs.day.strftime("%Y-%m"), []).append(line)
    return obs.id


def _apply_confirm(op: ConfirmFact, state: _State, src: list[str], result: ApplyResult) -> None:
    found = state.find_fact(op.fact_id)
    assert found
    page, fact, _ = found
    fact.src = [*fact.src, *(s for s in src if s not in fact.src)]
    state.touch(page)
    result.confirmed += 1


def _apply_episode(op: AddEpisode, state: _State, result: ApplyResult) -> None:
    episode = Episode.model_validate(
        {
            **op.model_dump(exclude={"op"}),
            "gist": _clean(op.gist),
            "summary": op.summary.strip(),
            "regions": list(dict.fromkeys(op.regions)),
            "entities": list(dict.fromkeys(_entity(e) for e in op.entities)),
        }
    )
    state.episodes[episode.id] = episode
    state.episode_paths[episode.id] = state.vault.episode_path(episode)
    state.dirty_episodes.add(episode.id)
    result.episodes.append(episode.id)


def _apply_update_episode(op: UpdateEpisode, state: _State, result: ApplyResult) -> None:
    episode = state.episodes[op.id]
    if op.status:
        episode.status = op.status
    for entity in map(_entity, op.add_entities):
        if entity not in episode.entities:
            episode.entities.append(entity)
    state.dirty_episodes.add(op.id)
    result.episodes.append(op.id)


def _apply_op(op: Operation, state: _State, cs: ChangeSet, result: ApplyResult) -> None:
    match op:
        case CreateEntity():
            _apply_create(op, state, result)
        case AddFact():
            _apply_add_fact(op, state, cs.src, result)
        case SupersedeFact():
            _apply_supersede(op, state, cs.src, result)
        case MergeEntities():
            _apply_merge(op, state)
        case UpdateSummary():
            page = state.page(op.id)
            assert page
            page.summary = op.summary.strip()
            state.touch(page)
        case AddAlias():
            page = state.page(op.id)
            assert page
            page.aliases.append(_clean(op.alias))
            state.touch(page)
        case AddIntention():
            result.new_ids.append(_apply_intention(op, state, cs.src))
        case AddObservation():
            result.new_ids.append(_apply_observation(op, state, cs.src, cs.created))
        case ConfirmFact():
            _apply_confirm(op, state, cs.src, result)
        case AddEpisode():
            _apply_episode(op, state, result)
        case UpdateEpisode():
            _apply_update_episode(op, state, result)


# --- reviews -----------------------------------------------------------------------------------


def _preview(spec: FactSpec) -> str:
    fields = [("p", spec.p), ("o", spec.o), *spec.extra.items()]
    fields += [("from", spec.valid_from), ("to", spec.valid_to), ("kind", spec.kind)]
    if spec.evidence:
        fields.append(("evidence", ", ".join(spec.evidence)))
    return spec.text + "".join(f" ({k}:: {v})" for k, v in fields if v)


def _diff(op: Operation, state: _State) -> list[str]:
    match op:
        case CreateEntity():
            privacy = op.privacy or state.lifemap.privacy_for(op.regions)
            lines = [f"+ wiki/{op.id}.md ({op.type}, {privacy})", f"+ # {op.title}"]
            return lines + ([f"+ aliases: {', '.join(op.aliases)}"] if op.aliases else [])
        case AddFact():
            return [f"+ [{_subject(op)}] {_preview(op)}"]
        case SupersedeFact():
            found = state.find_fact(op.fact_id)
            old = render_fact(found[1]) if found else f"^{op.fact_id}"
            lines = [f"- {old}", f"  (to:: {op.to or '?'})"]
            return lines + ([f"+ {_preview(op.new)}"] if op.new else [])
        case MergeEntities():
            return [f"- [[{_entity(op.merge)}]]", f"+ [[{_entity(op.keep)}]]"]
        case UpdateSummary():
            page = state.page(op.id)
            old = page.summary.splitlines() if page else []
            return [f"- {x}" for x in old] + [f"+ {x}" for x in op.summary.splitlines()]
        case AddAlias():
            return [f"+ alias {op.alias!r} on {op.id}"]
        case AddIntention():
            when = ", ".join(op.when)
            return [f"+ - [ ] {op.text}" + (f" (when:: {when})" if when else "")]
        case AddObservation():
            return [f"+ imp {op.importance} · {op.text}"]
        case ConfirmFact():
            return [f"~ ^{op.fact_id} confirmed"]
        case AddEpisode():
            return [f"+ episode {op.id}: {op.gist}"]
        case UpdateEpisode():
            return [f"~ episode {op.id}" + (f" → {op.status}" if op.status else "")]
    return []


def _target(op: Operation) -> str:
    match op:
        case CreateEntity() | UpdateSummary() | AddAlias():
            return op.id
        case AddFact():
            return _subject(op)
        case SupersedeFact() | ConfirmFact():
            return f"^{op.fact_id}"
        case MergeEntities():
            return f"{op.merge} → {op.keep}"
        case AddEpisode() | UpdateEpisode():
            return op.id
    return "journal" if isinstance(op, AddObservation) else "intentions"


def _render_review(
    review_id: str, cs: ChangeSet, queued: list[tuple[Operation, str]], state: _State
) -> str:
    fm = render_frontmatter(
        [
            ("id", review_id),
            ("origin", cs.origin),
            ("created", cs.created.replace(microsecond=0)),
            ("src", cs.src),
            ("rationale", cs.rationale or None),
            ("status", "pending"),
        ]
    )
    out = [f"# Review {review_id}", ""]
    if cs.rationale:
        out += [cs.rationale, ""]
    out += ["## Changes", ""]
    for n, (op, reason) in enumerate(queued, 1):
        out += [f"### {n}. {op.op} · {_target(op)}", "", f"Why: {reason}", ""]
        out += ["```diff", *_diff(op, state), "```", ""]
    ops_json = json.dumps(
        [op.model_dump(mode="json", by_alias=True) for op, _ in queued],
        indent=2,
        ensure_ascii=False,
    )
    out += ["## Operations", "", "```json", ops_json, "```"]
    return fm + "\n".join(out) + "\n"


def load_review(path: Path) -> Review:
    text = path.read_text(encoding="utf-8")
    fm, body, _ = split_frontmatter(text)
    blocks = re.findall(r"```json\n(.*?)\n```", body, re.S)
    if fm is None or not blocks:
        raise ValueError(f"{path} is not a review file")
    return Review(
        id=str(fm["id"]),
        origin=fm["origin"],
        created=datetime.fromisoformat(str(fm["created"])),
        src=[str(s) for s in fm.get("src") or []],
        rationale=str(fm.get("rationale") or ""),
        status=str(fm.get("status", "pending")),
        ops=_OPS.validate_python(json.loads(blocks[-1])),
        path=path,
    )


def pending_reviews(vault: Vault) -> list[Review]:
    base = vault.reviews_pending_dir
    return [load_review(p) for p in sorted(base.glob("*.md"))] if base.is_dir() else []


def _resolved(path: Path, status: str, now: datetime, reason: str | None = None) -> str:
    fm, body, _ = split_frontmatter(path.read_text(encoding="utf-8"))
    fm = {**(fm or {}), "status": status, "resolved": now.replace(microsecond=0)}
    if reason:
        fm["reason"] = reason
    return render_frontmatter(fm.items()) + body


def _done_path(vault: Vault, review_id: str, now: datetime) -> Path:
    return vault.reviews_done_dir / now.strftime("%Y/%m") / f"{review_id}.md"


def _pending_path(vault: Vault, review_id: str) -> Path:
    if not REVIEW_ID.match(review_id):  # never let an id walk out of reviews/pending/
        raise ReviewNotFound(review_id)
    path = vault.reviews_pending_dir / f"{review_id}.md"
    if not path.is_file():
        raise ReviewNotFound(review_id)
    return path


def _new_review_id(vault: Vault) -> str:
    existing = {p.stem for p in vault.root.glob("reviews/**/r-*.md")}
    while (rid := new_id("r")) in existing:
        pass
    return rid


# --- files, log, commit ------------------------------------------------------------------------


def _append_lines(text: str, lines: list[str]) -> str:
    base = text.rstrip("\n")
    last = base.rsplit("\n", 1)[-1]
    sep = "\n" if last.lstrip().startswith(("- ", "* ", "+ ")) else "\n\n"
    return base + sep + "\n".join(lines) + "\n"


def _log_entry(cs: ChangeSet, result: ApplyResult, now: datetime) -> list[str]:
    src = ", ".join(cs.src) or "-"
    counts = f"+{result.facts_added} facts, {result.superseded} superseded"
    if result.confirmed:
        counts += f", {result.confirmed} confirmed"
    counts += f", {len(result.queued)} queued"
    if result.skipped:
        counts += f", {len(result.skipped)} skipped"
    lines = [f"## [{now:%Y-%m-%d %H:%M}] {cs.origin} | {src} | {counts}"]
    lines += [f"- created {page_id}" for page_id in result.created]
    lines += [f"- owner correction applied: ^{fact_id}" for fact_id in result.corrected]
    lines += [f"- episode {episode_id}" for episode_id in dict.fromkeys(result.episodes)]
    if result.review_id:
        lines.append(f"- queued for review: {result.review_id}")
    lines += [f"- skipped {op.op}: {reason}" for op, reason in result.skipped]
    return lines


def _append_log(vault: Vault, files: dict[Path, str | None], lines: list[str]) -> None:
    current = files.get(vault.log_path)
    if current is None:
        current = vault.log_path.read_text("utf-8") if vault.log_path.exists() else "# Log\n"
    files[vault.log_path] = current.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n"


def _collect_files(state: _State) -> dict[Path, str | None]:
    vault = state.vault
    files: dict[Path, str | None] = dict(state.text_files)
    for page_id in sorted(state.dirty - state.deleted):
        files[vault.page_path(page_id)] = render_page(state.pages[page_id])
    for page_id in sorted(state.deleted):
        files[vault.page_path(page_id)] = None
    if state.intention_lines:
        base = state.read_text(vault.intentions_path) or "# Intentions\n"
        files[vault.intentions_path] = _append_lines(base, state.intention_lines)
    for month, lines in sorted(state.journal_lines.items()):
        path = vault.journal_path(month)
        base = state.read_text(path)
        if base is None:
            files[path] = render_journal(month, []) + "\n" + "\n".join(lines) + "\n"
        else:
            files[path] = _append_lines(base, lines)
    for episode_id in sorted(state.dirty_episodes):
        files[state.episode_paths[episode_id]] = render_episode(state.episodes[episode_id])
    return files


def _tmp(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _mkdirs(path: Path, made: list[Path]) -> None:
    missing = [d for d in (path, *path.parents) if not d.exists()]
    for d in reversed(missing):
        d.mkdir()
        made.append(d)


def _write_files(files: dict[Path, str | None]) -> dict[Path, bytes | None]:
    """Write every file or none: stage temp files, then swap them in. On failure the touched
    files are restored and the error re-raised. Returns the previous contents for rollback."""
    backup = {path: path.read_bytes() if path.is_file() else None for path in files}
    made: list[Path] = []
    try:
        for path, text in files.items():
            if text is not None:
                _mkdirs(path.parent, made)
                _tmp(path).write_text(text, encoding="utf-8")
        for path, text in files.items():
            if text is None:
                path.unlink(missing_ok=True)
            else:
                os.replace(_tmp(path), path)
    except BaseException:
        _restore(backup, made)
        raise
    return backup


def _restore(backup: dict[Path, bytes | None], made: list[Path] | None = None) -> None:
    for path, data in backup.items():
        _tmp(path).unlink(missing_ok=True)
        if data is None:
            path.unlink(missing_ok=True)
        elif not path.is_file() or path.read_bytes() != data:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    for d in reversed(made or []):
        with contextlib.suppress(OSError):  # not empty: something else lives there now
            d.rmdir()


def _commit(vault: Vault, message: str) -> str | None:
    return gitops.commit_all(vault.root, message)


def _write_and_commit(vault: Vault, files: dict[Path, str | None], message: str) -> str | None:
    """Write ``files`` and commit them as one change; on any failure leave the vault as it was."""
    backup = _write_files(files)
    try:
        return _commit(vault, message)
    except BaseException:
        _restore(backup)
        if gitops.is_repo(vault.root):  # unstage what commit_all may have added
            paths = [str(p.relative_to(vault.root)) for p in backup]
            gitops._git(vault.root, "reset", "-q", "--", *paths, check=False)
        raise


# --- public API --------------------------------------------------------------------------------


def _run(
    cs: ChangeSet,
    vault: Vault,
    *,
    review: bool | None,
    now: datetime,
    extra_files: dict[Path, str | None] | None = None,
    corrections: Collection[str] = (),
) -> ApplyResult:
    state = _State(vault, now)
    result = ApplyResult()
    for op in cs.ops:
        problem = _validate(op, state, cs.src, cs.origin)
        if isinstance(problem, str):
            result.skipped.append((op, problem))
            continue
        if problem is DEPENDS:
            reason: str | None = "depends on an operation queued for review"
        elif review is None:
            reason = _policy(op, state, cs.origin, corrections)
        else:
            reason = "review requested" if review else None
        if reason:
            result.queued.append((op, reason))
            if isinstance(op, CreateEntity):
                state.queued_entities.add(op.id)
            continue
        try:
            _apply_op(op, state, cs, result)
        except _Skip as exc:
            result.skipped.append((op, str(exc)))
            continue
        result.applied.append(op)
        if review is None and isinstance(op, SupersedeFact) and op.fact_id in corrections:
            result.corrected.append(op.fact_id)

    files = _collect_files(state)
    if result.queued:
        result.review_id = _new_review_id(vault)
        path = vault.reviews_pending_dir / f"{result.review_id}.md"
        files[path] = _render_review(result.review_id, cs, result.queued, state)
    files.update(extra_files or {})
    if not files:  # nothing applied or queued: no log entry, no commit
        return result
    _append_log(vault, files, _log_entry(cs, result, now))
    result.commit = _write_and_commit(
        vault, files, f"brain: {cs.origin} {', '.join(cs.src) or '-'}"
    )
    result.changed = sorted(files)
    return result


def apply(
    changeset: ChangeSet,
    vault: Vault,
    *,
    review: bool | None = None,
    now: datetime | None = None,
    corrections: Collection[str] = (),
) -> ApplyResult:
    """Apply a change set: ``review=None`` follows the policy, True queues every valid op,
    False applies every valid op (used by ``approve``).

    ``corrections``: ids of facts the owner explicitly corrected (decided by the caller); a
    ``supersede_fact`` of one of them is applied when ``correction_problem`` allows it."""
    return _run(changeset, vault, review=review, now=now or datetime.now(), corrections=corrections)


def log_event(
    vault: Vault, origin: Origin, src: list[str], message: str, *, now: datetime | None = None
) -> str | None:
    """Append one line to log.md and commit (an outcome that changes no memory, e.g. a
    conversation with nothing to remember)."""
    now = now or datetime.now()
    files: dict[Path, str | None] = {}
    entry = f"## [{now:%Y-%m-%d %H:%M}] {origin} | {', '.join(src) or '-'} | {_clean(message)}"
    _append_log(vault, files, [entry])
    return _write_and_commit(vault, files, f"brain: {origin} {', '.join(src) or '-'}")


def log_has(vault: Vault, origin: Origin, src: str, message_start: str) -> bool:
    """Whether log.md has an entry ``origin | src | message_start…``."""
    if not vault.log_path.exists():
        return False
    pattern = re.compile(
        rf"^## \[[^\]]+\] {re.escape(origin)} \| {re.escape(src)} \| {re.escape(message_start)}",
        re.M,
    )
    return bool(pattern.search(vault.log_path.read_text(encoding="utf-8")))


def approve(review_id: str, vault: Vault, *, now: datetime | None = None) -> ApplyResult:
    """Apply a pending review and move it to ``reviews/done/YYYY/MM/``."""
    now = now or datetime.now()
    path = _pending_path(vault, review_id)
    rev = load_review(path)
    src = rev.src if review_id in rev.src else [*rev.src, review_id]
    cs = ChangeSet(
        ops=rev.ops, origin=rev.origin, src=src, rationale=rev.rationale, created=rev.created
    )
    moved = {path: None, _done_path(vault, review_id, now): _resolved(path, "approved", now)}
    return _run(cs, vault, review=False, now=now, extra_files=moved)


def reject(review_id: str, reason: str, vault: Vault, *, now: datetime | None = None) -> Path:
    """Move a pending review to ``reviews/done/YYYY/MM/`` as rejected; return its new path."""
    now = now or datetime.now()
    path = _pending_path(vault, review_id)
    load_review(path)  # refuse to move anything that is not a review
    done = _done_path(vault, review_id, now)
    files: dict[Path, str | None] = {path: None, done: _resolved(path, "rejected", now, reason)}
    entry = f"## [{now:%Y-%m-%d %H:%M}] user | {review_id} | rejected: {_clean(reason)}"
    _append_log(vault, files, [entry])
    _write_and_commit(vault, files, f"brain: reject {review_id}")
    return done
