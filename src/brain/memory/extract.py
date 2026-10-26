"""Extract one episode with predict-calibrate (docs/architecture.md, Flow 3).

**Predict**: the model sees the episode and what memory already holds about the entities it
mentions (``known_facts``: current facts and ended ones, ≤ ``KNOWN_TOKENS``), and labels each
candidate fact ``new`` / ``confirms`` / ``contradicts`` / ``noop`` with the id of the known fact it
refers to. It also writes journal observations and intentions.

**Calibrate** (code, ``calibrate``): ``noop`` is dropped; ``confirms`` is checked (the known fact
exists, same predicate, a similar value) and becomes a ``confirm_fact`` (the episode is added to
the fact's sources; nothing else changes), otherwise it goes on as ``new``; ``contradicts``
without a value ends the known fact (``supersede_fact`` without a replacement).

The rest goes through ingest's machinery unchanged: grounding and typed checks
(``ingest.decisions.check_part``), normalisation (``ingest.normalise.build_changeset``) and
paraphrase merging (``ingest.dedup``). Then every ``add_fact`` is compared with memory in code
(``plan_ops``): a restatement (same subject, predicate, value; no new dates), of a current or an
ended fact, becomes a ``confirm_fact``; a clash (the writer's scoped contradiction check, or the
known fact the model named) becomes a ``supersede_fact``. A supersession is auto-applied only as
an **owner correction**: a typed decision ("does the owner explicitly say the stored fact is wrong,
has changed or is no longer true?") at ≥ ``CORRECTION_P`` and the writer's
``correction_problem`` (a fact about the owner, not a relationship, nothing private). Everything
else is queued for review by the writer's policy.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import httpx
from pydantic import BaseModel

from brain.ingest import dedup
from brain.ingest.decisions import Checks, check_part, named
from brain.ingest.extract import (
    INGEST_PREDICATES,
    ExtractedEntity,
    ExtractedFact,
    Extraction,
    KnownEntity,
    render_known,
)
from brain.ingest.normalise import SELF_WORDS, fold, normalise_date
from brain.ingest.normalise import build_changeset as normalise_facts
from brain.llm.base import OutputTruncated, StructuredOutputError
from brain.llm.decide import NO, YES, Choice, ChoiceQuestion
from brain.memory.conversation import in_range, render, render_plain
from brain.memory.resolve import Resolution
from brain.memory.segment import Segment
from brain.prompts import load_prompt
from brain.tokens import estimate_tokens
from brain.vault.lifemap import LifeMap
from brain.vault.models import (
    PREDICATES,
    SELF_ID,
    Episode,
    Fact,
    Page,
    is_multi_valued,
    link_target,
)
from brain.vault.ops import (
    AddAlias,
    AddFact,
    AddIntention,
    AddObservation,
    ConfirmFact,
    CreateEntity,
    FactSpec,
    Operation,
    SupersedeFact,
)
from brain.vault.raw import Turn
from brain.vault.store import Vault
from brain.vault.writer import correction_problem, overlaps

log = logging.getLogger(__name__)

MEMORY_PREDICATES = (
    *INGEST_PREDICATES,
    "dislikes",
    "fears",
    "health",
    "partner_of",
    "met_at",
    "last_seen",
)
assert set(MEMORY_PREDICATES) <= PREDICATES
Predicate = Literal[MEMORY_PREDICATES]  # type: ignore[valid-type]
Label = Literal["new", "confirms", "contradicts", "noop"]

KNOWN_TOKENS = 900  # the KNOWN FACTS block
PER_ENTITY = 10  # known facts shown per mentioned entity (other than the owner)
MAX_OBSERVATIONS = 2
MAX_TOKENS = 3072
# One extraction call reads at most about this many words of the episode: a long episode (an
# interview on one topic can run 80 turns) is extracted in windows of whole exchanges, and a
# window whose reply is still cut off is halved (docs/decisions.md).
WINDOW_WORDS = 450
MEMBERSHIP = frozenset({"works_at", "studied_at", "member_of"})
PERSON_ONLY = frozenset({"met_at", "last_seen"})  # the schema files these under a person
CORRECTION_P = 0.9  # p(the owner explicitly corrects the stored fact) to auto-apply
DEDUP = True  # merge paraphrased duties/skills/interests (ingest.dedup)
WORD = re.compile(r"[\w'-]+")
STOP = frozenset({"a", "an", "the", "of", "in", "at", "for", "and", "to", "with", "on", "as"})
ERRORS = (httpx.HTTPError, StructuredOutputError, ValueError, LookupError)

# Wording chosen on 20 labelled fictional cases (tests/fixtures/memory/corrections.yaml): this
# two-option choice gave every explicit correction p ≥ 0.99 and every other case ≤ 0.59; a
# yes/no predicate ("does the owner explicitly say … wrong, changed or no longer true?") gave
# explicit cases 0.78–0.97 (3 of 9 at ≥ 0.9); a four-way choice let "back in Lisbon with my
# parents" reach 0.92 (docs/decisions.md).
CORRECTION = ChoiceQuestion(
    name="correction",
    instructions="Does the owner tell that the stored fact is wrong or out of date?",
    choices=[
        Choice(
            value=YES,
            description="the owner states it is wrong, has changed or is no longer true",
        ),
        Choice(
            value=NO,
            description=(
                "the owner restates it, or talks about something else, a plan, a doubt or "
                "someone else's view"
            ),
        ),
    ],
)


def correction_input(stored: str, conversation: str) -> str:
    return f"Stored fact: {stored}\n\nConversation:\n{conversation}"


class MemoryFact(BaseModel):
    subject: str
    predicate: Predicate
    object: str
    text: str
    start: str
    end: str
    label: Label
    known: str


class MemoryIntention(BaseModel):
    text: str
    after: str
    about: list[str]


class MemoryExtraction(BaseModel):
    reasoning: str
    entities: list[ExtractedEntity]
    facts: list[MemoryFact]
    observations: list[str]
    intentions: list[MemoryIntention]


# --- what memory holds ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnownFact:
    fact: Fact
    subject: str
    label: str  # the subject's name as shown
    ended: bool  # in History without a replacement


def _words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.casefold()) if w not in STOP and len(w) > 1}


def _titles(vault: Vault) -> dict[str, str]:
    return {p.subject: p.title for p in vault.pages()}


def _value(o: str, titles: dict[str, str]) -> str:
    target = link_target(o)
    if target is None:
        return o
    return "the owner" if target == SELF_ID else titles.get(target, target)


def _span(fact: Fact) -> str:
    if fact.valid_from and fact.valid_to:
        return f"{fact.valid_from} to {fact.valid_to}"
    if fact.valid_from:
        return f"since {fact.valid_from}"
    return f"until {fact.valid_to}" if fact.valid_to else ""


def _facts_of(pages: Iterable[Page]) -> list[tuple[Fact, bool]]:
    out = [(f, False) for p in pages for f in p.facts]
    return out + [(f, True) for p in pages for f in p.history if not f.by]


def known_facts(
    vault: Vault,
    entities: Sequence[str],
    *,
    regions: Sequence[str],
    text: str,
    lifemap: LifeMap,
    budget: int = KNOWN_TOKENS,
) -> list[KnownFact]:
    """What memory holds that this episode may restate or change: the facts of the entities it
    mentions (up to ``PER_ENTITY`` each) and the owner's facts that belong to its regions or share
    words with it, best first, under ``budget`` tokens."""
    titles = _titles(vault)
    words = _words(text)
    region_ps = {s.p for r in regions for s in lifemap.slots(r)}
    region_ps |= {s.p for r in regions if (reg := lifemap.region(r)) for s in reg.slots}
    scored: list[tuple[float, KnownFact]] = []
    for subject in [SELF_ID, *dict.fromkeys(e for e in entities if e != SELF_ID)]:
        pages = vault.pages_about(subject)
        label = "self" if subject == SELF_ID else titles.get(subject, subject)
        ranked = []
        for fact, ended in _facts_of(pages):
            overlap = len(words & _words(f"{fact.text} {_value(fact.o, titles)}"))
            score = min(overlap, 3) + (2 if fact.p in region_ps else 0) - (0.5 if ended else 0)
            if subject == SELF_ID and score <= 0:
                continue
            ranked.append(
                (score + (1 if subject != SELF_ID else 0), KnownFact(fact, subject, label, ended))
            )
        ranked.sort(key=lambda r: -r[0])
        scored += ranked if subject == SELF_ID else ranked[:PER_ENTITY]
    scored.sort(key=lambda r: -r[0])
    out: list[KnownFact] = []
    used = 0
    for _, known in scored:
        cost = estimate_tokens(render_known_fact(known, titles)) + 1
        if used + cost > budget:
            continue
        out.append(known)
        used += cost
    return out


def render_known_fact(known: KnownFact, titles: dict[str, str]) -> str:
    fact = known.fact
    parts = [known.label, fact.p, _value(fact.o, titles)]
    if span := _span(fact):
        parts.append(span + (" (ended)" if known.ended else ""))
    return f"- {fact.id}: " + " · ".join(parts)


def render_known_facts(known: Sequence[KnownFact], titles: dict[str, str]) -> str:
    return "\n".join(render_known_fact(k, titles) for k in known) or "(none)"


# --- the call ----------------------------------------------------------------------------------


def build_messages(
    turns: Sequence[Turn],
    *,
    owner: str,
    today: date,
    known_entities: Sequence[KnownEntity],
    owner_names: Sequence[str],
    known: Sequence[KnownFact],
    titles: dict[str, str],
    summary: str = "",
) -> list[dict[str, str]]:
    user = (
        f"Today is {today.isoformat()}. Owner: {owner}.\n\n"
        f"KNOWN ENTITIES (id: names):\n{render_known(known_entities, owner_names)}\n\n"
        f"KNOWN FACTS (id: subject · predicate · value · dates):\n"
        f"{render_known_facts(known, titles)}\n\n"
        + (f"EPISODE SUMMARY: {summary}\n\n" if summary else "")
        + f"EPISODE:\n{render(turns)}"
    )
    return [
        {"role": "system", "content": load_prompt("memory_extract")},
        {"role": "user", "content": user},
    ]


def windows(turns: Sequence[Turn], *, words: int = WINDOW_WORDS) -> list[list[Turn]]:
    """Consecutive spans of about ``words`` words, cut only after an owner turn so a question
    stays with its answer."""
    out: list[list[Turn]] = []
    current: list[Turn] = []
    size = 0
    for turn in turns:
        current.append(turn)
        size += len(turn.text.split())
        if turn.role == "user" and size >= words:
            out.append(current)
            current, size = [], 0
    if current:
        if out and not any(t.role == "user" for t in current):
            out[-1] += current  # a trailing assistant turn belongs to the last exchange
        else:
            out.append(current)
    return out


def merge_extractions(parts: Sequence[MemoryExtraction]) -> MemoryExtraction:
    """One extraction from several windows: entities once per name, everything else kept."""
    entities: dict[str, ExtractedEntity] = {}
    for part in parts:
        for entity in part.entities:
            entities.setdefault(entity.name.casefold(), entity)
    return MemoryExtraction(
        reasoning=" ".join(p.reasoning for p in parts if p.reasoning),
        entities=list(entities.values()),
        facts=[f for p in parts for f in p.facts],
        observations=[o for p in parts for o in p.observations],
        intentions=[i for p in parts for i in p.intentions],
    )


def extract_windows(
    llm: Any, span: Sequence[Turn], build: Callable[[Sequence[Turn]], list[dict[str, str]]]
) -> MemoryExtraction:
    """Extract ``span`` window by window; a window cut off at ``MAX_TOKENS`` is halved until it
    is a single exchange (then the error is raised)."""
    queue = windows(span)
    parts: list[MemoryExtraction] = []
    while queue:
        window = queue.pop(0)
        try:
            parts.append(llm.structured(build(window), MemoryExtraction, max_tokens=MAX_TOKENS))
        except OutputTruncated:
            owners = [i for i, t in enumerate(window) if t.role == "user"]
            if len(owners) < 2:
                raise
            cut = owners[len(owners) // 2 - 1] + 1  # after the owner turn closing the first half
            log.info("memory extraction window truncated; split in two")
            queue[:0] = [window[:cut], window[cut:]]
    return parts[0] if len(parts) == 1 else merge_extractions(parts)


# --- calibrate ----------------------------------------------------------------------------------


@dataclass
class Calibrated:
    facts: list[MemoryFact] = field(default_factory=list)  # new / contradicts with a value
    confirms: list[str] = field(default_factory=list)  # known fact ids restated
    endings: list[tuple[str, str | None]] = field(default_factory=list)  # (fact id, end)
    hints: dict[tuple[str, str], str] = field(default_factory=dict)  # (p, value) → known id
    dropped: int = 0


def _similar(a: str, b: str) -> bool:
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and (wa <= wb or wb <= wa or len(wa & wb) / len(wa | wb) >= 0.5)


def calibrate(
    extraction: MemoryExtraction,
    known: dict[str, KnownFact],
    titles: dict[str, str],
    notes: list[str],
) -> Calibrated:
    out = Calibrated()
    for f in extraction.facts:
        about = f"{f.predicate} {f.object or '∅'}"
        ref = known.get(f.known.strip().lstrip("^"))
        if f.label == "noop":
            out.dropped += 1
            continue
        if f.predicate in PERSON_ONLY and fold(f.subject) in SELF_WORDS:
            notes.append(f"calibrate: dropped {about}: {f.predicate} is about a person")
            out.dropped += 1
            continue
        if f.label == "confirms":
            same = ref and ref.fact.p == f.predicate
            same = same and (not f.object or _similar(f.object, _value(ref.fact.o, titles)))
            if same:
                assert ref is not None
                out.confirms.append(ref.fact.id)
                continue
            notes.append(f"calibrate: {about} confirms {f.known or 'nothing'}? no: kept as new")
        if f.label == "contradicts" and not f.object.strip():
            if ref and not ref.ended:
                out.endings.append((ref.fact.id, normalise_date(f.end)))
                notes.append(f"calibrate: {ref.fact.id} ({ref.fact.p}) no longer true")
            else:
                notes.append(f"calibrate: dropped {about}: no value and no current known fact")
            continue
        if f.label == "contradicts" and ref and ref.fact.p == f.predicate:
            out.hints[(f.predicate, fold(f.object))] = ref.fact.id
        out.facts.append(f)
    return out


def as_extraction(
    calibrated: Calibrated, raw: MemoryExtraction, resolutions: Sequence[Resolution], vault: Vault
) -> Extraction:
    """Ingest's ``Extraction`` for the kept facts. Entities on known pages carry all their names
    (title, aliases, the name used in this episode), so grounding finds "Sam" for "Sam Okafor";
    a resolved mention the model did not declare is added."""
    names: dict[str, list[str]] = {}  # entity id → names the episode may use
    for res in resolutions:
        if res.entity_id:
            names.setdefault(res.entity_id, []).append(res.name)
    by_name = {
        fold(n): page.subject
        for entity_id in names
        if (page := vault.page(entity_id))
        for n in (page.title, *page.aliases, *names[entity_id])
    }
    entities: list[ExtractedEntity] = []
    for entity in raw.entities:
        page = vault.page(entity.id.strip()) if entity.id.strip() else None
        if page is None and (known := by_name.get(fold(entity.name))):
            page = vault.page(known)  # a resolved page under a mangled or missing id
        if page and page.subject != SELF_ID:
            extra = [page.title, *page.aliases, *names.pop(page.subject, [])]
            aliases = [*entity.aliases]
            for name in extra:
                if fold(name) not in {fold(a) for a in (entity.name, *aliases)}:
                    aliases.append(name)
            entity = entity.model_copy(update={"id": page.subject, "aliases": aliases})
        entities.append(entity)
    declared = {fold(n) for e in entities for n in (e.name, *e.aliases)}
    for entity_id, used in names.items():
        page = vault.page(entity_id)
        if page is None or all(fold(n) in declared for n in used):
            continue
        etype = page.type if page.type in ("person", "org", "project", "place", "event") else "org"
        aliases = [n for n in dict.fromkeys([*page.aliases, *used]) if fold(n) != fold(page.title)]
        entities.append(
            ExtractedEntity(name=page.title, type=etype, id=page.subject, aliases=aliases)
        )
    facts = [
        ExtractedFact.model_construct(  # the memory vocabulary is wider than ingest's enum
            subject=f.subject,
            predicate=f.predicate,
            object=f.object,
            text=_membership_text(f),
            start=f.start,
            end=f.end,
        )
        for f in calibrated.facts
    ]
    return Extraction(reasoning=raw.reasoning, owner_names=[], entities=entities, facts=facts)


def _membership_text(f: MemoryFact) -> str:
    """A ``works_at`` / ``studied_at`` / ``member_of`` whose text names its own organisation
    ("Left Fabrikam Analytics in March 2023") is about the membership, not a duty: its text is
    dropped (rendered from the template), so ingest's verb-led-``works_at`` guard does not turn
    it into a responsibility when the organisation has no page yet."""
    if f.predicate in MEMBERSHIP and f.object.strip() and named(f.text, f.object.strip()):
        return ""
    return f.text


# --- plan operations ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """A supersession that may be an owner correction."""

    op: SupersedeFact
    old: Fact
    p: float | None = None
    problem: str | None = None  # why it cannot be auto-applied (writer policy)


def _subject(op: AddFact) -> str:
    s = link_target(op.subject) or op.subject
    return SELF_ID if s in (SELF_ID, "self/profile") else s


def _same_value(a: str, b: str, titles: dict[str, str]) -> bool:
    """Two objects name the same thing: the same page, or the same name (a literal
    "Fabrikam Analytics" and a link to the page of that name)."""
    ta, tb = link_target(a), link_target(b)
    return (ta is not None and ta == tb) or fold(_value(a, titles)) == fold(_value(b, titles))


def _adds_nothing(new: str | None, old: str | None) -> bool:
    return new is None or (old is not None and old.startswith(new))


def _restates(fact: Fact, spec: FactSpec, titles: dict[str, str]) -> bool:
    return (
        fact.p == spec.p
        and _same_value(fact.o, spec.o, titles)
        and _adds_nothing(spec.valid_from, fact.valid_from)
        and _adds_nothing(spec.valid_to, fact.valid_to)
    )


def _clashes(
    facts: Iterable[Fact], subject: str, spec: FactSpec, titles: dict[str, str]
) -> list[Fact]:
    """The writer's scoped check (same p, + the same value for multi-valued predicates,
    overlapping valid time), comparing values by name."""
    multi = is_multi_valued(spec.p, subject)
    return [
        f
        for f in facts
        if f.p == spec.p and (not multi or _same_value(f.o, spec.o, titles)) and overlaps(f, spec)
    ]


def _merged(spec: FactSpec, old: Fact, titles: dict[str, str]) -> FactSpec:
    """A corrected fact about the same value keeps what the correction did not restate."""
    if not _same_value(spec.o, old.o, titles):
        return spec
    update: dict[str, Any] = {}
    if (
        spec.valid_from is None
        and old.valid_from
        and not (spec.valid_to and spec.valid_to < old.valid_from)
    ):
        update["valid_from"] = old.valid_from
    if spec.valid_to is None and old.valid_to and old.valid_to >= (spec.valid_from or ""):
        update["valid_to"] = old.valid_to
    extra = {**old.extra, **spec.extra}
    if extra != spec.extra:
        update["extra"] = extra
    return spec.model_copy(update=update) if update else spec


def _spec(op: AddFact) -> FactSpec:
    return FactSpec.model_validate(op.model_dump(by_alias=True, exclude={"op", "subject"}))


def plan_ops(
    ops: Sequence[Operation],
    vault: Vault,
    calibrated: Calibrated,
    titles: dict[str, str],
    notes: list[str],
) -> tuple[list[Operation], list[Candidate]]:
    """Compare each ``add_fact`` with memory: restatements → ``confirm_fact``; clashes →
    ``supersede_fact`` (correction candidates). Confirmations and endings are added; planned
    pages no remaining operation uses are dropped."""
    titles = {**titles, **{op.id: op.title for op in ops if isinstance(op, CreateEntity)}}
    out: list[Operation] = []
    candidates: list[Candidate] = []
    confirmed: set[str] = set()
    superseded: set[str] = set()

    def confirm(fact_id: str, why: str) -> None:
        if fact_id not in confirmed:
            confirmed.add(fact_id)
            out.append(ConfirmFact(fact_id=fact_id))
            notes.append(f"plan: ^{fact_id} {why}")

    for fact_id in calibrated.confirms:
        confirm(fact_id, "confirmed")
    for op in ops:
        if not isinstance(op, AddFact):
            out.append(op)
            continue
        subject = _subject(op)
        pages = vault.pages_about(subject)
        current = [f for p in pages for f in p.facts]
        ended = [f for p in pages for f in p.history if not f.by]
        spec = _spec(op)
        if dup := next((f for f in (*current, *ended) if _restates(f, spec, titles)), None):
            confirm(dup.id, f"restated ({op.p} {_value(op.o, titles)})")
            continue
        clashes = _clashes(current, subject, spec, titles)
        hinted = calibrated.hints.get((op.p, fold(_value(op.o, titles))))
        if not clashes:
            # an ended fact about the same value with other dates, or the fact the model named
            same = [f for f in ended if f.p == op.p and _same_value(f.o, op.o, titles)]
            named = [f for f in (*current, *ended) if f.id == hinted and f.p == op.p]
            clashes = same or named
        clashes = [f for f in clashes if f.id not in superseded]
        if not clashes:
            out.append(op)
            continue
        old = clashes[0]
        superseded.add(old.id)
        sup = SupersedeFact(fact_id=old.id, new=_merged(spec, old, titles))
        out.append(sup)
        candidates.append(Candidate(sup, old))
        notes.append(f"plan: {op.p} {_value(op.o, titles)} would replace ^{old.id}")
    for fact_id, end in calibrated.endings:
        found = vault.find_fact(fact_id)
        if found and not found[2] and fact_id not in superseded:
            superseded.add(fact_id)
            sup = SupersedeFact(fact_id=fact_id, to=end)
            out.append(sup)
            candidates.append(Candidate(sup, found[1]))
    return _drop_unused_entities(out, notes), candidates


def ground_links(
    ops: Sequence[Operation], vault: Vault, text: str, notes: list[str]
) -> list[Operation]:
    """Drop facts whose subject or object is a page the episode never names (by its title, an
    alias or a capitalised word of its name). Ingest grounds the objects of page predicates
    only; a conversation's facts point at known pages for any predicate (a "met at" a city)."""
    planned = {op.id: (op.title, op.aliases) for op in ops if isinstance(op, CreateEntity)}

    def names_of(entity: str) -> tuple[str, list[str]] | None:
        if entity == SELF_ID:
            return None
        if entity in planned:
            return planned[entity][0], list(planned[entity][1])
        page = vault.page(entity)
        return (page.title, list(page.aliases)) if page else None

    out: list[Operation] = []
    for op in ops:
        if isinstance(op, AddFact):
            for entity in (_subject(op), link_target(op.o)):
                found = names_of(entity) if entity else None
                if found and not named(text, found[0], found[1]):
                    notes.append(
                        f"ground: dropped {op.p} {op.o}: the episode never names {found[0]!r}"
                    )
                    break
            else:
                out.append(op)
            continue
        out.append(op)
    return out


def _drop_unused_entities(ops: list[Operation], notes: list[str]) -> list[Operation]:
    used: set[str] = set()
    for op in ops:
        if isinstance(op, AddFact):
            used.add(_subject(op))
        spec = op if isinstance(op, AddFact) else getattr(op, "new", None)
        if isinstance(spec, FactSpec):
            used |= {t for v in (spec.o, *spec.extra.values()) if (t := link_target(v))}
    out = []
    for op in ops:
        if isinstance(op, CreateEntity) and op.id not in used:
            notes.append(f"plan: dropped entity {op.id}: its facts were restatements")
            continue
        out.append(op)
    return out


def decide_corrections(
    llm: Any, candidates: Sequence[Candidate], vault: Vault, episode_text: str, notes: list[str]
) -> tuple[set[str], float, int]:
    """The ids of facts the owner explicitly corrected (eligible ones only); seconds; asked."""
    start = time.perf_counter()
    titles = _titles(vault)
    corrections: set[str] = set()
    asked = 0
    for cand in candidates:
        cand.problem = correction_problem(cand.op, vault)
        if cand.problem:
            notes.append(f"correction: ^{cand.old.id}: {cand.problem}: to review")
            continue
        stored = f"{cand.old.text} ({cand.old.p}: {_value(cand.old.o, titles)}"
        stored += f", {span})" if (span := _span(cand.old)) else ")"
        try:
            [decision] = llm.decide(correction_input(stored, episode_text), [CORRECTION])
        except ERRORS as err:
            notes.append(f"correction: ^{cand.old.id}: no decision ({err}): to review")
            continue
        asked += 1
        cand.p = None if decision.refusal else decision.probabilities.get(YES, 0.0)
        if cand.p is not None and cand.p >= CORRECTION_P:
            corrections.add(cand.old.id)
            notes.append(f"correction: ^{cand.old.id} explicitly corrected (p={cand.p:.3f})")
        else:
            shown = "no answer" if cand.p is None else f"p={cand.p:.3f} < {CORRECTION_P}"
            notes.append(f"correction: ^{cand.old.id}: {shown}: to review")
    return corrections, time.perf_counter() - start, asked


# --- one episode -------------------------------------------------------------------------------


@dataclass
class EpisodeExtraction:
    ops: list[Operation] = field(default_factory=list)
    corrections: set[str] = field(default_factory=set)
    candidates: list[Candidate] = field(default_factory=list)
    raw: MemoryExtraction | None = None
    created: list[str] = field(default_factory=list)  # entity ids the ops create
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0  # extraction call
    check_seconds: float = 0.0
    correction_seconds: float = 0.0
    asked: int = 0  # typed questions (checks + corrections)


def extract_episode(
    vault: Vault,
    llm: Any,
    episode: Episode | Segment,
    turns: Sequence[Turn],
    *,
    episode_id: str,
    resolutions: Sequence[Resolution],
    owner: str,
    owner_names: Sequence[str],
    today: date,
    importance: int,
) -> EpisodeExtraction:
    """Extract one episode into operations (not applied). Raises the model's errors."""
    out = EpisodeExtraction()
    first, last = (episode.first, episode.last) if isinstance(episode, Segment) else episode.turns
    regions = list(episode.regions)
    summary = episode.summary
    span = in_range(turns, first, last)
    text = render(span)
    titles = _titles(vault)
    lifemap = vault.lifemap
    entities = [r.entity_id for r in resolutions if r.entity_id]
    known = known_facts(vault, entities, regions=regions, text=text, lifemap=lifemap)
    known_entities = [
        KnownEntity(p.subject, tuple(n for n in (p.title, *p.aliases) if n))
        for e in dict.fromkeys(entities)
        if (p := vault.page(e))
    ]

    def build(window: Sequence[Turn]) -> list[dict[str, str]]:
        return build_messages(
            window,
            owner=owner,
            today=today,
            known_entities=known_entities,
            owner_names=owner_names,
            known=known,
            titles=titles,
            summary=summary,
        )

    start = time.perf_counter()
    raw = extract_windows(llm, span, build)
    out.seconds = time.perf_counter() - start
    out.raw = raw

    calibrated = calibrate(raw, {k.fact.id: k for k in known}, titles, out.notes)
    extraction = as_extraction(calibrated, raw, resolutions, vault)
    names = [n for e in known_entities for n in e.names]
    start = time.perf_counter()
    try:
        checks = check_part(llm, text, extraction, known=names, owner=owner_names)
    except ERRORS as err:
        checks = Checks(notes=[f"check: no decision ({err})"])
    out.check_seconds = time.perf_counter() - start
    out.asked += checks.asked
    out.notes += checks.notes
    normalised = normalise_facts(
        [extraction],
        vault,
        src=episode_id,
        kind="stated",
        today=today,
        entity_kinds=checks.kinds,
        texts=[text],
        origin="extract",
    )
    out.notes += normalised.notes
    changeset = normalised.changeset
    if DEDUP and any(isinstance(op, AddFact) for op in changeset.ops):
        changeset, report = dedup.merge_paraphrases(changeset, vault, llm, ask=True)
        out.notes += report.notes
    grounded = ground_links(changeset.ops, vault, text, out.notes)
    ops, out.candidates = plan_ops(grounded, vault, calibrated, titles, out.notes)
    for res in resolutions:
        if res.alias and res.entity_id and len(res.name) >= 3:
            ops.append(AddAlias(id=vault.page_id_for(res.entity_id), alias=res.name))
    ops += observations(raw, importance)
    ops += intentions(raw, resolutions, vault, today, out.notes)
    if getattr(episode, "privacy", "normal") == "private":
        ops = make_private(ops, out.notes)
    out.ops = ops
    out.created = [op.id for op in ops if isinstance(op, CreateEntity)]
    out.corrections, out.correction_seconds, asked = decide_corrections(
        llm, out.candidates, vault, render_plain(span), out.notes
    )
    out.asked += asked
    return out


def make_private(ops: Sequence[Operation], notes: list[str]) -> list[Operation]:
    """Everything from a private answer stays private: facts get ``privacy:: private``, new
    pages are private, and no alias is added to a page (a name used in confidence). Journal
    lines and intentions cite the private episode, which makes them private in the index."""
    out: list[Operation] = []
    for op in ops:
        if isinstance(op, AddFact):
            op = op.model_copy(update={"extra": {**op.extra, "privacy": "private"}})
        elif isinstance(op, SupersedeFact) and op.new is not None:  # in place: a candidate
            op.new = op.new.model_copy(update={"extra": {**op.new.extra, "privacy": "private"}})
        elif isinstance(op, CreateEntity):
            op = op.model_copy(update={"privacy": "private"})
        elif isinstance(op, AddAlias):
            notes.append(f"private: no alias {op.alias!r} on {op.id}")
            continue
        out.append(op)
    return out


def observations(raw: MemoryExtraction, importance: int) -> list[AddObservation]:
    out = []
    for text in raw.observations[:MAX_OBSERVATIONS]:
        text = " ".join(text.split())
        if text:
            out.append(AddObservation(text=text, importance=importance))
    return out


def intentions(
    raw: MemoryExtraction,
    resolutions: Sequence[Resolution],
    vault: Vault,
    today: date,
    notes: list[str],
) -> list[AddIntention]:
    by_name = {fold(r.name): r.entity_id for r in resolutions if r.entity_id}
    aliases = vault.alias_index()
    open_texts = {fold(i.text) for i in vault.intentions() if not i.done}
    out = []
    for item in raw.intentions:
        text = " ".join(item.text.split())
        if not text or fold(text) in open_texts:
            continue
        after = normalise_date(item.after) if item.after.strip() else None
        if item.after.strip() and after is None:
            notes.append(f"intention: unreadable date {item.after!r} dropped: {text}")
        when = []
        for name in item.about:
            ids = [i for i in aliases.get(fold(name), []) if i != SELF_ID]
            target = by_name.get(fold(name)) or (ids[0] if len(ids) == 1 else None)
            if target and target not in when:
                when.append(target)
        out.append(AddIntention(text=text, when=when, after=after))
        open_texts.add(fold(text))
    return out
