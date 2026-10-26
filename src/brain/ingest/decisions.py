"""Typed decisions around extraction (``brain.llm`` ``decide``; docs/decision-models.md).

- **Gate**: before a part is extracted, one yes/no question on a compact view of it (headings
  and the first words of each block): does it say anything about a person? A part with
  ``p(yes) < SKIP_BELOW`` (a cookie notice, a generic tutorial) is skipped; it would cost a full
  extraction call (~30-45 s) for nothing. A document's only part is never skipped.
- **Grounding** (code, before the checks): a fact pointing at an entity the part never names
  is dropped. On a part about something else (a tutorial next to a contact block) the extractor
  restated facts about every entity in its KNOWN ENTITIES list.
- **Checks**: after a part is extracted, typed questions about what the extractor proposed and
  code cannot verify: whether a *new* entity (one that would get a page) is really a person,
  organisation, project, place or event and not a team, tool or topic; and whether a skill /
  interest / habit / likes value is a work skill or a leisure activity; whether a ``works_at``
  organisation is an employer and not a school or a club joined; and whether a ``speaks`` value
  outside the known languages is a language at all. A confident answer
  (``confidence >= ACT_ABOVE``) overrides the extractor; otherwise its choice is kept and noted.

Cost shapes the checks. The runtime answers questions one at a time and re-reads the input for
each (~0.35 s + ~1.6 ms per input token on the base M4; a shared input is not reused), so asking
about every item of a CV part took 8-14 s. Instead, each set of items is **screened** with one
choice question ("which of these is not a person, organisation, …? / none of them"); only the
items it flags (``SCREEN_FLAG``) get their own typed question. A clean set costs one question.
Several suspects in one set share the screen's probability and are all flagged when each keeps
``SCREEN_FLAG``; one far behind another is missed (re-screening without the first mostly flagged
ordinary items). Inputs are the sentences or list lines naming the items, under their heading,
not the whole part.

Questions show the evidence, never the extractor's answer: shown a proposal, the model agrees with
it. ``context`` (e.g. a document brief) is prepended to every input when given.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from brain.ingest.extract import ExtractedFact, Extraction
from brain.ingest.normalise import (
    ENTITY_OBJECTS,
    LANGUAGES,
    LITERAL_TYPES,
    OBJECT_TYPES,
    SELF_WORDS,
    literal_reason,
)
from brain.llm.decide import Choice, ChoiceQuestion, Decision, PredicateQuestion, Question

SKIP_BELOW = 0.15  # gate: skip a part when p(it says anything about a person) is under this
ACT_ABOVE = 0.8  # checks: override the extractor only at or above this confidence
SNIPPET_CHARS = 400  # the evidence shown with a check (sentences naming the item)
GATE_WORDS = 24  # the gate sees headings and this many first words of every other block
# A screened item at or above this probability gets its own question. Two suspects in one group
# share the mass (hobbies typed as projects: 0.73 and 0.16), so the flag sits low; a false flag
# costs one question.
SCREEN_FLAG = 0.15
# One screening round per group: a second one, after the flagged item is removed, mostly flagged
# the next least typical item (a project among people and places).
# Groups of at most this many items: in a screen of 11 names one suspect took the mass and two
# others (0.18, under the flag) were missed.
SCREEN_GROUP = 6
# A screen of two items almost always flags one ("none of them" rarely wins a small set: a past
# employer next to the current one at 0.99), so up to two items are asked about directly.
DIRECT_UP_TO = 2
NONE = "none of them"

GATE = PredicateQuestion(
    name="about_a_person",
    instructions=(
        "Does this excerpt say anything about its author or the person it describes "
        "(work, education, projects, people, places, interests, values or habits)?"
    ),
)

# What a new entity is. Page kinds keep the extractor's entity types; the others are values.
KINDS = (
    Choice(value="person"),
    Choice(value="org", description="organisation: company, university, school or club"),
    Choice(value="team", description="team or department"),
    Choice(
        value="project",
        description=(
            "work the person builds, leads or takes part in, including their own app or library"
        ),
    ),
    Choice(value="place"),
    Choice(value="event"),
    Choice(value="tool", description="tool, product, dataset or technology the person uses"),
    Choice(value="topic", description="topic or field"),
    Choice(value="activity", description="activity, hobby or sport"),
)
PAGE_KINDS = frozenset({"person", "org", "project", "place", "event"})
VALUE_KINDS = frozenset({"team", "tool", "topic", "activity"})  # literal values, never pages
SKILL_PREDICATES = ("has_skill", "interest", "habit", "likes")
USES = (  # a work-related interest is not a hobby: two options sent it to has_skill; without
    # "other" the owner's own library listed as an interest became a skill (0.88)
    Choice(value="skill", description="a professional skill or tool the person can use"),
    Choice(value="hobby", description="a hobby, sport or leisure activity"),
    Choice(
        value="interest",
        description="a subject the person reads about, follows or is curious about",
    ),
    Choice(
        value="other",
        description="none of these: a project, an organisation, a place or something else",
    ),
)


class DecisionModel(Protocol):
    def decide(self, text: str, questions: Sequence[Question]) -> list[Decision]: ...


def decision_input(text: str, context: str = "") -> str:
    return f"{context.strip()}\n\n{text}" if context.strip() else text


# --- gate ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    decision: Decision
    seconds: float

    @property
    def p_yes(self) -> float | None:
        return self.decision.p_yes

    @property
    def skip(self) -> bool:
        p = self.decision.p_yes
        return p is not None and p < SKIP_BELOW


def compact(text: str, words: int = GATE_WORDS) -> str:
    """Headings, and the first ``words`` words of every other block: enough to tell a tutorial or
    a privacy notice from a CV, at a third of the tokens of a long part."""
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        flat = " ".join(block.split())
        if not flat:
            continue
        if flat.startswith("#"):
            out.append(flat)
            continue
        first = flat.split()
        out.append(" ".join(first[:words]) + (" …" if len(first) > words else ""))
    return "\n".join(out)


def gate(llm: DecisionModel, text: str, *, context: str = "") -> Gate:
    start = time.perf_counter()
    [decision] = llm.decide(decision_input(compact(text), context), [GATE])
    return Gate(decision, time.perf_counter() - start)


# --- evidence -----------------------------------------------------------------------------------

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def _units(text: str) -> list[tuple[str, str]]:
    """``(heading, unit)``: each list item / table row / sentence with the heading above it."""
    out: list[tuple[str, str]] = []
    heading = ""
    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.strip().split("\n") if ln.strip()]
        if not lines:
            continue
        if len(lines) == 1 and lines[0].startswith("#"):
            heading = lines[0]
            continue
        if all(re.match(r"^(?:[-*+]|\d+[.)]|\|)\s?", ln) for ln in lines):
            out += [(heading, ln) for ln in lines]
        else:
            out += [(heading, s) for s in _SENTENCE.split(" ".join(lines)) if s.strip()]
    return out


def _mentions(text: str, name: str) -> bool:
    """``name`` occurs in ``text`` as whole words (line breaks and runs of spaces ignored)."""
    words = r"\s+".join(map(re.escape, name.split()))
    return bool(words) and bool(re.search(rf"(?<!\w){words}(?!\w)", text, re.I))


def evidence(text: str, names: str | Sequence[str], cap: int = SNIPPET_CHARS) -> str:
    """The sentences / list lines of ``text`` naming any of ``names``, each run under its
    heading, up to about ``cap`` characters per name; the whole text if none is named (the
    model paraphrased it)."""
    names = [names] if isinstance(names, str) else list(names)
    budget = cap * len(names)
    lines: list[str] = []
    heading = None
    size = 0
    for h, unit in _units(text):
        if not any(_mentions(unit, n) for n in names):
            continue
        if lines and size + len(unit) > budget:
            break
        if h != heading and h:
            lines.append(h)
        heading = h
        lines.append(unit)
        size += len(unit)
    return "\n".join(lines) if lines else text


# --- checks -------------------------------------------------------------------------------------


@dataclass
class Checks:
    """What the checks of one part decided. ``kinds``: folded name → kind for confident
    overrides (a page kind, or ``team``/``tool``/``topic``/``activity`` for a value)."""

    kinds: dict[str, str] = field(default_factory=dict)
    rerouted: int = 0  # facts whose predicate changed
    overridden: int = 0  # entities whose kind changed
    asked: int = 0  # questions (screens and typed questions)
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    decisions: list[tuple[str, Decision]] = field(default_factory=list)  # (item(s), decision)


def _fold(text: str) -> str:
    return " ".join(text.split()).casefold()


def new_entities(
    extraction: Extraction, *, known: Iterable[str], owner: Iterable[str], literal: Mapping
) -> list[tuple[str, str]]:
    """``(name, extractor's kind)`` of the entities the normaliser would create a page for: not
    the owner, not already known (vault or an earlier part), not a value (tool, team, id)."""
    owner = list(owner)
    skip = {_fold(n) for n in known} | {_fold(n) for n in owner} | SELF_WORDS | set(literal)
    out: dict[str, tuple[str, str]] = {}
    declared: set[str] = set()
    for entity in extraction.entities:
        names = [entity.name, *entity.aliases]
        declared |= {_fold(n) for n in names}
        name = " ".join(entity.name.split())
        if (
            not name
            or entity.type in LITERAL_TYPES
            or any(_fold(n) in skip for n in names)
            or literal_reason(name)
        ):
            continue
        out.setdefault(_fold(name), (name, entity.type))
    for fact in extraction.facts:  # works_at "Fabrikam" with no entity → an org page
        name = " ".join(fact.object.split())
        kind = ENTITY_OBJECTS.get(fact.predicate)
        about_owner = _fold(fact.subject) in SELF_WORDS | {_fold(n) for n in owner}
        new = name and _fold(name) not in skip | declared and not literal_reason(name)
        if kind and about_owner and new:
            out.setdefault(_fold(name), (name, kind))
    return list(out.values())


@dataclass(frozen=True)
class Screen:
    """One screening question for a set of items."""

    instructions: str
    none: str  # what "none of them" means


SCREEN_ENTITIES = Screen(  # the project clause: migrations and own libraries read as "products"
    "Which of these names is a team or department, a tool or technology the person uses, a topic "
    "or field, or an activity or hobby, rather than a person, an organisation, a project "
    "(including the person's own app or library), a place or an event?",
    "each is a person, organisation, project, place or event",
)
SCREEN_EMPLOYERS = Screen(
    "Which of these is not an organisation the person works or worked for, but one they studied "
    "at, joined as a member or volunteer, or only mention?",
    "each is an organisation the person works or worked for",
)
RELATIONS = (
    Choice(value="employer", description="the person works or worked there"),
    Choice(value="school", description="the person studied there"),
    Choice(value="member", description="the person joined it as a member or volunteer"),
    Choice(value="other", description="it is only mentioned, or the text does not mention it"),
)
ROUTE_EMPLOYER = {"school": "studied_at", "member": "member_of"}  # "other": dropped
SCREEN_SKILLS = Screen(
    "Which of these is a hobby, a leisure activity or a subject of interest rather than a "
    "professional skill or tool?",
    "all of them are professional skills or tools",
)
SCREEN_LEISURE = Screen(
    "Which of these is a professional skill or tool used for work rather than a hobby, "
    "interest or habit?",
    "none of them is a professional skill or tool",
)


# Story details: a write-up of an experiment, a trip or a project yields "duties" and "skills"
# that are really one-off events, results, lessons and plans ("Quick evaluation underestimated
# the full run", "Plans to point the AI at feature engineering next round", the mountains of
# one hike as interests). Facts of these predicates about the person are screened and asked.
STORY_PREDICATES = ("responsibility", "has_skill", "interest", "habit", "likes")
STORY_ACTION = "drop"  # what a confident "story" answer does: "drop", "note" or "off" (no check)
SCREEN_STORY = Screen(
    "Which of these statements records one event or step of a story the text tells, a finding, "
    "an opinion formed, a lesson learnt or a plan for next time, rather than the person's job, "
    "duties, achievements, skills, habits or interests?",
    "each records the person's job, duties, achievements, skills, habits or interests",
)
# Measured on 29 statements from the fictional eval documents (13 story details, 16 facts), at
# 0.8: these four options caught 9 story details and lost no fact; "duty / trait / story" caught
# 8 and lost a CV achievement; "still true a year later?" and "would it belong in a profile?"
# (yes/no) caught 3-5 and lost one fact; "a fact about the person or a story detail?" kept all.
RECORDS = (
    Choice(value="job", description="their job, role, duties or achievements at work"),
    Choice(value="trait", description="a skill, habit, hobby or interest they have"),
    Choice(
        value="event",
        description="one event or step of a story the text tells, or what happened in it",
    ),
    Choice(
        value="lesson",
        description="a finding, an opinion formed, a lesson learnt or a plan for next time",
    ),
)
STORY = frozenset({"event", "lesson"})


def _kind_question(name: str) -> ChoiceQuestion:
    return ChoiceQuestion(
        name="kind",
        instructions=f'What kind of thing is "{name}" in the text?',
        choices=list(KINDS),
    )


def _use_question(value: str) -> ChoiceQuestion:
    return ChoiceQuestion(
        name="use",
        instructions=(
            f'In the text, is "{value}" a skill the person has, a hobby or leisure activity, '
            "or a subject the person is interested in?"
        ),
        choices=list(USES),
    )


class _Asker:
    """Asks questions, timing them and logging every decision in ``checks``."""

    def __init__(self, llm: DecisionModel, text: str, context: str, checks: Checks) -> None:
        self.llm, self.text, self.context, self.checks = llm, text, context, checks
        # item → the names its evidence is found by (a statement is found by its object's name)
        self.names: dict[str, list[str]] = {}

    def ask(self, items: Sequence[str], question: Question) -> Decision:
        start = time.perf_counter()
        source = evidence(self.text, [n for i in items for n in self.names.get(i, [i])])
        [decision] = self.llm.decide(decision_input(source, self.context), [question])
        self.checks.seconds += time.perf_counter() - start
        self.checks.asked += 1
        self.checks.decisions.append((", ".join(items), decision))
        return decision

    def screen(
        self, items: Sequence[str], screen: Screen, label: str, *, named: bool = True
    ) -> list[str]:
        """The items to ask about: those the excerpt never names (``named``: items are names, not
        paraphrased statements), and those a screening flags (at least ``SCREEN_FLAG``). Items
        are screened in groups of at most ``SCREEN_GROUP``; a group of up to ``DIRECT_UP_TO``
        items is not screened (as cheap to ask each)."""
        remaining = [i for i in dict.fromkeys(items) if _fold(i) != _fold(NONE)]
        # a name the excerpt never contains has no evidence here: asked about, never screened
        flagged = [i for i in remaining if named and not _mentions(self.text, i)]
        if flagged:
            self.checks.notes.append(f"check: {label} not named in the part: {flagged}")
        remaining = [i for i in remaining if i not in flagged]
        if len(remaining) <= DIRECT_UP_TO:
            return flagged + remaining
        groups = -(-len(remaining) // SCREEN_GROUP)
        size = -(-len(remaining) // groups)
        for start in range(0, len(remaining), size):
            group = remaining[start : start + size]
            if len(group) <= DIRECT_UP_TO:
                flagged += group
                continue
            question = ChoiceQuestion(
                name="screen",
                instructions=screen.instructions,
                choices=[
                    *(Choice(value=i) for i in group),
                    Choice(value=NONE, description=screen.none),
                ],
            )
            decision = self.ask(group, question)
            if decision.refusal:
                self.checks.notes.append(f"check: screen of {label}: no answer; not checked")
                continue
            top = sorted(decision.probabilities.items(), key=lambda kv: -kv[1])[:3]
            shown = ", ".join(f"{v} {p:.2f}" for v, p in top)
            hits = [i for i in group if decision.probabilities.get(i, 0.0) >= SCREEN_FLAG]
            self.checks.notes.append(
                f"check: screen of {label} ({len(group)}): {shown}"
                + (f" → asking about {hits}" if hits else "")
            )
            flagged += hits
        return flagged


def _route(predicate: str, use: str) -> str | None:
    """The predicate a confident answer moves a fact to (None: keep it). A hobby or a subject
    listed as a skill becomes an interest; a skill listed as an interest, habit or like becomes
    has_skill. Between interest, habit and likes the extractor's choice stands."""
    if use == "skill" and predicate != "has_skill":
        return "has_skill"
    if use in ("hobby", "interest") and predicate == "has_skill":
        return "interest"
    return None


def _a(word: str) -> str:
    return f"{'an' if word[:1] in 'aeiou' else 'a'} {word}"


def _conf(decision: Decision) -> str:
    return f"conf {decision.confidence:.2f}" + (", fallback" if decision.fallback else "")


def check_part(
    llm: DecisionModel,
    text: str,
    extraction: Extraction,
    *,
    known: Iterable[str] = (),
    owner: Iterable[str] = (),
    literal: Mapping[str, str] | None = None,
    context: str = "",
) -> Checks:
    """Screen the part's new entities and skill/leisure values, ask about the flagged ones and
    apply confident answers: skill/hobby facts are re-routed in ``extraction`` (in place),
    entity kinds are returned in ``Checks.kinds`` for the normaliser.

    ``known``: names already resolved (vault, earlier parts); ``literal``: folded names already
    known to be values."""
    checks = Checks()
    owner = list(owner)
    asker = _Asker(llm, text, context, checks)
    ground(extraction, text, SELF_WORDS | {_fold(n) for n in owner}, checks)

    entities = new_entities(extraction, known=known, owner=owner, literal=literal or {})
    proposed = {_fold(n): k for n, k in entities}
    for name in asker.screen([n for n, _ in entities], SCREEN_ENTITIES, "new entities"):
        decision = asker.ask([name], _kind_question(name))
        was = proposed[_fold(name)]
        if decision.refusal or decision.choice not in {c.value for c in KINDS}:
            checks.notes.append(f"check: {name!r}: no answer; kept the extractor's {was}")
        elif decision.choice == was:
            checks.notes.append(f"check: {name!r} is {_a(was)} ({_conf(decision)}): agrees")
        elif decision.confidence < ACT_ABOVE:
            checks.notes.append(
                f"check: {name!r} kind {decision.choice} ({_conf(decision)}) under "
                f"{ACT_ABOVE}: kept the extractor's {was}"
            )
        else:
            checks.kinds[_fold(name)] = decision.choice
            checks.overridden += 1
            becomes = "a value, no page" if decision.choice in VALUE_KINDS else "its page type"
            checks.notes.append(
                f"check: {name!r} is {_a(decision.choice)} ({_conf(decision)}), not {_a(was)}: "
                f"{becomes}"
            )

    self_names = SELF_WORDS | {_fold(n) for n in owner}
    # these may remove facts: before the skill facts are indexed
    _check_languages(asker, extraction, self_names)
    _check_employers(asker, extraction, self_names)
    if STORY_ACTION != "off":
        _check_stories(asker, extraction, self_names)
    facts: dict[str, list[int]] = {}  # folded value → indices of its facts
    names: dict[str, str] = {}
    for i, fact in enumerate(extraction.facts):
        value = " ".join(fact.object.split())
        if fact.predicate in SKILL_PREDICATES and value and _fold(fact.subject) in self_names:
            facts.setdefault(_fold(value), []).append(i)
            names.setdefault(_fold(value), value)
    skills = [v for k, v in names.items() if extraction.facts[facts[k][0]].predicate == "has_skill"]
    leisure = [v for k, v in names.items() if v not in skills]
    flagged = asker.screen(skills, SCREEN_SKILLS, "skills")
    flagged += asker.screen(leisure, SCREEN_LEISURE, "interests and habits")
    for value in flagged:
        decision = asker.ask([value], _use_question(value))
        for i in facts[_fold(value)]:
            fact = extraction.facts[i]
            target = None if decision.refusal else _route(fact.predicate, decision.choice or "")
            if decision.refusal:
                checks.notes.append(f"check: {value!r}: no answer; kept {fact.predicate}")
            elif target is None:
                checks.notes.append(
                    f"check: {value!r} is {decision.choice} ({_conf(decision)}): "
                    f"kept {fact.predicate}"
                )
            elif decision.confidence < ACT_ABOVE:
                checks.notes.append(
                    f"check: {value!r} looks like {_a(decision.choice)} ({_conf(decision)}) under "
                    f"{ACT_ABOVE}: kept {fact.predicate}"
                )
            else:
                checks.notes.append(
                    f"check: {value!r} is {_a(decision.choice)} ({_conf(decision)}): "
                    f"{fact.predicate} → {target}"
                )
                fact.predicate = target  # type: ignore[assignment]
                fact.text = ""  # rendered from the template of the new predicate
                checks.rerouted += 1
    return checks


def statement(fact: ExtractedFact) -> str:
    """The fact as the extractor stated it ("Learns that an AI researcher needs clear
    objectives"), or its predicate and value when it gave no text."""
    text = " ".join(fact.text.split())
    return text or f"{fact.predicate.replace('_', ' ')}: {' '.join(fact.object.split())}"


def _check_stories(asker: _Asker, extraction: Extraction, self_names: set[str]) -> None:
    """Duties, skills, interests, habits and likes that are details of a story (an event, a
    result, a lesson, a plan): screen the statements, ask about the flagged ones and drop, or turn
    into a ``note``, those confidently a story detail (``STORY_ACTION``)."""
    checks = asker.checks
    by_statement: dict[str, list[int]] = {}
    shown: dict[str, str] = {}
    for i, fact in enumerate(extraction.facts):
        language = _fold(" ".join(fact.object.split())) in LANGUAGES  # routed to speaks later
        if (
            fact.predicate in STORY_PREDICATES
            and _fold(fact.subject) in self_names
            and not language
        ):
            said = statement(fact)
            by_statement.setdefault(_fold(said), []).append(i)
            if _fold(said) not in shown:
                shown[_fold(said)] = said
                value = " ".join(fact.object.split())
                asker.names[said] = [n for n in (value, said) if n]
    drop: set[int] = set()
    for said in asker.screen(list(shown.values()), SCREEN_STORY, "statements", named=False):
        question = ChoiceQuestion(
            name="records",
            instructions=f'What does this statement about the person record: "{said}"?',
            choices=list(RECORDS),
        )
        decision = asker.ask([said], question)
        p = sum(decision.probabilities.get(c, 0.0) for c in STORY)
        if decision.refusal:
            checks.notes.append(f"check: {said!r}: no answer: kept")
            continue
        if p < ACT_ABOVE:
            under = f" under {ACT_ABOVE}" if decision.choice in STORY else ""
            checks.notes.append(
                f"check: {said!r} records {decision.choice} (p(story)={p:.2f}){under}: kept"
            )
            continue
        for i in by_statement[_fold(said)]:
            fact = extraction.facts[i]
            if STORY_ACTION == "note":
                fact.predicate, fact.object, fact.text = "note", said, said  # type: ignore[assignment]
            else:
                drop.add(i)
            checks.rerouted += 1
        then = "a note" if STORY_ACTION == "note" else "dropped"
        checks.notes.append(f"check: {said!r} is a story detail (p={p:.2f}): {then}")
    if drop:
        extraction.facts[:] = [f for i, f in enumerate(extraction.facts) if i not in drop]


def _check_languages(asker: _Asker, extraction: Extraction, self_names: set[str]) -> None:
    """A ``speaks`` value that is not a known language ("speaks email" from "reach me by
    email"): ask whether it is one, and drop the fact when the answer is a confident no."""
    checks = asker.checks
    keep = []
    for fact in extraction.facts:
        value = " ".join(fact.object.split())
        if (
            fact.predicate != "speaks"
            or _fold(fact.subject) not in self_names
            or not value
            or _fold(value) in LANGUAGES
        ):
            keep.append(fact)
            continue
        question = PredicateQuestion(
            name="language", instructions=f'In the text, is "{value}" a language the person speaks?'
        )
        decision = asker.ask([value], question)
        p = decision.p_yes
        if p is not None and p <= 1 - ACT_ABOVE:
            checks.notes.append(f"check: {value!r} is not a language (p={p:.2f}): dropped speaks")
            checks.rerouted += 1
            continue
        shown = "no answer" if p is None else f"p={p:.2f}"
        checks.notes.append(f"check: {value!r} a language? {shown}: kept speaks")
        keep.append(fact)
    extraction.facts[:] = keep


def _check_employers(asker: _Asker, extraction: Extraction, self_names: set[str]) -> None:
    """``works_at`` an organisation the person studied at or joined ("the library club I joined"):
    screen the employers, ask the relation of a flagged one, re-route on a confident answer."""
    checks = asker.checks
    by_name: dict[str, list[int]] = {}
    names: dict[str, str] = {}
    for i, fact in enumerate(extraction.facts):
        value = " ".join(fact.object.split())
        about_owner = _fold(fact.subject) in self_names
        if (
            fact.predicate == "works_at"
            and value
            and about_owner
            and _fold(value) not in self_names
        ):
            by_name.setdefault(_fold(value), []).append(i)
            names.setdefault(_fold(value), value)
    drop: set[int] = set()
    for name in asker.screen(list(names.values()), SCREEN_EMPLOYERS, "employers"):
        question = ChoiceQuestion(
            name="relation",
            instructions=f'In the text, what is the person\'s relation to "{name}"?',
            choices=list(RELATIONS),
        )
        decision = asker.ask([name], question)
        if decision.refusal or decision.choice in (None, "employer"):
            shown = "no answer" if decision.refusal else f"employer ({_conf(decision)})"
            checks.notes.append(f"check: {name!r}: {shown}: kept works_at")
            continue
        if decision.confidence < ACT_ABOVE:
            checks.notes.append(
                f"check: {name!r} looks like {decision.choice} ({_conf(decision)}) under "
                f"{ACT_ABOVE}: kept works_at"
            )
            continue
        target = ROUTE_EMPLOYER.get(decision.choice)
        for i in by_name[_fold(name)]:
            fact = extraction.facts[i]
            if target is None:
                drop.add(i)
            else:
                fact.predicate = target  # type: ignore[assignment]
                fact.text = ""
            checks.rerouted += 1
        then = f"works_at → {target}" if target else "dropped works_at"
        checks.notes.append(f"check: {name!r} is {decision.choice} ({_conf(decision)}): {then}")
    if drop:
        extraction.facts[:] = [f for i, f in enumerate(extraction.facts) if i not in drop]


def _named(text: str, name: str, aliases: Sequence[str] = ()) -> bool:
    """``text`` names the entity: its name or an alias, or a capitalised word of its name of 4+
    letters ("Northwind" for "Northwind Labs"), as whole words."""
    if any(_mentions(text, n) for n in (name, *aliases)):
        return True
    words = [w for w in re.findall(r"[A-Z][\w'-]{3,}", name)]
    return any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", text) for w in words)


def named(text: str, name: str, aliases: Sequence[str] = ()) -> bool:
    """Public form of ``_named`` (used by the memory pipeline's grounding)."""
    return _named(text, name, aliases)


def ground(extraction: Extraction, text: str, self_names: set[str], checks: Checks) -> None:
    """Drop facts whose entities the part never names: a fact's subject (unless the owner) and,
    for predicates that point at a page (``works_at``, ``worked_on``, relations, …), its object.
    Values (skills, duties, habits) are not checked: paraphrase is normal there."""
    aliases: dict[str, list[str]] = {}
    for entity in extraction.entities:
        aliases[_fold(entity.name)] = list(entity.aliases)
    keep, dropped = [], []
    for fact in extraction.facts:
        names = []
        if _fold(fact.subject) not in self_names:
            names.append(fact.subject)
        if fact.predicate in OBJECT_TYPES and _fold(fact.object) not in self_names:
            names.append(fact.object)
        missing = [n for n in names if n.strip() and not _named(text, n, aliases.get(_fold(n), []))]
        if missing:
            dropped.append(f"{fact.predicate} {missing[0]!r}")
        else:
            keep.append(fact)
    if dropped:
        checks.notes.append(
            f"check: dropped {len(dropped)} fact(s) about entities the part never names: "
            + ", ".join(dropped)
        )
        checks.rerouted += len(dropped)
        extraction.facts[:] = keep
