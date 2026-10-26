"""Turn per-chunk extractions into one validated ``ChangeSet`` for a document.

Everything the model returns is treated as a proposal: names are resolved to entity ids in code
(known aliases first, then entities named earlier in the same document, then new ids), dates are
normalised or dropped, duplicates are merged, and entities no kept fact refers to are dropped.

A ~3B-active model still slips, so guards repair or drop what breaks the schema's shape:

- a predicate → subject/object type table (``works_at`` points at an org, relations go from a
  person to the owner, ``has_skill`` is a literal, …): a violation is repaired (``works_at`` a
  project → ``worked_on``, a verb-led ``works_at`` → ``responsibility``, an entity where a literal
  belongs → the literal) or dropped;
- tools, products, datasets, record ids and teams are literal values, never pages;
- counters read as 0 ("Has 0 years of …") are dropped;
- a single-valued fact in the past tense without an end ("Lived in Seville") becomes a ``note``
  so it does not contradict the present;
- fact text with a pronoun, the owner's name, no object, or too long is rendered from a
  per-predicate template.

Every repair is listed in ``Normalised.notes``. What the policy queues for review is decided
later by the writer.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

from brain.ingest.brief import PAGE_TYPES, Brief, BriefName, homes, lookup, name_index
from brain.ingest.dates import MONTHS
from brain.ingest.extract import ExtractedFact, Extraction
from brain.vault.models import RELATIONSHIP_PREDICATES, SELF_ID, Kind, is_multi_valued
from brain.vault.ops import AddAlias, AddFact, ChangeSet, CreateEntity, Operation
from brain.vault.raw import slugify
from brain.vault.store import SELF_PAGE, Vault

TYPE_DIRS = {
    "person": "people",
    "org": "orgs",
    "project": "projects",
    "place": "places",
    "topic": "topics",
    "event": "events",
}
LITERAL_TYPES = {"tool"}  # extraction types that are values, never pages
SELF_WORDS = {"self", "owner", "the owner", "me", "i", "myself"}
# A literal object of these predicates names an entity of this type: create its page.
ENTITY_OBJECTS = {"works_at": "org", "studied_at": "org"}
# Objects that are values, not things: they link only to pages that already exist in the vault,
# never to entities the model proposed in this document (a tool is not a project page).
_LITERAL = "name born nationality speaks role team responsibility degree certification has_skill"
LITERAL_OBJECTS = frozenset(
    f"{_LITERAL} habit values interest likes goal status deadline uses note".split()
)
# What an entity object must be (after relations are turned towards the owner).
OBJECT_TYPES: dict[str, frozenset[str]] = {
    **{p: frozenset({"org"}) for p in ("works_at", "studied_at", "member_of")},
    **{p: frozenset({"place"}) for p in ("lives_in", "hometown", "located_in")},
    "worked_on": frozenset({"project"}),
    "manager": frozenset({"person"}),
    "owner": frozenset({"person", "self"}),
    "stakeholder": frozenset({"person", "org", "self"}),
    "part_of": frozenset({"org", "project", "place", "event"}),
    **{p: frozenset({"self"}) for p in RELATIONSHIP_PREDICATES},
}
PROJECT_PREDICATES = frozenset({"owner", "status", "deadline", "stakeholder", "uses"})
# Predicates whose subject is the other party when the owner is involved: a relation points
# from the person to the owner, ownership from the project to the owner.
TOWARDS_OWNER = RELATIONSHIP_PREDICATES | {"owner", "stakeholder"}
LEADING_SUBJECT = re.compile(r"^(?:he|she|they|it|i)\s+(?=\w)", re.I)
PRONOUN = re.compile(
    r"\b(?:he|she|him|his|her|hers|they|them|their|theirs|i|me|my|mine|myself|we|our|ours|"
    r"himself|herself|themselves)\b",
    re.I,
)
OWNER_WORDS = re.compile(r"\bthe owner\b|\bowner's\b", re.I)
MAX_TEXT = 160
TEAM = re.compile(r"\b(?:team|squad)\b", re.I)
# A record id such as "TL4521": upper-case letters and digits, both present.
RECORD_ID = re.compile(r"^(?=[A-Z0-9._-]*\d)(?=[A-Z0-9._-]*[A-Z])[A-Z0-9][A-Z0-9._-]{2,11}$")
ZERO_COUNT = re.compile(r"(?<![\w.,])0\+?\s+[a-z]", re.I)  # "0 years of", "0 degrees"
PAST = re.compile(
    r"^(?:lived|worked|was|were|spent|moved|stayed|resided|had been|grew up)\b"
    r"|\b(?:previously|formerly|used to|in the past)\b",
    re.I,
)
# "Works at X", "Is a data engineer at X": the employer, not a duty.
EMPLOYMENT_START = {"is", "was", "has", "had", "joined", "employed", "currently", "now", "holds"}
WORK_VERBS = {"work", "works", "worked", "working"}
PAST_SINGLE = ("lives_in", "role", "team", "manager")  # a stale value would contradict today's
LANGUAGES = frozenset(
    [
        "arabic",
        "bengali",
        "catalan",
        "chinese",
        "mandarin",
        "cantonese",
        "czech",
        "danish",
        "dutch",
        "english",
        "finnish",
        "french",
        "german",
        "greek",
        "hebrew",
        "hindi",
        "hungarian",
        "indonesian",
        "italian",
        "japanese",
        "korean",
        "norwegian",
        "persian",
        "polish",
        "portuguese",
        "romanian",
        "russian",
        "spanish",
        "swahili",
        "swedish",
        "thai",
        "turkish",
        "ukrainian",
        "urdu",
        "vietnamese",
        "basque",
        "galician",
        "welsh",
        "irish",
    ]
)
# (present, past) templates for fact text; {o} is the object's name.
TEMPLATES: dict[str, tuple[str, str]] = {
    "name": ("Is called {o}", "Was called {o}"),
    "born": ("Born {o}", "Born {o}"),
    "hometown": ("Grew up in {o}", "Grew up in {o}"),
    "lives_in": ("Lives in {o}", "Lived in {o}"),
    "nationality": ("Has {o} nationality", "Had {o} nationality"),
    "speaks": ("Speaks {o}", "Spoke {o}"),
    "works_at": ("Works at {o}", "Worked at {o}"),
    "role": ("Works as {o}", "Worked as {o}"),
    "manager": ("Reports to {o}", "Reported to {o}"),
    "worked_on": ("Works on {o}", "Worked on {o}"),
    "studied_at": ("Studied at {o}", "Studied at {o}"),
    "degree": ("Holds the degree {o}", "Holds the degree {o}"),
    "certification": ("Holds the certification {o}", "Held the certification {o}"),
    "has_skill": ("Skilled in {o}", "Was skilled in {o}"),
    "habit": ("Does {o} regularly", "Used to do {o}"),
    "values": ("Values {o}", "Valued {o}"),
    "interest": ("Is interested in {o}", "Was interested in {o}"),
    "likes": ("Likes {o}", "Liked {o}"),
    "goal": ("Aims for {o}", "Aimed for {o}"),
    "owner": ("Led by {o}", "Was led by {o}"),
    "status": ("Is {o}", "Was {o}"),
    "deadline": ("Due {o}", "Was due {o}"),
    "stakeholder": ("Involves {o}", "Involved {o}"),
    "uses": ("Uses {o}", "Used {o}"),
    "friend_of": ("Friend of {o}", "Was a friend of {o}"),
    "family_of": ("Family of {o}", "Family of {o}"),
    "partner_of": ("Partner of {o}", "Was the partner of {o}"),
    "colleague_of": ("Colleague of {o}", "Was a colleague of {o}"),
    "mentor_of": ("Mentor of {o}", "Was a mentor of {o}"),
    "knows": ("Knows {o}", "Knew {o}"),
    "member_of": ("Member of {o}", "Was a member of {o}"),
    "located_in": ("Located in {o}", "Was located in {o}"),
    "part_of": ("Part of {o}", "Was part of {o}"),
    "note": ("{o}", "{o}"),
}
VERB_PHRASE = ("responsibility", "habit", "goal")  # objects that may already be a verb phrase
# (self, p, X): the life-map region X belongs to. Privacy follows from the region.
OBJECT_REGIONS = {
    "studied_at": "education",
    "worked_on": "projects",
    "manager": "relationships.work",
}
# (X, p, self): the region of X.
SUBJECT_REGIONS = {
    "friend_of": "relationships.friends",
    "family_of": "relationships.family",
    "colleague_of": "relationships.work",
    "mentor_of": "relationships.work",
    "owner": "projects",
}
TYPE_REGIONS = {"project": "projects"}
OPEN_ENDED = {"present", "now", "current", "currently", "ongoing", "today", "to date"}


def clean(text: str) -> str:
    return " ".join(text.split()).replace("::", ":").strip()


def fold(text: str) -> str:
    return clean(text).casefold()


def normalise_date(value: str) -> str | None:
    """``YYYY`` / ``YYYY-MM`` / ``YYYY-MM-DD``, also from "March 2025", "2025/03", "03/2025".

    Returns None for empty, open-ended ("present") or unparseable values.
    """
    v = fold(value).rstrip(".")
    if not v or v in OPEN_ENDED:
        return None
    year = month = day = None
    if m := re.fullmatch(r"(\d{4})(?:[-/.](\d{1,2})(?:[-/.](\d{1,2}))?)?", v):
        year, month, day = m.group(1), m.group(2), m.group(3)
    elif m := re.fullmatch(r"(\d{1,2})[-/.](\d{4})", v):
        month, year = m.group(1), m.group(2)
    elif (m := re.fullmatch(r"([a-z]+)\.?,?\s+(\d{4})", v)) and m.group(1) in MONTHS:
        month, year = str(MONTHS[m.group(1)]), m.group(2)
    else:
        return None
    try:
        date(int(year), int(month or 1), int(day or 1))
    except ValueError:
        return None
    out = year
    if month:
        out += f"-{int(month):02d}"
        if day:
            out += f"-{int(day):02d}"
    return out


def literal_reason(name: str) -> str | None:
    """Why a name is a value and never a page: a team or a record id."""
    name = clean(name)
    if TEAM.search(name):
        return "a team"
    if RECORD_ID.match(name):
        return "a record id"
    return None


def is_duty(text: str, value: str) -> bool:
    """A ``works_at`` fact that describes work done ("Works on models that…", "Experiments
    with…") rather than the employer."""
    words = re.findall(r"[\w'-]+", text.casefold())
    employer = rf"\b(?:at|for|with)\s+(?:the\s+)?{re.escape(value.casefold())}"
    if value and re.search(employer, text.casefold()):
        return False
    if words:
        first, second = words[0], (words[1] if len(words) > 1 else "")
        if first in EMPLOYMENT_START:
            return False
        if first in WORK_VERBS:
            return second not in ("at", "for", "as", "")
        return True
    return len(value.split()) > 6


def _sentence(text: str) -> str:
    text = clean(text).rstrip(".;")
    return text[:1].upper() + text[1:]


def _verb_led(text: str) -> bool:
    words = text.split()
    return len(words) >= 2 and words[0].casefold().endswith(("s", "ed")) and words[0].isalpha()


def template_text(p: str, o: str, *, past: bool) -> str:
    if p in VERB_PHRASE and _verb_led(o):
        return _sentence(o)
    if p == "responsibility":
        return f"Responsible for {o}"
    if p == "habit" and len(o.split()) > 1:  # "go bouldering": no verb to put in front
        return f"{'Had' if past else 'Has'} the habit: {o}"
    if p == "team":
        o = o if TEAM.search(o) else f"{o} team"
        return f"{'Was' if past else 'Is'} on the {o}"
    present, past_form = TEMPLATES.get(p, ("{o}", "{o}"))
    return (past_form if past else present).format(o=o)


def fact_text(text: str, fallback: str) -> str:
    """Drop a leading pronoun ("He speaks Spanish" → "Speaks Spanish")."""
    text = LEADING_SUBJECT.sub("", clean(text)) or fallback
    return text[:1].upper() + text[1:]


def text_problem(text: str, *, o_name: str, owner_names: Sequence[str], about_self: bool) -> str:
    """Why model-written fact text cannot be kept ("" if it can)."""
    if not text:
        return "empty"
    if PRONOUN.search(text):
        return "a pronoun"
    if about_self and (
        OWNER_WORDS.search(text)
        or any(re.search(rf"(?<!\w){re.escape(n)}(?!\w)", text, re.I) for n in owner_names)
    ):
        return "the owner's name"
    if len(text) > MAX_TEXT:
        return "too long"
    words = [w for w in re.findall(r"\w+", o_name.casefold()) if len(w) > 2] or [o_name.casefold()]
    if o_name and not any(w in text.casefold() for w in words):
        return "no object"
    return ""


@dataclass
class _Planned:
    op: CreateEntity
    regions: list[str] = field(default_factory=list)


@dataclass
class _Fact:
    subject: str
    p: str
    o: str  # "[[id]]" or a literal
    text: str  # as the model wrote it (cleaned); final text is rendered at the end
    start: str | None
    end: str | None

    @property
    def target(self) -> str | None:
        return self.o[2:-2] if self.o.startswith("[[") else None


@dataclass
class Normalised:
    changeset: ChangeSet
    notes: list[str]  # what was dropped or changed, and why
    owner_names: list[str]


class _Resolver:
    def __init__(self, vault: Vault, owner_names: Iterable[str]) -> None:
        self.pages = {p.subject: p for p in vault.pages() if p.subject != SELF_ID}
        self.aliases = {k: v for k, v in vault.alias_index().items()}
        self.owner = {fold(n) for n in owner_names if n.strip()} | SELF_WORDS
        self.doc_names: dict[str, str] = {}  # names seen in this document → entity id
        self.planned: dict[str, _Planned] = {}  # new entity id → create op
        self.literals: dict[str, str] = {}  # folded name → why it is a value, not a page
        self.entity_kinds: dict[str, str] = {}  # folded name → page type from a typed check
        self.brief_names: dict[str, BriefName] = {}  # folded name / alias → document brief entry
        self.teams: set[str] = set()  # folded values of the document's team facts

    def is_self(self, name: str) -> bool:
        return fold(name) in self.owner or name.strip() in (SELF_ID, SELF_PAGE, f"[[{SELF_ID}]]")

    def lookup(self, name: str) -> str | None:
        key = fold(name).strip("[]")
        if not key:
            return None
        if self.is_self(name):
            return SELF_ID
        if key in self.doc_names:
            return self.doc_names[key]
        known = [i for i in self.aliases.get(key, []) if i != SELF_ID]
        if len(known) == 1:
            return known[0]
        if key in self.pages or key in self.planned:
            return key
        entry = lookup(self.brief_names, name)  # a variant of a name in the document brief
        if entry and fold(entry.name) != key:
            return self.lookup(entry.name)
        return None

    def kind(self, entity_id: str) -> str:
        if entity_id == SELF_ID:
            return "self"
        if entity_id in self.planned:
            return self.planned[entity_id].op.type
        page = self.pages.get(entity_id)
        return page.type if page else "unknown"

    def title(self, entity_id: str) -> str:
        if entity_id in self.planned:
            return self.planned[entity_id].op.title
        page = self.pages.get(entity_id)
        return page.title if page else entity_id

    def literal(self, name: str) -> str | None:
        return self.literals.get(fold(name)) or literal_reason(name)

    def plan(self, name: str, etype: str, aliases: Sequence[str]) -> str:
        """Id for an entity named in the document, planning a new page if none matches."""
        title = clean(name)
        existing = self.lookup(title)
        if existing is None:
            existing = next((i for a in aliases if (i := self.lookup(a))), None)
        entity_id = existing or f"{TYPE_DIRS[etype]}/{slugify(title)}"
        if entity_id not in self.pages and entity_id != SELF_ID:
            planned = self.planned.get(entity_id)
            if planned is None:
                op = CreateEntity(id=entity_id, type=etype, title=title)
                planned = self.planned[entity_id] = _Planned(op)
            names = {fold(n) for n in (planned.op.title, *planned.op.aliases)}
            for alias in (title, *aliases):
                if (a := clean(alias)) and fold(a) not in names and not self.is_self(a):
                    planned.op.aliases.append(a)
                    names.add(fold(a))
        for alias in (title, *aliases):
            if fold(alias) and not self.is_self(alias):
                self.doc_names.setdefault(fold(alias), entity_id)
        return entity_id


def _merge_start(a: str | None, b: str | None) -> str | None:
    """The most precise start wins ("2024-04" over a vague "2022" from a summary line); among
    equally precise ones the earliest (two stints at one employer span both)."""
    starts = [d for d in (a, b) if d]
    if not starts:
        return None
    best = max(len(d) for d in starts)
    return min(d for d in starts if len(d) == best)


def _month(value: str | None, *, end: bool) -> int | None:
    """Months since year 0 of a ``YYYY`` / ``YYYY-MM`` date (a year ends in December)."""
    if not value:
        return None
    month = int(value[5:7]) if len(value) >= 7 else (12 if end else 1)
    return int(value[:4]) * 12 + month - 1


def _apart(a: _Fact, b: _Fact) -> bool:
    """Two dated stints with more than a month between them (an internship, then a job years
    later at the same employer). Touching or overlapping dates are one stint."""

    def gap(first: _Fact, second: _Fact) -> bool:
        end, start = _month(first.end, end=True), _month(second.start, end=False)
        return end is not None and start is not None and start - end > 1

    return gap(a, b) or gap(b, a)


def _merge(
    facts: list[_Fact], notes: list[str], stints: frozenset[str] = frozenset()
) -> list[_Fact]:
    """One fact per (subject, p, o): the most precise start (see ``_merge_start``), and no end
    if any of the duplicates is still true. ``stints``: folded objects the document brief shows
    in separate periods (two stints at one employer); their facts stay separate when their dates
    leave a gap (``_apart``)."""
    out: dict[tuple[str, str, str, int], _Fact] = {}
    for fact in facts:
        base = (fact.subject, fact.p, fold(fact.o))
        n = 0
        while (prev := out.get((*base, n))) and fold(fact.o) in stints and _apart(prev, fact):
            n += 1
        key = (*base, n)
        if prev := out.get(key):
            start = _merge_start(prev.start, fact.start)
            if prev.start and fact.start and len(prev.start) != len(fact.start):
                vague = fact.start if start == prev.start else prev.start
                notes.append(f"{fact.p} {fact.o}: kept start {start} over the vaguer {vague}")
            prev.start = start
            prev.end = None if None in (prev.end, fact.end) else max(prev.end or "", fact.end or "")
            prev.text = prev.text or fact.text
            notes.append(f"merged duplicate {fact.p} fact: {fact.text or fact.o}")
        else:
            out[key] = fact
    return list(out.values())


def _one_current(facts: list[_Fact], notes: list[str]) -> list[_Fact]:
    """Single-valued predicates: an undated value next to a dated current one from the same
    document is a vaguer mention of the same state ("Researcher" beside "Research Engineer since
    2024-04", "France" beside "Paris since 2024-04"), not a contradiction for review."""
    dated = {
        (f.subject, f.p)
        for f in facts
        if not is_multi_valued(f.p, f.subject) and f.start and f.end is None
    }
    out = []
    for f in facts:
        undated = f.start is None and f.end is None
        if undated and not is_multi_valued(f.p, f.subject) and (f.subject, f.p) in dated:
            notes.append(f"dropped undated {f.p} {f.o!r}: the document gives a dated current one")
            continue
        out.append(f)
    return out


def _team_as_employer(facts: list[_Fact], name_of: Callable[[str], str], notes: list[str]):
    """``works_at`` naming one of the document's teams ("Data Platform" beside that team)."""
    teams = {fold(f.o) for f in facts if f.p == "team"}
    out = []
    for f in facts:
        if f.p == "works_at" and fold(name_of(f.o)) in teams:
            notes.append(f"dropped works_at {name_of(f.o)!r}: it is a team, not an employer")
            continue
        out.append(f)
    return out


def _order(facts: list[_Fact]) -> list[_Fact]:
    """Single-valued predicates: current facts first, then newest, so that when two versions
    overlap it is the older one the writer queues for review."""

    groups: dict[tuple[str, str], list[int]] = {}
    for i, f in enumerate(facts):
        if not is_multi_valued(f.p, f.subject):
            groups.setdefault((f.subject, f.p), []).append(i)
    out = list(facts)
    for positions in groups.values():
        group = sorted((facts[i] for i in positions), key=lambda f: f.start or "", reverse=True)
        group.sort(key=lambda f: f.end is not None)  # stable: current first, then newest
        for i, f in zip(positions, group, strict=True):
            out[i] = f
    return out


def _past_without_end(
    facts: list[_Fact], name_of: Callable[[str], str], notes: list[str]
) -> list[_Fact]:
    """A past-tense single-valued fact with no end ("Lived in Seville") would read as current
    and contradict today's value. Its end is unknown (guessing one would invent an overlap), so
    it is kept as a ``note``; the prompt asks the model for ``end`` when the text gives it."""
    out = []
    for f in facts:
        if f.p not in PAST_SINGLE or f.end or not PAST.search(f.text):
            out.append(f)
            continue
        text = template_text(f.p, name_of(f.o), past=True)
        notes.append(f"past {f.p} {name_of(f.o)!r} without an end date: kept as a note")
        out.append(_Fact(f.subject, "note", text, text, f.start, None))
    return out


def _regions(facts: Sequence[_Fact], planned: dict[str, _Planned]) -> None:
    for fact in facts:
        obj = fact.target
        if fact.subject == SELF_ID and obj in planned:
            region = OBJECT_REGIONS.get(fact.p)
            if fact.p == "works_at":
                region = "career.history" if fact.end else "career.current_job"
            if region and region not in planned[obj].regions:
                planned[obj].regions.append(region)
        if obj == SELF_ID and fact.subject in planned:
            region = SUBJECT_REGIONS.get(fact.p)
            if region and region not in planned[fact.subject].regions:
                planned[fact.subject].regions.append(region)
    for entity in planned.values():
        if not entity.regions and (default := TYPE_REGIONS.get(entity.op.type)):
            entity.regions.append(default)
        entity.op.regions = entity.regions


# --- the document brief -------------------------------------------------------------------------

PLAN_BRIEF_NAMES = True  # plan the brief's names before facts are resolved
STRUCTURE = {  # (child kind, parent kind) → the structural predicate from the brief's parent
    ("project", "org"): "part_of",
    ("project", "project"): "part_of",
    ("org", "place"): "located_in",
    ("event", "place"): "located_in",
    ("place", "place"): "located_in",
}
ROLE_RELATION = {"friend": "friend_of", "family": "family_of", "colleague": "colleague_of"}


def _brief_entity(res: _Resolver, entry: BriefName) -> str | None:
    """The id of a brief name: an entity of the document, or a page planned from the brief (kept
    only if a fact ends up using it). None for values (teams, tools) and the owner."""
    for name in (entry.name, *entry.aliases):
        if (found := res.lookup(name)) is not None:
            return None if found == SELF_ID else found
    names = (entry.name, *entry.aliases)
    if entry.type not in PAGE_TYPES or any(res.literal(n) or fold(n) in res.teams for n in names):
        return None
    if any(res.is_self(n) for n in (entry.name, *entry.aliases)):
        return None
    return res.plan(entry.name, res.entity_kinds.get(fold(entry.name), entry.type), entry.aliases)


# What the parent of a brief name is when the brief does not list the parent itself: the
# organisation a project or team belongs to, the city an organisation, event or district is in.
PARENT_TYPE = {"project": "org", "team": "org", "org": "place", "event": "place", "place": "place"}


def _plan_brief(res: _Resolver, brief: Brief) -> dict[str, str | None]:
    """Folded brief name (and parent name) → entity id, planning pages for those not in the
    facts yet (dropped again if no fact uses them)."""
    ids = {fold(e.name): _brief_entity(res, e) for e in brief.names}
    for entry in brief.names:
        parent = entry.parent
        if not parent or fold(parent) in ids or entry.type not in PARENT_TYPE:
            continue
        ptype = PARENT_TYPE[entry.type]
        guess = BriefName(name=parent, type=ptype, aliases=[], parent="", role="other")
        ids[fold(parent)] = _brief_entity(res, guess)
    return ids


def _brief_structure(
    facts: list[_Fact], brief: Brief, res: _Resolver, notes: list[str]
) -> list[_Fact]:
    """What the brief knows about how the document's entities relate:

    - a ``located_in``/``part_of`` fact stated the wrong way round (the city in its district)
      is turned around when the brief gives the opposite parent;
    - an entity the facts use whose brief parent is an organisation or place gets ``part_of`` /
      ``located_in`` it (``STRUCTURE``) unless the facts already say what it is part of or in;
    - an organisation the brief gives the role "member" gets ``member_of`` from the owner when
      no fact links the owner to it (dated from a fact naming it, if any);
    - a person's vague ``knows`` takes the relation the brief gives (friend, family, colleague),
      and ``friend_of`` becomes ``family_of`` when the brief says family (a sister read as a
      friend in a part that only says "Linnea").
    """
    index = name_index(brief)
    ids = _plan_brief(res, brief)

    def entry_of(entity_id: str) -> BriefName | None:
        return next((e for e in brief.names if ids.get(fold(e.name)) == entity_id), None)

    def parent_of(entry: BriefName) -> str | None:
        if not entry.parent:
            return None
        parent = index.get(fold(entry.parent))
        return ids.get(fold(parent.name if parent else entry.parent))

    for f in facts:
        # the object may have been kept as a value: a city "located in" an organisation
        target = f.target or res.lookup(f.o) if f.p in ("located_in", "part_of") else None
        entry = entry_of(target) if target and target != SELF_ID else None
        if entry and parent_of(entry) == f.subject:
            notes.append(
                f"brief: turned {f.p} around: {res.title(target)} is in {res.title(f.subject)}"
            )
            f.subject, f.o = target, f"[[{f.subject}]]"

    used = {f.subject for f in facts} | {t for f in facts if (t := f.target)}
    has = {(f.subject, f.p) for f in facts}
    out = list(facts)
    for entry in brief.names:
        child = ids.get(fold(entry.name))
        parent = parent_of(entry) if child in used and entry.role != "side project" else None
        if not child or not parent or parent == child:
            continue
        p = STRUCTURE.get((res.kind(child), res.kind(parent)))
        if p and (child, p) not in has:
            notes.append(f"brief: {res.title(child)} {p} {res.title(parent)}")
            out.append(_Fact(child, p, f"[[{parent}]]", "", None, None))
            has.add((child, p))

    linked = {f.target for f in out if f.subject == SELF_ID and f.target}
    teams = res.teams | {fold(f.o) for f in out if f.p == "team"}
    for entry in brief.names:
        org = ids.get(fold(entry.name))
        if entry.role != "member" or not org or res.kind(org) != "org" or org in linked:
            continue
        names = [entry.name, *entry.aliases]
        if any(fold(n) in teams for n in names):  # the document's own team, not a club
            continue
        starts = sorted(
            f.start
            for f in out
            if f.subject == SELF_ID
            and f.start
            and any(
                re.search(rf"(?<!\w){re.escape(n)}(?!\w)", f"{f.o} {f.text}", re.I) for n in names
            )
        )
        start = starts[0] if starts else None
        notes.append(f"brief: member_of {res.title(org)} (a group the owner belongs to)")
        out.append(_Fact(SELF_ID, "member_of", f"[[{org}]]", "", start, None))
        linked.add(org)

    for f in out:
        entry = entry_of(f.subject) if f.target == SELF_ID else None
        relation = ROLE_RELATION.get(entry.role) if entry else None
        vaguer = ("knows", "friend_of") if relation == "family_of" else ("knows",)
        if relation and f.p in vaguer and f.p != relation:
            notes.append(f"brief: {res.title(f.subject)} {f.p} → {relation}")
            f.p, f.text = relation, ""
    return out


def _brief_stints(brief: Brief, res: _Resolver) -> frozenset[str]:
    """Folded ``[[org id]]`` (or names) of the employers and schools the brief's periods show
    more than once: two separate stints at one organisation."""
    count: dict[str, int] = {}
    for period in brief.periods:
        if period.org.strip():
            target = res.lookup(period.org)
            key = fold(f"[[{target}]]") if target else fold(period.org)
            count[key] = count.get(key, 0) + 1
    return frozenset(k for k, n in count.items() if n > 1)


def _brief_homes(facts: list[_Fact], brief: Brief, res: _Resolver, notes: list[str]) -> list[_Fact]:
    """The brief's homes (cities the owner lived in, in order). A ``lives_in`` fact for a city
    the owner later left (a later home follows it) gets the end the brief gives when it has none:
    a past city would otherwise read as current, or be kept as a note. A home the facts miss is
    added when its period is known (it ended before the next one, or it is the current one); not
    the hometown (``hometown`` says it), and never with an end on the last home."""
    out = list(facts)
    index = name_index(brief)
    places = homes(brief)
    hometowns = {f.target for f in facts if f.subject == SELF_ID and f.p == "hometown" and f.target}
    for n, home in enumerate(places):
        entry = lookup(index, home.city)
        if (entry and entry.role == "hometown") or res.lookup(home.city) in hometowns:
            continue
        later = n < len(places) - 1  # the owner moved on from this city
        city = res.lookup(home.city)
        if city is None and not res.literal(home.city):
            city = res.plan(home.city, "place", entry.aliases if entry else [])
        if city is None or city == SELF_ID or res.kind(city) != "place":
            continue
        lives = [
            f
            for f in out
            if f.subject == SELF_ID
            and f.p == "lives_in"
            and (f.target == city or fold(f.o) == fold(home.city))
        ]
        for f in lives:
            if f.end is None and home.end and later:
                f.end = home.end
                notes.append(f"brief: lives_in {home.city}: end {home.end} (moved on)")
        if lives or not (home.current or (home.end and later)):
            continue
        end = None if home.current else home.end
        notes.append(f"brief: lives_in {home.city} ({home.start or '?'}–{end or 'now'})")
        out.append(_Fact(SELF_ID, "lives_in", f"[[{city}]]", "", home.start, end))
    return out


def _subject_ok(p: str, kind: str) -> bool:
    if p in RELATIONSHIP_PREDICATES:
        return kind == "person"
    if p in PROJECT_PREDICATES:
        return kind == "project"
    if p == "goal":
        return kind in ("self", "project")
    if p in ("located_in", "part_of"):
        return kind in ("org", "place", "event", "project")
    if p == "member_of":
        return kind in ("self", "person", "org")
    return kind in ("self", "person")


def _convert_fact(raw: ExtractedFact, res: _Resolver, notes: list[str]) -> list[_Fact]:
    p, text, value = raw.predicate, clean(raw.text), clean(raw.object)
    about = text or f"{p} {value}"
    subject = res.lookup(raw.subject)
    if subject is None:
        if why := res.literal(raw.subject):
            notes.append(f"dropped {p} fact about {clean(raw.subject)!r} ({why}): {about}")
        else:
            notes.append(f"dropped fact with unknown subject {raw.subject!r}: {about}")
        return []
    if not value:
        notes.append(f"dropped {p} fact without an object: {about}")
        return []
    if ZERO_COUNT.search(value) or ZERO_COUNT.search(text):
        notes.append(f"dropped {p} fact with a zero count (an unread page counter?): {about}")
        return []
    if p == "has_skill" and fold(value) in LANGUAGES:
        p = "speaks"
        notes.append(f"has_skill → speaks ({value} is a language)")
    if p == "uses" and res.kind(subject) in ("self", "person"):
        p = "has_skill"
        notes.append(f"uses → has_skill (the subject is not a project): {value}")
    if p == "lives_in" and re.match(r"grew up\b", text, re.I):
        p = "hometown"
        notes.append(f"lives_in → hometown (grew up there): {value}")
    if p in ("works_at", "member_of") and (why := res.literal(value)) == "a team":
        notes.append(f"{p} → team ({value!r} is {why}): {about}")
        p = "team"

    target = res.lookup(value)
    if p in LITERAL_OBJECTS:
        if target == subject and p != "name":
            notes.append(f"dropped {p} fact whose subject is its object: {about}")
            return []
        target = target if target in res.pages else None
    else:
        if target == SELF_ID and p not in TOWARDS_OWNER | {"manager", "mentor_of"}:
            target = None  # e.g. the owner's name is a value, not a link to the owner
        if (
            p == "works_at"
            and (target is None or res.kind(target) == "org")
            and is_duty(text, value)
        ):
            notes.append(f"works_at → responsibility (describes the work): {about}")
            p, target, value = "responsibility", None, _sentence(text or value)
        elif target is None and (why := res.literal(value)) and p in OBJECT_TYPES:
            notes.append(f"dropped {p} fact: {value!r} is {why}, not a page: {about}")
            return []
        elif target is None and p in ENTITY_OBJECTS and subject == SELF_ID:
            target = res.plan(value, res.entity_kinds.get(fold(value), ENTITY_OBJECTS[p]), [])
    if target == subject:
        notes.append(f"dropped {p} fact whose subject is its object: {about}")
        return []
    if subject == SELF_ID and target and p in TOWARDS_OWNER:
        subject, target = target, SELF_ID  # (self, owner, project) → (project, owner, self)

    skind = res.kind(subject)
    if not _subject_ok(p, skind):
        notes.append(f"dropped {p} fact: its subject {res.title(subject)!r} is a {skind}: {about}")
        return []
    allowed = OBJECT_TYPES.get(p)
    if target and allowed and (okind := res.kind(target)) not in allowed:
        name = res.title(target) if target != SELF_ID else value
        if p == "works_at" and okind == "project":
            notes.append(f"works_at → worked_on ({name!r} is a project): {about}")
            p = "worked_on"
        elif p in RELATIONSHIP_PREDICATES or p in ("works_at", "studied_at", "owner", "manager"):
            notes.append(f"dropped {p} fact: {name!r} is a {okind}: {about}")
            return []
        else:
            notes.append(f"{p}: {name!r} is a {okind}, kept as a literal value: {about}")
            target, value = None, name
    elif not target and p in RELATIONSHIP_PREDICATES:
        notes.append(f"dropped {p} fact: it must point at the owner: {about}")
        return []

    o = f"[[{target}]]" if target else value
    start, end = normalise_date(raw.start), normalise_date(raw.end)
    for label, given, got in (("start", raw.start, start), ("end", raw.end, end)):
        if given.strip() and got is None and fold(given) not in OPEN_ENDED:
            notes.append(f"dropped unreadable {label} date {given!r}: {about}")
    if start and end and end < start:
        notes.append(f"dropped end {end} before start {start}: {about}")
        end = None
    return [_Fact(subject, p, o, text, start, end)]


def build_changeset(
    extractions: Sequence[Extraction],
    vault: Vault,
    *,
    src: str,
    kind: Kind = "stated",
    rationale: str = "",
    today: date | None = None,
    entity_kinds: Mapping[str, str] | None = None,
    brief: Brief | None = None,
) -> Normalised:
    """``today`` is the ingest date: the model sometimes dates facts to it ("Today is …" is in
    its prompt); a document can't, so such starts are dropped.

    ``entity_kinds``: folded name → entity kind decided by a typed check
    (``brain.ingest.decisions``): a page kind (person, org, project, place, event) replaces the
    extractor's type; any other kind (team, tool, topic, activity) makes the name a literal
    value. (Not the facts' ``kind``, stated/observed/inferred.)

    ``brief``: the document brief (``brain.ingest.brief``): its names merge variant names into one
    entity, its parents and roles add structural facts, its homes date past cities."""
    entity_kinds = entity_kinds or {}
    notes: list[str] = []
    profile = vault.page(SELF_PAGE)
    profile_names = [profile.title, *profile.aliases] if profile else []
    profile_names = [n for n in profile_names if n and n != "Profile"]
    doc_owner: list[str] = []
    for extraction in extractions:
        for name in extraction.owner_names:
            if (n := clean(name)) and fold(n) not in {fold(x) for x in doc_owner}:
                doc_owner.append(n)
    owner_all = [*profile_names, *doc_owner]
    if brief and brief.subject.strip() and fold(brief.subject) not in map(fold, owner_all):
        owner_all.append(clean(brief.subject))
    res = _Resolver(vault, owner_all)
    index = res.brief_names = name_index(brief)
    res.teams = {fold(f.object) for x in extractions for f in x.facts if f.predicate == "team"}
    for name, decided in entity_kinds.items():
        if decided in TYPE_DIRS and decided != "topic":
            res.entity_kinds[name] = decided
        else:  # a team, a tool, a topic, an activity: a value
            article = "an" if decided[:1] in "aeiou" else "a"
            res.literals.setdefault(name, f"{article} {decided}")

    for extraction in extractions:
        for entity in extraction.entities:
            name = clean(entity.name)
            if not name or res.is_self(name):
                continue
            if entity.id.strip() in res.pages:
                res.doc_names.setdefault(fold(name), entity.id.strip())
                for alias in entity.aliases:
                    res.doc_names.setdefault(fold(alias), entity.id.strip())
                continue
            etype = res.entity_kinds.get(fold(name), entity.type)
            why = res.literals.get(fold(name))
            why = why or ("a tool" if etype in LITERAL_TYPES else literal_reason(name))
            if why:
                for alias in (name, *entity.aliases):
                    res.literals.setdefault(fold(alias), why)
                continue
            aliases = list(entity.aliases)
            entry = next((e for n in (name, *aliases) if (e := lookup(index, n))), None)
            variant = entry and entry.type == etype and fold(entry.name) != fold(name)
            if variant and not res.is_self(entry.name):
                # a variant of a name in the brief ("Atlas" for "Atlas Migration"): one entity
                notes.append(f"brief: {name!r} is {entry.name!r}")
                aliases = [name, *aliases]
                name = entry.name
            if entry:
                aliases += [a for a in entry.aliases if fold(a) not in map(fold, aliases)]
            res.plan(name, etype, aliases)
    planned = set(res.planned)
    if brief is not None and PLAN_BRIEF_NAMES:
        # the brief's other names, so that facts naming them link to a page (a city given as a
        # plain value, a person no part declared); kept only if a fact uses them
        _plan_brief(res, brief)
    from_brief = set(res.planned) - planned

    facts: list[_Fact] = []
    cutoff = (today or date.today()).isoformat()
    for extraction in extractions:
        for raw in extraction.facts:
            for fact in _convert_fact(raw, res, notes):
                if fact.start and len(fact.start) == len(cutoff) and fact.start >= cutoff:
                    notes.append(
                        f"dropped start {fact.start} (the ingest day): {fact.text or fact.o}"
                    )
                    fact.start = None
                facts.append(fact)

    owner_display = owner_all[0] if owner_all else "the owner"

    def name_of(o: str) -> str:
        target = o[2:-2] if o.startswith("[[") else None
        if target is None:
            return o
        return owner_display if target == SELF_ID else res.title(target)

    if brief is not None:
        facts = _brief_homes(_brief_structure(facts, brief, res, notes), brief, res, notes)
    stints = _brief_stints(brief, res) if brief is not None else frozenset()
    facts = _past_without_end(_merge(facts, notes, stints), name_of, notes)
    facts = _order(_one_current(_team_as_employer(facts, name_of, notes), notes))

    used = {f.subject for f in facts} | {t for f in facts if (t := f.target)}
    for entity_id in [i for i in res.planned if i not in used]:
        if entity_id not in from_brief:
            notes.append(f"dropped entity {entity_id}: no fact refers to it")
        del res.planned[entity_id]
    _regions(facts, res.planned)

    ops: list[Operation] = [p.op for p in res.planned.values()]
    if profile:
        known = {fold(n) for n in (profile.title, *profile.aliases)}
        for name in doc_owner:
            if fold(name) not in known and not any(
                i != SELF_ID for i in res.aliases.get(fold(name), [])
            ):
                ops.append(AddAlias(id=SELF_PAGE, alias=name))
                known.add(fold(name))
    for f in facts:
        o_name = name_of(f.o)
        text = fact_text(f.text, "") if f.text else ""
        problem = text_problem(
            text,
            o_name="" if f.target == SELF_ID else o_name,
            owner_names=owner_all,
            about_self=f.subject == SELF_ID and f.p != "name",
        )
        if problem:
            if problem != "empty":
                notes.append(f"rewrote {f.p} text ({problem}): {text}")
            text = template_text(f.p, o_name, past=f.end is not None)
        spec = {"from": f.start, "to": f.end}
        ops.append(AddFact(subject=f.subject, p=f.p, o=f.o, text=text, kind=kind, **spec))
    changeset = ChangeSet(ops=ops, origin="ingest", src=[src], rationale=rationale)
    return Normalised(changeset, notes, doc_owner)
