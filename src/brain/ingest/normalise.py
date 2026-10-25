"""Turn per-chunk extractions into one validated ``ChangeSet`` for a document.

Everything the model returns is treated as a proposal: names are resolved to entity ids in code
(known aliases first, then entities named earlier in the same document, then new ids), dates are
normalised or dropped, duplicates are merged, and entities no kept fact refers to are dropped.
What the policy queues for review is decided later by the writer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date

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
SELF_WORDS = {"self", "owner", "the owner", "me", "i", "myself"}
# A literal object of these predicates names an entity of this type: create its page.
ENTITY_OBJECTS = {"works_at": "org", "studied_at": "org"}
# Objects that are values, not things: they link only to pages that already exist in the vault,
# never to entities the model proposed in this document (a tool is not a project page).
_LITERAL = "name born nationality speaks role team responsibility degree certification has_skill"
LITERAL_OBJECTS = frozenset(f"{_LITERAL} values interest likes goal status deadline uses".split())
# Predicates whose subject is the other party when the owner is involved: a relation points
# from the person to the owner, ownership from the project to the owner.
TOWARDS_OWNER = RELATIONSHIP_PREDICATES | {"owner", "stakeholder"}
LEADING_SUBJECT = re.compile(r"^(?:he|she|they|it|i)\s+(?=\w)", re.I)
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
MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for m in names
}


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


@dataclass
class _Planned:
    op: CreateEntity
    regions: list[str] = field(default_factory=list)


@dataclass
class _Fact:
    subject: str
    p: str
    o: str  # "[[id]]" or a literal
    text: str
    start: str | None
    end: str | None


@dataclass
class Normalised:
    changeset: ChangeSet
    notes: list[str]  # what was dropped or changed, and why
    owner_names: list[str]


class _Resolver:
    def __init__(self, vault: Vault, owner_names: Iterable[str]) -> None:
        self.pages = {p.id: p for p in vault.pages()}
        self.aliases = {k: v for k, v in vault.alias_index().items()}
        self.owner = {fold(n) for n in owner_names if n.strip()} | SELF_WORDS
        self.doc_names: dict[str, str] = {}  # names seen in this document → entity id
        self.planned: dict[str, _Planned] = {}  # new entity id → create op

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
        return None

    def plan(self, name: str, etype: str, aliases: Sequence[str], summary: str) -> str:
        """Id for an entity named in the document, planning a new page if none matches."""
        title = clean(name)
        existing = self.lookup(title)
        if existing is None:
            existing = next((i for a in aliases if (i := self.lookup(a))), None)
        entity_id = existing or f"{TYPE_DIRS[etype]}/{slugify(title)}"
        if entity_id not in self.pages and entity_id != SELF_ID:
            planned = self.planned.get(entity_id)
            if planned is None:
                op = CreateEntity(id=entity_id, type=etype, title=title, summary=clean(summary))
                planned = self.planned[entity_id] = _Planned(op)
            elif not planned.op.summary and summary.strip():
                planned.op.summary = clean(summary)
            names = {fold(n) for n in (planned.op.title, *planned.op.aliases)}
            for alias in (title, *aliases):
                if (a := clean(alias)) and fold(a) not in names and not self.is_self(a):
                    planned.op.aliases.append(a)
                    names.add(fold(a))
        for alias in (title, *aliases):
            if fold(alias) and not self.is_self(alias):
                self.doc_names.setdefault(fold(alias), entity_id)
        return entity_id


def _merge(facts: list[_Fact], notes: list[str]) -> list[_Fact]:
    """One fact per (subject, p, o), spanning the union of the duplicates' periods: the
    earliest start, and no end if any of them is still true."""
    out: dict[tuple[str, str, str], _Fact] = {}
    for fact in facts:
        key = (fact.subject, fact.p, fold(fact.o))
        if prev := out.get(key):
            starts = [d for d in (prev.start, fact.start) if d]
            prev.start = min(starts) if starts else None
            prev.end = None if None in (prev.end, fact.end) else max(prev.end or "", fact.end or "")
            notes.append(f"merged duplicate {fact.p} fact: {fact.text}")
        else:
            out[key] = fact
    return list(out.values())


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


def _regions(facts: Sequence[_Fact], planned: dict[str, _Planned]) -> None:
    for fact in facts:
        obj = fact.o[2:-2] if fact.o.startswith("[[") else None
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


def fact_text(text: str, fallback: str) -> str:
    """Drop a leading pronoun ("He speaks Spanish" → "Speaks Spanish")."""
    text = LEADING_SUBJECT.sub("", clean(text)) or fallback
    return text[:1].upper() + text[1:]


def _object(raw: ExtractedFact, subject: str, value: str, res: _Resolver) -> str | None:
    """Entity id for the object, or None for a literal value."""
    if raw.predicate in LITERAL_OBJECTS:
        target = res.lookup(value)
        return target if target in res.pages else None
    target = res.lookup(value)
    if target == SELF_ID and raw.predicate not in TOWARDS_OWNER | {"manager", "mentor_of"}:
        return None  # e.g. the owner's name is a value, not a link to the owner
    if target is None and raw.predicate in ENTITY_OBJECTS and subject == SELF_ID:
        target = res.plan(value, ENTITY_OBJECTS[raw.predicate], [], "")
    return target


def _convert_fact(raw: ExtractedFact, res: _Resolver, notes: list[str]) -> _Fact | None:
    subject = res.lookup(raw.subject)
    if subject is None:
        notes.append(f"dropped fact with unknown subject {raw.subject!r}: {raw.text}")
        return None
    value = clean(raw.object)
    if not value:
        notes.append(f"dropped {raw.predicate} fact without an object: {raw.text}")
        return None
    target = _object(raw, subject, value, res)
    if target == subject:
        notes.append(f"dropped {raw.predicate} fact whose subject is its object: {raw.text}")
        return None
    if subject == SELF_ID and target and raw.predicate in TOWARDS_OWNER:
        subject, target = target, SELF_ID  # (self, owner, project) → (project, owner, self)
    o = f"[[{target}]]" if target else value
    start, end = normalise_date(raw.start), normalise_date(raw.end)
    for label, given, got in (("start", raw.start, start), ("end", raw.end, end)):
        if given.strip() and got is None and fold(given) not in OPEN_ENDED:
            notes.append(f"dropped unreadable {label} date {given!r}: {raw.text}")
    if start and end and end < start:
        notes.append(f"dropped end {end} before start {start}: {raw.text}")
        end = None
    text = fact_text(raw.text, f"{raw.predicate.replace('_', ' ')} {value}")
    return _Fact(subject, raw.predicate, o, text, start, end)


def build_changeset(
    extractions: Sequence[Extraction],
    vault: Vault,
    *,
    src: str,
    kind: Kind = "stated",
    rationale: str = "",
) -> Normalised:
    notes: list[str] = []
    profile = vault.page(SELF_PAGE)
    profile_names = [profile.title, *profile.aliases] if profile else []
    profile_names = [n for n in profile_names if n and n != "Profile"]
    doc_owner: list[str] = []
    for extraction in extractions:
        for name in extraction.owner_names:
            if (n := clean(name)) and fold(n) not in {fold(x) for x in doc_owner}:
                doc_owner.append(n)
    res = _Resolver(vault, [*profile_names, *doc_owner])

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
            res.plan(name, entity.type, entity.aliases, entity.summary)

    facts: list[_Fact] = []
    for extraction in extractions:
        for raw in extraction.facts:
            if fact := _convert_fact(raw, res, notes):
                facts.append(fact)
    facts = _order(_merge(facts, notes))

    used = {f.subject for f in facts} | {f.o[2:-2] for f in facts if f.o.startswith("[[")}
    for entity_id in [i for i in res.planned if i not in used]:
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
        spec = {"from": f.start, "to": f.end}
        ops.append(AddFact(subject=f.subject, p=f.p, o=f.o, text=f.text, kind=kind, **spec))
    changeset = ChangeSet(ops=ops, origin="ingest", src=[src], rationale=rationale)
    return Normalised(changeset, notes, doc_owner)
