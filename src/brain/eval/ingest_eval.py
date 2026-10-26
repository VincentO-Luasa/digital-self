"""``brain eval ingest``: ingest each evaluation document into a fresh vault and score what
landed in it against hand-written gold facts.

The set lives in ``tests/eval/ingest/`` (fictional persona): ``docs/`` holds documents of
different types, ``gold/<doc stem>.yaml`` the facts a careful human would extract from each
(format in ``tests/eval/README.md``). Matching is lenient on wording and strict on structure:

- subject and predicate must agree (the gold may list several acceptable predicates);
- objects match on tokens after folding case, accents and punctuation, dropping stop words and
  light suffixes: at least 80% of the gold's tokens must be in the extracted object, and (except
  for free-text predicates such as ``responsibility``) at least half of the extracted object's
  tokens must be the gold's; entity objects are compared through the page's title and aliases;
- dates are compared at the gold's precision (gold ``2021`` accepts ``2021-03``; gold
  ``2021-03`` rejects ``2021``); ``current: true`` means the fact must have no end.

Each extracted fact matches at most one gold fact (greedy, best score first). Extracted facts
that match nothing are "unverified": not necessarily wrong, listed for a human to read.
"""

from __future__ import annotations

import os
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml

from brain.config import REPO_DIR
from brain.vault.models import PREDICATES, SELF_ID, is_multi_valued, link_target
from brain.vault.ops import AddFact, CreateEntity
from brain.vault.store import Vault

EVAL_DIR = REPO_DIR / "tests" / "eval" / "ingest"
# Fixed "today" for every run: the extraction prompt carries the date, so a moving date would
# miss the cache every day, and the documents are written relative to this one.
EVAL_TODAY = date(2026, 10, 25)

Level = Literal["required", "optional", "acceptable"]
LEVELS: tuple[Level, ...] = ("required", "optional", "acceptable")
ENTITY_TYPES = frozenset({"person", "org", "project", "place", "topic", "event"})
FREE_TEXT = frozenset({"responsibility", "goal", "note", "habit", "interest", "likes", "values"})
SELF_NAMES = frozenset({"self", "self/profile"})
STOPWORDS = frozenset(
    {"a", "an", "the", "of", "in", "at", "for", "and", "to", "with", "on", "as", "by", "from"}
    | {"my", "our", "is"}
)
TEAM = frozenset({"team"})  # the vault renders a team "Data Platform" as "Data Platform team"
SUFFIXES = ("ings", "ing", "ers", "er", "es", "s", "ed")
GOLD_COVER = 0.8  # share of the gold's tokens the extracted value must contain
GOT_COVER = 0.5  # share of the extracted value's tokens that must be the gold's
FOLD = str.maketrans({"ø": "o", "æ": "ae", "ß": "ss", "đ": "d", "ł": "l", "œ": "oe"})


# --- text matching -----------------------------------------------------------------------------


def _stem(word: str) -> str:
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def tokens(text: str) -> list[str]:
    """Folded content words: case, accents, punctuation, stop words and light suffixes go."""
    text = unicodedata.normalize("NFKD", text.casefold().translate(FOLD))
    text = "".join(c for c in text if not unicodedata.combining(c))
    words = re.findall(r"[a-z0-9]+", text)
    return [_stem(w) for w in words if w not in STOPWORDS and (len(w) > 1 or w.isdigit())]


def similarity(
    gold: str, got: str, *, free_text: bool = False, drop: frozenset[str] = frozenset()
) -> float:
    """Dice score of the two token sets (without ``drop``), or 0 when below the coverage
    thresholds."""
    g, e = set(tokens(gold)) - drop, set(tokens(got)) - drop
    if not g or not e:
        return 1.0 if gold.strip().casefold() == got.strip().casefold() and gold.strip() else 0.0
    shared = len(g & e)
    if shared / len(g) < GOLD_COVER or (not free_text and shared / len(e) < GOT_COVER):
        return 0.0
    return 2 * shared / (len(g) + len(e))


def best_similarity(
    golds: Iterable[str],
    gots: Iterable[str],
    *,
    free_text: bool = False,
    drop: frozenset[str] = frozenset(),
) -> float:
    gots = list(gots)
    pairs = ((g, e) for g in golds for e in gots)
    return max((similarity(g, e, free_text=free_text, drop=drop) for g, e in pairs), default=0.0)


def names_literal(literal: str, names: Iterable[str]) -> bool:
    """A page named like a literal: "Data Platform team" is also the page "Data Platform"."""
    bare = set(tokens(literal)) - TEAM
    return bool(best_similarity([literal], names)) or any(
        bare and set(tokens(n)) - TEAM == bare for n in names
    )


# --- gold --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Form:
    """One way of stating a gold fact: subject, acceptable predicates, acceptable objects."""

    s: str
    p: tuple[str, ...]
    o: tuple[str, ...]


@dataclass(frozen=True)
class GoldFact:
    form: Form
    level: Level
    start: str | None = None
    end: str | None = None
    current: bool = False
    alts: tuple[Form, ...] = ()  # other acceptable forms (e.g. the relation the other way)
    why: str = ""
    at: tuple[str, ...] = ()  # a degree's school (the fact's ``at::`` link)

    @property
    def forms(self) -> tuple[Form, ...]:
        return (self.form, *self.alts)

    @property
    def checks_dates(self) -> bool:
        return bool(self.start or self.end or self.current)

    def show(self) -> str:
        f = self.form
        when = _when(self.start, self.end, "now" if self.current else "")
        return f"{f.s} · {'|'.join(f.p)} · {' | '.join(f.o)}{when}"


@dataclass(frozen=True)
class GoldEntity:
    name: str
    type: str
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


@dataclass
class GoldDoc:
    name: str  # the gold file's stem
    path: Path  # the document
    kind: Literal["stated", "observed"]
    owner: tuple[str, ...]
    entities: list[GoldEntity]
    literals: list[str]
    facts: list[GoldFact]

    def expected(self, entity: GoldEntity) -> bool:
        """Named by a required or optional fact, so it should have a page (an entity named only
        in ``acceptable`` facts may or may not get one)."""
        names = {n.casefold() for n in entity.names}
        return any(
            v.casefold() in names
            for f in self.facts
            if f.level != "acceptable"
            for form in f.forms
            for v in (form.s, *form.o)
        )

    def level(self, level: Level) -> list[GoldFact]:
        return [f for f in self.facts if f.level == level]

    def entity_names(self, name: str) -> tuple[str, ...]:
        """``name`` plus the aliases of the gold entity it names, if any."""
        for entity in self.entities:
            if any(n.casefold() == name.casefold() for n in entity.names):
                return entity.names
        return (name,)


def _date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return value.isoformat() if isinstance(value, date) else str(value)


def _tuple(value: Any) -> tuple[str, ...]:
    values = value if isinstance(value, list) else [value]
    return tuple(str(v) for v in values if v is not None and str(v).strip())


def _form(raw: dict[str, Any], base: Form | None = None) -> Form:
    s = str(raw["s"]) if "s" in raw else (base.s if base else "")
    p = _tuple(raw["p"]) if "p" in raw else (base.p if base else ())
    o = _tuple(raw["o"]) if "o" in raw else (base.o if base else ())
    return Form(s, p, o)


def parse_fact(raw: dict[str, Any], level: Level) -> GoldFact:
    form = _form(raw)
    alts = tuple(_form(a, form) for a in raw.get("alt") or [])
    return GoldFact(
        form=form,
        level=level,
        start=_date(raw.get("from")),
        end=_date(raw.get("to")),
        current=bool(raw.get("current", False)),
        alts=alts,
        why=str(raw.get("why", "")),
        at=_tuple(raw.get("at") or []),
    )


def load_gold(path: Path, docs_dir: Path | None = None) -> GoldDoc:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    docs_dir = docs_dir or path.parent.parent / "docs"
    facts = [parse_fact(f, level) for level in LEVELS for f in data.get(level) or []]
    entities = [
        GoldEntity(str(e["name"]), str(e["type"]), _tuple(e.get("aliases") or []))
        for e in data.get("entities") or []
    ]
    return GoldDoc(
        name=path.stem,
        path=docs_dir / str(data["doc"]),
        kind=data.get("kind", "stated"),
        owner=_tuple(data.get("owner") or []),
        entities=entities,
        literals=[str(x) for x in data.get("literals") or []],
        facts=facts,
    )


def load_gold_set(eval_dir: Path = EVAL_DIR, only: Sequence[str] = ()) -> list[GoldDoc]:
    """Every gold file, or those whose name or document name contains one of ``only``."""
    golds = [load_gold(p, eval_dir / "docs") for p in sorted((eval_dir / "gold").glob("*.yaml"))]
    if only:
        wanted = [w.casefold() for w in only]
        golds = [
            g for g in golds if any(w in g.name.casefold() or w in g.path.name for w in wanted)
        ]
    return golds


def gold_problems(gold: GoldDoc) -> list[str]:
    """What is wrong with a gold file ([] if nothing)."""
    problems = []
    if not gold.path.is_file():
        problems.append(f"document missing: {gold.path.name}")
    if gold.kind not in ("stated", "observed"):
        problems.append(f"kind must be stated or observed: {gold.kind}")
    for entity in gold.entities:
        if entity.type not in ENTITY_TYPES:
            problems.append(f"entity {entity.name}: bad type {entity.type}")
    known = {n.casefold() for e in gold.entities for n in e.names}
    for fact in gold.facts:
        for form in fact.forms:
            if not form.s or not form.p or not form.o:
                problems.append(f"incomplete fact: {fact.show()}")
            for p in form.p:
                if p not in PREDICATES:
                    problems.append(f"unknown predicate {p!r}: {fact.show()}")
            if form.s != SELF_ID and form.s.casefold() not in known:
                problems.append(f"subject {form.s!r} is not a declared entity: {fact.show()}")
        for school in fact.at:
            if school.casefold() not in known:
                problems.append(f"at {school!r} is not a declared entity: {fact.show()}")
        for value in (fact.start, fact.end):
            if value and not re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", value):
                problems.append(f"bad date {value!r}: {fact.show()}")
    if not gold.level("required"):
        problems.append("no required facts")
    return problems


# --- what the ingest produced --------------------------------------------------------------------


@dataclass(frozen=True)
class GotEntity:
    id: str
    type: str
    names: tuple[str, ...]
    queued: bool = False


@dataclass(frozen=True)
class GotFact:
    subject: str  # entity id or "self"
    p: str
    o: str  # "[[id]]" or a literal
    start: str | None = None
    end: str | None = None
    text: str = ""
    history: bool = False
    queued: bool = False  # waiting in a review, not in the wiki yet
    at: str | None = None  # a degree's ``at::`` qualifier (the school)

    @property
    def target(self) -> str | None:
        return link_target(self.o)


@dataclass
class Extracted:
    entities: list[GotEntity] = field(default_factory=list)
    facts: list[GotFact] = field(default_factory=list)

    def names(self, entity_id: str) -> tuple[str, ...]:
        for entity in self.entities:
            if entity.id == entity_id:
                return entity.names
        return (entity_id.rsplit("/", 1)[-1].replace("-", " "),)

    def object_names(self, fact: GotFact) -> tuple[str, ...]:
        target = fact.target
        return self.names(target) if target else (fact.o,)

    def show(self, fact: GotFact) -> str:
        subject = SELF_ID if fact.subject == SELF_ID else self.names(fact.subject)[0]
        target = fact.target
        obj = SELF_ID if target in SELF_NAMES else self.object_names(fact)[0]
        flags = (" (queued)" if fact.queued else "") + (" (history)" if fact.history else "")
        return f"{subject} · {fact.p} · {obj}{_when(fact.start, fact.end)}{flags}"


def _when(start: str | None, end: str | None, open_end: str = "") -> str:
    if not start and not end:
        return f" [{open_end}]" if open_end else ""
    return f" [{start or '?'}–{end or open_end}]"


def collect(vault: Vault, queued: Iterable[tuple[Any, str]] = ()) -> Extracted:
    """Pages and facts (current and history) in the vault, plus what was queued for review."""
    out = Extracted()
    for page in vault.pages():
        if page.subject != SELF_ID:
            out.entities.append(GotEntity(page.subject, page.type, (page.title, *page.aliases)))
        for facts, history in ((page.facts, False), (page.history, True)):
            for f in facts:
                out.facts.append(
                    GotFact(
                        page.subject,
                        f.p,
                        f.o,
                        f.valid_from,
                        f.valid_to,
                        f.text,
                        history,
                        at=f.extra.get("at"),
                    )
                )
    for op, _ in queued:
        if isinstance(op, CreateEntity):
            out.entities.append(GotEntity(op.id, op.type, (op.title, *op.aliases), queued=True))
        elif isinstance(op, AddFact):
            out.facts.append(
                GotFact(
                    op.subject,
                    op.p,
                    op.o,
                    op.valid_from,
                    op.valid_to,
                    op.text,
                    queued=True,
                    at=op.extra.get("at"),
                )
            )
    return out


# --- scoring -----------------------------------------------------------------------------------


@dataclass
class Match:
    gold: GoldFact
    got: GotFact
    score: float
    date_problems: list[str]
    link_problem: str = ""  # the gold names the school (``at``) and the fact links another


@dataclass
class Miss:
    gold: GoldFact
    kind: str  # "not extracted", "wrong predicate", ...
    detail: str = ""


@dataclass
class EntityResult:
    gold: GoldEntity
    found: GotEntity | None
    expected: bool = True

    @property
    def type_ok(self) -> bool:
        return self.found is not None and self.found.type == self.gold.type


@dataclass
class DocScore:
    gold: GoldDoc
    seconds: float = 0.0
    chunks: int = 0
    error: str = ""
    extracted: Extracted = field(default_factory=Extracted)
    matches: list[Match] = field(default_factory=list)
    misses: list[Miss] = field(default_factory=list)
    unverified: list[GotFact] = field(default_factory=list)
    entities: list[EntityResult] = field(default_factory=list)
    literal_pages: list[tuple[str, str]] = field(default_factory=list)  # (literal, page id)
    unexpected_pages: list[GotEntity] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    failed_chunks: list[str] = field(default_factory=list)

    def _matched(self, level: Level) -> int:
        return sum(m.gold.level == level for m in self.matches)

    def counts(self) -> dict[str, int]:
        dated = [m for m in self.matches if m.gold.checks_dates and m.gold.level != "acceptable"]
        found = [e for e in self.entities if e.found]
        expected = [e for e in self.entities if e.expected]
        return {
            "required": len(self.gold.level("required")),
            "required_matched": self._matched("required"),
            "optional": len(self.gold.level("optional")),
            "optional_matched": self._matched("optional"),
            "extracted": len(self.extracted.facts),
            "extracted_matched": len(self.matches),
            "unverified": len(self.unverified),
            "dates_checked": len(dated),
            "dates_ok": sum(not m.date_problems for m in dated),
            "links_checked": len(linked := [m for m in self.matches if m.gold.at]),
            "links_ok": sum(not m.link_problem for m in linked),
            "entities": len(expected),
            "entities_found": sum(e.found is not None for e in expected),
            "entities_typed": len(found),
            "entity_types_ok": sum(e.type_ok for e in found),
            "literal_pages": len(self.literal_pages),
            "queued": sum(f.queued for f in self.extracted.facts),
        }


def _ratio(n: int, d: int) -> float | None:
    return round(n / d, 3) if d else None


def metrics(c: dict[str, int], seconds: float) -> dict[str, float | int | None]:
    return {
        "required_recall": _ratio(c["required_matched"], c["required"]),
        "optional_recall": _ratio(c["optional_matched"], c["optional"]),
        "precision": _ratio(c["extracted_matched"], c["extracted"]),
        "date_accuracy": _ratio(c["dates_ok"], c["dates_checked"]),
        "link_accuracy": _ratio(c.get("links_ok", 0), c.get("links_checked", 0)),
        "entity_recall": _ratio(c["entities_found"], c["entities"]),
        "entity_type_accuracy": _ratio(c["entity_types_ok"], c["entities_typed"]),
        "literal_pages": c["literal_pages"],
        "queued": c["queued"],
        "unverified": c["unverified"],
        "seconds": round(seconds, 1),
    }


def date_problems(gold: GoldFact, got: GotFact) -> list[str]:
    """How the extracted dates disagree with the gold's, at the gold's precision."""
    problems = []
    for name, want, have in (("start", gold.start, got.start), ("end", gold.end, got.end)):
        if not want:
            continue
        if not have:
            problems.append(f"missing {name} (expected {want})")
        elif len(have) < len(want) and want.startswith(have):
            problems.append(f"less precise {name} (expected {want}, got {have})")
        elif have[: len(want)] != want:
            problems.append(f"wrong {name} (expected {want}, got {have})")
    if gold.current and got.end:
        problems.append(f"ended but current (got end {got.end})")
    return problems


def link_problem(gold: GoldFact, got: GotFact, extracted: Extracted, doc: GoldDoc) -> str:
    """How the fact's ``at::`` school disagrees with the gold's ("" if it agrees or the gold
    names none)."""
    if not gold.at:
        return ""
    if not got.at:
        return f"no school (expected {gold.at[0]})"
    target = link_target(got.at)
    names = extracted.names(target) if target else (got.at,)
    if best_similarity([n for school in gold.at for n in doc.entity_names(school)], names):
        return ""
    return f"wrong school (expected {gold.at[0]}, got {names[0]})"


class _Matcher:
    def __init__(self, gold: GoldDoc, extracted: Extracted) -> None:
        self.gold = gold
        self.x = extracted

    def subject_score(self, form: Form, got: GotFact) -> float:
        if form.s == SELF_ID:
            return 1.0 if got.subject == SELF_ID else 0.0
        if got.subject == SELF_ID:
            return 0.0
        return best_similarity(self.gold.entity_names(form.s), self.x.names(got.subject))

    def object_score(self, form: Form, got: GotFact) -> float:
        target = got.target
        if any(o == SELF_ID for o in form.o):
            return 1.0 if target in SELF_NAMES else 0.0
        if target in SELF_NAMES:
            return 0.0
        golds = [n for o in form.o for n in self.gold.entity_names(o)]
        free, drop = got.p in FREE_TEXT, TEAM if got.p == "team" else frozenset()
        return best_similarity(golds, self.x.object_names(got), free_text=free, drop=drop)

    def score(self, gold: GoldFact, got: GotFact) -> float:
        best = 0.0
        for form in gold.forms:
            if got.p not in form.p:
                continue
            s = self.subject_score(form, got)
            o = self.object_score(form, got) if s else 0.0
            if s and o:
                dates = 0.5 if gold.checks_dates and not date_problems(gold, got) else 0.0
                best = max(best, s + o + dates)
        return best

    def diagnose(self, gold: GoldFact) -> Miss:
        """Why a gold fact found no match: the nearest extracted fact, if any."""
        for form in gold.forms:
            subject = [f for f in self.x.facts if self.subject_score(form, f)]
            both = [f for f in subject if self.object_score(form, f)]
            if wrong_p := [f for f in both if f.p not in form.p]:
                return Miss(gold, "wrong predicate", self.x.show(wrong_p[0]))
            if same := [f for f in both if f.p in form.p]:  # taken by another gold fact
                return Miss(gold, "merged into another fact", self.x.show(same[0]))
            same_p = [f for f in self.x.facts if f.p in form.p and self.object_score(form, f)]
            if same_p:
                return Miss(gold, "wrong subject", self.x.show(same_p[0]))
            others = [f for f in subject if f.p in form.p]
            if others and not is_multi_valued(form.p[0], form.s):
                return Miss(gold, "other value", self.x.show(others[0]))
        form = gold.form
        if form.s != SELF_ID and not any(
            best_similarity(self.gold.entity_names(form.s), e.names) for e in self.x.entities
        ):
            return Miss(gold, "subject has no page", form.s)
        return Miss(gold, "not extracted")


def score_doc(gold: GoldDoc, extracted: Extracted, *, seconds: float = 0.0) -> DocScore:
    result = DocScore(gold, seconds=seconds, extracted=extracted)
    m = _Matcher(gold, extracted)
    bonus = {"required": 0.02, "optional": 0.01, "acceptable": 0.0}
    pairs = [
        (score + bonus[g.level], gi, fi)
        for gi, g in enumerate(gold.facts)
        for fi, f in enumerate(extracted.facts)
        if (score := m.score(g, f))
    ]
    used_gold: set[int] = set()
    used_got: set[int] = set()
    for score, gi, fi in sorted(pairs, key=lambda t: (-t[0], t[1], t[2])):
        if gi in used_gold or fi in used_got:
            continue
        used_gold.add(gi)
        used_got.add(fi)
        g, f = gold.facts[gi], extracted.facts[fi]
        match = Match(
            g, f, round(score, 3), date_problems(g, f), link_problem(g, f, extracted, gold)
        )
        result.matches.append(match)
    result.matches.sort(key=lambda x: gold.facts.index(x.gold))
    result.misses = [
        m.diagnose(g)
        for gi, g in enumerate(gold.facts)
        if gi not in used_gold and g.level != "acceptable"
    ]
    result.unverified = [f for fi, f in enumerate(extracted.facts) if fi not in used_got]

    claimed: set[str] = set()
    for entity in gold.entities:
        scored = [(best_similarity(entity.names, e.names), e) for e in extracted.entities]
        scored = [(s, e) for s, e in scored if s and e.id not in claimed]
        found = max(scored, key=lambda t: t[0])[1] if scored else None
        if found:
            claimed.add(found.id)
        result.entities.append(EntityResult(entity, found, gold.expected(entity)))
    for literal in gold.literals:
        for e in extracted.entities:
            if names_literal(literal, e.names):
                result.literal_pages.append((literal, e.id))
                claimed.add(e.id)
    result.unexpected_pages = [e for e in extracted.entities if e.id not in claimed]
    return result


def error_kinds(scores: Iterable[DocScore]) -> Counter[str]:
    """How often each kind of error occurs across documents (misses of required and optional
    facts, date problems, entity problems, unverified extras by predicate)."""
    kinds: Counter[str] = Counter()
    for s in scores:
        if s.error:  # its misses say nothing about extraction
            kinds["ingest failed"] += 1
            continue
        for miss in s.misses:
            kinds[f"{miss.gold.level} fact {miss.kind}"] += 1
        for match in s.matches:
            for problem in match.date_problems:
                kinds[f"date: {problem.split(' (')[0]}"] += 1
            if match.link_problem:
                kinds[f"degree: {match.link_problem.split(' (')[0]}"] += 1
        for e in s.entities:
            if e.found is None and e.expected:
                kinds["entity missing"] += 1
            elif e.found is None:
                continue
            elif not e.type_ok:
                kinds[f"entity type {e.gold.type} → {e.found.type}"] += 1
        kinds.update("literal turned into a page" for _ in s.literal_pages)
        for page in s.unexpected_pages:
            owner = best_similarity(s.gold.owner, page.names) if s.gold.owner else 0
            kinds["page for the owner" if owner else "page not in gold"] += 1
        for f in s.unverified:
            kinds[f"unverified {f.p}"] += 1
    return kinds


def totals(scores: Sequence[DocScore]) -> dict[str, float | int | None]:
    summed: Counter[str] = Counter()
    for s in scores:
        summed.update(s.counts())
    return metrics(dict(summed), sum(s.seconds for s in scores))


# --- running -----------------------------------------------------------------------------------


def default_cache_dir() -> Path:
    """Extraction cache shared by every run (scratch vaults are thrown away, so the cache
    cannot live in their ``.brain/``)."""
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "digital-self" / "eval-ingest"


def run_doc(gold: GoldDoc, llm: Any, workdir: Path, *, today: date = EVAL_TODAY) -> DocScore:
    """Ingest one document into a fresh vault under ``workdir`` and score it."""
    from brain.ingest import ingest
    from brain.vault.init import init_vault

    vault = Vault(init_vault(workdir / gold.name / "vault").root)
    start = time.perf_counter()
    try:
        report = ingest(gold.path, vault, llm, kind=gold.kind, today=today)
    except Exception as err:  # a crash on one document is a result, not the end of the run
        score = score_doc(gold, collect(vault), seconds=time.perf_counter() - start)
        score.error = f"{type(err).__name__}: {err}"
        return score
    seconds = time.perf_counter() - start
    queued = report.applied.queued if report.applied else []
    score = score_doc(gold, collect(vault, queued), seconds=seconds)
    score.chunks = len(report.chunks)
    score.notes = list(report.normalised.notes) if report.normalised else []
    score.failed_chunks = [f"{c.heading or c.index}: {reason}" for c, reason in report.failed]
    return score


# --- reporting ---------------------------------------------------------------------------------


def doc_json(s: DocScore) -> dict[str, Any]:
    x = s.extracted
    return {
        "doc": s.gold.path.name,
        "metrics": metrics(s.counts(), s.seconds),
        "counts": s.counts(),
        "chunks": s.chunks,
        "error": s.error,
        "matches": [
            {
                "level": m.gold.level,
                "gold": m.gold.show(),
                "got": x.show(m.got),
                "text": m.got.text,
                "date_problems": m.date_problems,
                "link_problem": m.link_problem,
            }
            for m in s.matches
        ],
        "misses": [
            {"level": m.gold.level, "gold": m.gold.show(), "kind": m.kind, "detail": m.detail}
            for m in s.misses
        ],
        "unverified": [{"got": x.show(f), "text": f.text} for f in s.unverified],
        "entities": [
            {
                "name": e.gold.name,
                "expected": e.expected,
                "expected_type": e.gold.type,
                "page": e.found.id if e.found else None,
                "type": e.found.type if e.found else None,
            }
            for e in s.entities
        ],
        "literal_pages": [{"literal": lit, "page": page} for lit, page in s.literal_pages],
        "unexpected_pages": [{"page": e.id, "type": e.type} for e in s.unexpected_pages],
        "notes": s.notes,
        "failed_chunks": s.failed_chunks,
    }


def report_json(scores: Sequence[DocScore], meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "meta": meta,
        "overall": totals(scores),
        "error_kinds": dict(error_kinds(scores).most_common()),
        "docs": [doc_json(s) for s in scores],
    }
