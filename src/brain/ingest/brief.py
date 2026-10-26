"""A document brief ("skeleton first"): one structured call over the whole document before its
parts are extracted.

Parts are read one at a time, so on its own a part does not know that the project it describes
belongs to the job two sections up, that "Atlas" and "the Atlas Migration project" are one
thing, or that the city in the intro was left four years ago. The brief holds that context:
subject, kind and voice; the periods (jobs, studies) and homes; a glossary of names with their
type, aliases, parent (the organisation a project belongs to, the city an organisation is in)
and role for the subject.

Normalisation (``normalise.build_changeset``) uses it: names and aliases merge variants into one
entity, parents give structural facts (``part_of``, ``located_in``), roles give ``member_of``
and sharpen a vague ``knows``, homes date past cities, periods keep separate stints at one
employer apart. ``render`` turns it into a prompt block (``extract.build_messages(brief=…)``,
first in the user message so it is a cached prefix across parts) and ``header`` into a line of
context for the typed checks; both were measured on the eval set and are off by default
(``pipeline.BRIEF_USES``).

The model fills gaps (a job's start becomes a move, an employer's city a home), so ``ground``
keeps only dates and homes the document states, and every drop is noted. The brief adds facts
only about entities the extraction already uses, or memberships and homes it states.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import httpx
from pydantic import BaseModel

from brain.ingest.chunking import Chunk
from brain.ingest.dates import ends_with_date, starts_with_range, years_in
from brain.ingest.extract import StructuredModel
from brain.llm.base import StructuredOutputError
from brain.prompts import load_prompt
from brain.tokens import estimate_tokens

log = logging.getLogger(__name__)

FULL_TEXT_TOKENS = 6000  # a longer document is briefed from its outline (prefill ~400 tok/s)
PART_PREVIEW_TOKENS = 150  # per part, when briefed from the outline
BRIEF_MAX_TOKENS = 2048
BRIEF_TEMPERATURE = 0.0  # one call per document: greedy, so a re-run gives the same brief
MAX_NAMES = 20

PeriodKind = Literal["job", "study"]
NameType = Literal["person", "org", "project", "place", "event", "team", "tool"]
Role = Literal[
    "employer",
    "school",
    "member",
    "project",
    "side project",
    "colleague",
    "friend",
    "family",
    "home",
    "hometown",
    "other",
]
PAGE_TYPES = frozenset({"person", "org", "project", "place", "event"})


class Period(BaseModel):
    kind: PeriodKind
    title: str
    org: str
    start: str
    end: str


class HomePeriod(BaseModel):
    city: str
    start: str
    end: str


class BriefName(BaseModel):
    name: str
    type: NameType
    aliases: list[str]
    parent: str
    role: Role


class Brief(BaseModel):
    reasoning: str
    subject: str
    kind: Literal["homepage", "cv", "bio", "notes", "export", "article", "other"]
    voice: Literal["first", "third", "mixed"]
    periods: list[Period]
    homes: list[HomePeriod]
    names: list[BriefName]


@dataclass
class BriefReport:
    brief: Brief | None
    seconds: float = 0.0
    full_text: bool = True  # False: briefed from the outline of a long document
    notes: list[str] = field(default_factory=list)


def _fold(text: str) -> str:
    return " ".join(text.split()).casefold()


# --- the call -----------------------------------------------------------------------------------


def brief_source(text: str, chunks: Sequence[Chunk]) -> tuple[str, bool]:
    """The text the brief reads: the whole document when it fits ``FULL_TEXT_TOKENS``, else each
    part's heading and first ``PART_PREVIEW_TOKENS`` tokens."""
    if estimate_tokens(text) <= FULL_TEXT_TOKENS or len(chunks) <= 1:
        return text.strip(), True
    lines = []
    for chunk in chunks:
        words, out = chunk.text.split(), []
        for word in words:
            out.append(word)
            if estimate_tokens(" ".join(out)) >= PART_PREVIEW_TOKENS:
                out.append("…")
                break
        lines.append(f"## Part {chunk.index + 1}: {chunk.heading or '(start)'}\n{' '.join(out)}")
    return "\n\n".join(lines), False


def build_messages(source: str, *, doc_title: str, full_text: bool) -> list[dict[str, str]]:
    what = "DOCUMENT" if full_text else "DOCUMENT OUTLINE (each part's first lines)"
    user = f"Document: {doc_title}\n\n{what}:\n<<<\n{source}\n>>>"
    return [
        {"role": "system", "content": load_prompt("ingest_brief")},
        {"role": "user", "content": user},
    ]


BRIEF_ERRORS = (StructuredOutputError, httpx.HTTPError)


def make_brief(
    llm: StructuredModel, text: str, chunks: Sequence[Chunk], *, doc_title: str
) -> BriefReport:
    """The document's brief (``brief=None`` when the call fails: ingest goes on without one)."""
    source, full = brief_source(text, chunks)
    start = time.perf_counter()
    try:
        brief = llm.structured(
            build_messages(source, doc_title=doc_title, full_text=full),
            Brief,
            temperature=BRIEF_TEMPERATURE,
            max_tokens=BRIEF_MAX_TOKENS,
        )
    except BRIEF_ERRORS as err:
        log.warning("ingest brief failed: %s", err)
        return BriefReport(None, time.perf_counter() - start, full, [f"brief: none ({err})"])
    notes: list[str] = []
    brief = ground(clean_brief(brief), text, notes)
    return BriefReport(brief, time.perf_counter() - start, full, notes)


def _is_name(text: str) -> bool:
    """A name, not a description: a digit, one capitalised word ("Atlas"), or a capital after the
    first word ("Atlas Migration"). "my team" or "Trip planner" (sentence case) describe."""
    words = text.split()
    if not words:
        return False
    if any(c.isdigit() for c in text):
        return True
    if len(words) == 1:
        return words[0][:1].isupper()
    return any(w[:1].isupper() for w in words[1:])


def clean_brief(brief: Brief) -> Brief:
    """Drop empty and repeated names (first wins), cap them, and strip whitespace."""
    seen: set[str] = set()
    names = []
    for entry in brief.names:
        name = " ".join(entry.name.split())
        if not name or _fold(name) in seen or _fold(name) == _fold(brief.subject):
            continue
        seen.add(_fold(name))
        aliases = [" ".join(a.split()) for a in entry.aliases]
        aliases = [a for a in dict.fromkeys(aliases) if _is_name(a) and _fold(a) != _fold(name)]
        parent = " ".join(entry.parent.split())
        parent = "" if _fold(parent) == _fold(name) else parent
        names.append(entry.model_copy(update={"name": name, "aliases": aliases, "parent": parent}))
    periods = [p for p in brief.periods if p.title.strip() or p.org.strip()]
    homes = [
        h.model_copy(update={"city": h.city.split(",")[0].strip()})
        for h in brief.homes
        if h.city.split(",")[0].strip()
    ]
    return brief.model_copy(update={"names": names[:MAX_NAMES], "periods": periods, "homes": homes})


# --- grounding ----------------------------------------------------------------------------------

WRAPPED = 60  # a line at least this long without end punctuation is wrapped prose: joined
LIST_LINE = re.compile(r"^(?:[-*+|#>]|\d+[.)])")
YEAR = re.compile(r"^(\d{4})")
# A sentence that places someone's home (an employer's or a school's city is not one)
RESIDENCE = re.compile(
    r"\b(?:live[sd]?|living|moved?|moving|relocat\w*|based|home|settled|grew up)\b", re.I
)


def _wrapped(prev: str, line: str) -> bool:
    """``line`` continues the prose of ``prev``: a long line without end punctuation, unless one
    of them is a CV or timeline line (``prev`` ends with a date and ``line`` does not go on in
    lower case; ``line`` starts with a date range, or starts with a capital and ends with a date
    and no full stop) or ``line`` is a list item."""
    if len(prev) < WRAPPED or re.search(r"[.!?:;]$", prev) or LIST_LINE.match(line):
        return False
    if ends_with_date(prev) and not line[:1].islower():
        return False
    if line[:1].isupper() and ends_with_date(line) and not line.endswith("."):
        return False
    return not starts_with_range(line)


def units(text: str) -> list[str]:
    """Sentences and lines of ``text``: wrapped prose lines are joined into sentences; short
    lines (a CV's "Org, City" above "Title, dates"), lines ending with a date ("MSc …, Stockholm,
    2013 – 2015"), timeline rows and list lines stay apart, and a record line (``_record``) is
    one unit, never cut into sentences."""
    out: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        lines: list[str] = []
        for line in (ln.strip() for ln in block.splitlines()):
            if not line:
                continue
            prev = lines[-1] if lines else ""
            if _wrapped(prev, line):
                lines[-1] = f"{prev} {line}"
            else:
                lines.append(line)
        for line in lines:
            if _record(line):  # its "sentences" are fields of one entry
                out.append(line)
            else:
                out += [s for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
    return out


def _record(line: str) -> bool:
    """A CV, timeline or table line: it starts with a date range ("2019 – 2021 · School · MSc.
    Thesis on …") or ends with a date and no full stop ("Org,Title,"Did X. Did Y.",Jan 2024,")."""
    end = line.rstrip()
    return starts_with_range(end) or (ends_with_date(end) and not end.endswith("."))


def _named(unit: str, names: Sequence[str]) -> bool:
    return any(
        re.search(rf"(?<!\w){r'\s+'.join(map(re.escape, n.split()))}(?!\w)", unit, re.I)
        for n in names
        if n.strip()
    )


def stated(sentences: Sequence[str], names: Sequence[str], value: str) -> bool:
    """A sentence or line names one of ``names`` and gives the year of ``value``, in any form
    (2021, Sep 21, 09/2021, 2021-09, ’21)."""
    m = YEAR.match(value.strip())
    if not m:
        return False
    year = int(m.group(1))
    return any(_named(u, names) and year in years_in(u) for u in sentences)


def ground(brief: Brief, text: str, notes: list[str]) -> Brief:
    """Keep only dates the document states next to what they date: a home's start or end must
    share a sentence with the city (an end may also be the next home's stated start), a period's
    with its organisation or title. A home no sentence names together with a word of residence
    (lives, moved, based, …) is dropped: the model takes an employer's city for a home. The
    model also fills gaps from nearby dates (a job's start becomes a move); those dates are
    blanked. Everything dropped is noted."""
    sentences = units(text)
    index = name_index(brief)

    def names_of(name: str) -> list[str]:
        entry = index.get(_fold(name))
        return [name, *(entry.aliases if entry else [])]

    homes_out: list[HomePeriod] = []
    for h in brief.homes:
        names = names_of(h.city)
        if not any(_named(u, names) and RESIDENCE.search(u) for u in sentences):
            notes.append(
                f"brief: dropped home {h.city!r}: no sentence says the subject lives there"
            )
            continue
        start = h.start if stated(sentences, names, h.start) else ""
        if h.start.strip() and not start:
            notes.append(f"brief: home {h.city}: start {h.start} not stated with it; dropped")
        homes_out.append(h.model_copy(update={"start": start}))
    for i, h in enumerate(homes_out):
        end = h.end.strip()
        if not end or _fold(end) in ("now", "present", "current"):
            continue
        nxt = homes_out[i + 1].start if i + 1 < len(homes_out) else ""
        if not stated(sentences, names_of(h.city), end) and end != nxt:
            notes.append(f"brief: home {h.city}: end {end} not stated with it; dropped")
            homes_out[i] = h.model_copy(update={"end": ""})

    periods: list[Period] = []
    for p in brief.periods:
        names = [n for n in (*names_of(p.org), p.title) if n.strip()]
        update = {}
        for label in ("start", "end"):
            value = getattr(p, label).strip()
            if YEAR.match(value) and not stated(sentences, names, value):
                notes.append(f"brief: {p.title} at {p.org}: {label} {value} not stated; dropped")
                update[label] = ""
        periods.append(p.model_copy(update=update))
    return brief.model_copy(update={"homes": homes_out, "periods": periods})


# --- rendering ----------------------------------------------------------------------------------


def _span(start: str, end: str) -> str:
    start, end = start.strip(), end.strip()
    if not start and not end:
        return "undated"
    return f"{start or '?'}–{end or '?'}"


def header(brief: Brief) -> str:
    """One line: who and what the document is."""
    who = brief.subject.strip() or "an unnamed person"
    voice = {"first": "first person", "third": "third person", "mixed": "mixed voice"}
    return f"About: {who} ({brief.kind}, {voice[brief.voice]})"


def render(brief: Brief, *, dates: bool = False, timeline: bool = True) -> str:
    """The brief as a compact block for prompts. Periods are shown without dates by default: the
    brief rounds "Jun 2014" to "2014", and an extractor shown the brief's years copies them over
    the excerpt's months."""
    types = {_fold(n.name): n.type for n in brief.names}
    lines = [header(brief)]
    if timeline and (brief.periods or brief.homes):
        lines.append("Timeline (oldest first):")
        for p in brief.periods:
            what = " at ".join(x for x in (" ".join(p.title.split()), p.org.strip()) if x)
            span = f" {_span(p.start, p.end)}" if dates else ""
            lines.append(f"- {p.kind}{span}: {what}")
        for h in brief.homes:
            span = f" {_span(h.start, h.end)}" if dates else ""
            lines.append(f"- home{span}: {h.city.strip()}")
    if brief.names:
        lines.append("Names:")
        for n in brief.names:
            inside = ""
            if n.parent:
                word = "in" if types.get(_fold(n.parent)) == "place" else "of"
                inside = f" {word} {n.parent}"
            also = f"; also {', '.join(n.aliases)}" if n.aliases else ""
            role = "" if n.role == "other" else f": {n.role}"
            lines.append(f"- {n.name} ({n.type}{inside}{also}){role}")
    return "\n".join(lines)


# --- what normalisation reads -------------------------------------------------------------------


def name_index(brief: Brief | None) -> dict[str, BriefName]:
    """Folded name or alias → its glossary entry (a name wins over another entry's alias)."""
    out: dict[str, BriefName] = {}
    if brief is None:
        return out
    for entry in brief.names:
        out.setdefault(_fold(entry.name), entry)
    for entry in brief.names:
        for alias in entry.aliases:
            out.setdefault(_fold(alias), entry)
    return out


DESCRIPTOR = re.compile(r"^(?:[a-z][a-z-]*|v?\d+(?:\.\d+)*)$")  # "design", "system", "2.0", "v3"


def lookup(index: dict[str, BriefName], name: str) -> BriefName | None:
    """The brief entry ``name`` refers to: by name or alias, or as a brief name followed by
    lowercase words or a version ("Atlas Migration project", "Atlas 2.0" for "Atlas
    Migration"/"Atlas"), the longest such name first."""
    words = name.split()
    if not words:
        return None
    if entry := index.get(_fold(name)):
        return entry
    for n in range(len(words) - 1, 0, -1):
        rest = words[n:]
        if all(DESCRIPTOR.match(w) for w in rest) and (
            entry := index.get(_fold(" ".join(words[:n])))
        ):
            return entry if entry.type in PAGE_TYPES else None
    return None


DATE = re.compile(r"^\d{4}(?:-\d{2})?$")


@dataclass(frozen=True)
class Home:
    city: str
    start: str | None
    end: str | None  # None: still true or unknown (see ``current``)
    current: bool


def homes(brief: Brief | None) -> list[Home]:
    """The subject's cities in order, each ending where the next begins when its own end is not
    given; only the last one can be current (its end "now", or none after a known start): two
    homes marked "now" would be two current cities."""
    if brief is None:
        return []
    periods = [h for h in brief.homes if h.city.strip()]
    out: list[Home] = []
    for i, p in enumerate(periods):
        start = p.start.strip() if DATE.match(p.start.strip()) else None
        end = p.end.strip() if DATE.match(p.end.strip()) else None
        now = _fold(p.end) in ("now", "present", "current")
        last = i == len(periods) - 1
        if end is None and not last:
            nxt = periods[i + 1].start.strip()
            end = nxt if DATE.match(nxt) else None
        current = last and end is None and (now or start is not None)
        out.append(Home(" ".join(p.city.split()), start, end, current))
    return out
