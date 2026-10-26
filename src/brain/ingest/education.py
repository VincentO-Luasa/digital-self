"""Which school a degree is from, read from the part of the document that states it.

Extraction gives a ``degree`` fact and separate ``studied_at`` facts; nothing says which school
granted which degree. Pairing them later by overlapping dates goes wrong on ordinary timelines:
"2017 — 2019 · School C · Preparatory classes" and "2019 — 2022 · School A · Engineering degree"
share the year 2019, and a double degree's row names two schools. So normalisation links a
degree to a school (the ``at::`` qualifier, docs/vault-schema.md) from the text itself:

1. the unit (sentence, CV line, timeline row, record) that names most of the degree's words;
2. the document's schools named in that unit, else in the short line just above it (an
   "Org, City" line or a heading above "Title, dates");
3. of several, the one whose ``studied_at`` has the degree's dates; else, in a record or row
   (it starts with a date range or ends with a date), the nearest school named before the
   degree ("2019 — 2020 · School B · MSc X, double degree with School A"), in prose the nearest
   after it ("a master's in X at School B"), then the other side.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from brain.ingest.brief import _record, units

STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "in",
        "at",
        "for",
        "and",
        "to",
        "with",
        "on",
        "from",
        "by",
        "degree",
        "diploma",
        "programme",
        "program",
    ]
)
COVER = 0.6  # share of the degree's words a unit must name
SHORT_LINE = 12  # words: a line above that may hold the school ("Org, City", a heading)


@dataclass(frozen=True)
class School:
    id: str
    names: tuple[str, ...]  # title and aliases
    dates: tuple[str | None, str | None] = (None, None)  # its studied_at in the same part


def _fold(text: str) -> str:
    """Lowercase without accents, one character per character (positions stay valid)."""
    return "".join(unicodedata.normalize("NFKD", c)[:1] or c for c in text).lower()


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", _fold(text)) if w not in STOP and len(w) > 1]


def _cover(unit: str, value: str) -> tuple[float, int]:
    """Share of the value's words the unit names, and where the first of them is."""
    words = _words(value)
    if not words:
        return 0.0, -1
    folded = _fold(unit)
    spots = [m.start() for w in words if (m := re.search(rf"(?<![a-z0-9]){w}", folded))]
    return len(spots) / len(words), min(spots, default=-1)


def _mentions(unit: str, school: School) -> int:
    """Where the unit first names the school (-1 if it does not)."""
    folded = _fold(unit)
    spots = []
    for name in school.names:
        words = _fold(name).split()
        if not words:
            continue
        pattern = r"\s+".join(map(re.escape, words))
        if m := re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", folded):
            spots.append(m.start())
    return min(spots, default=-1)


def degree_school(
    value: str, dates: tuple[str | None, str | None], schools: Sequence[School], text: str
) -> tuple[str | None, str]:
    """The id of the school ``value`` (a degree) is from, and how it was found ("" if none)."""
    if not schools or not text.strip():
        return None, ""
    parts = units(text)
    scored = [(_cover(u, value), i) for i, u in enumerate(parts)]
    best = max(scored, key=lambda t: (t[0][0], -t[1]), default=None)
    if best is None or best[0][0] < COVER:
        return None, ""
    (_, at), i = best
    unit = parts[i]
    named = [(s, pos) for s in schools if (pos := _mentions(unit, s)) >= 0]
    how = "its line"
    if not named and i > 0:
        above = parts[i - 1]
        if len(above.split()) <= SHORT_LINE and not above.rstrip().endswith("."):
            named = [(s, 0) for s in schools if _mentions(above, s) >= 0]
            how = "the line above"
            if len(named) > 1:
                return None, ""
    if not named:
        return None, ""
    if len(named) == 1:
        return named[0][0].id, how
    same = [s for s, _ in named if any(dates) and s.dates == dates]
    if len(same) == 1:
        return same[0].id, f"{how}, same dates"
    before = sorted(((at - pos, s) for s, pos in named if pos < at), key=lambda t: t[0])
    after = sorted(((pos - at, s) for s, pos in named if pos > at), key=lambda t: t[0])
    sides = (before, after) if _record(unit) else (after, before)
    for side in sides:
        if side:
            return side[0][1].id, f"{how}, nearest"
    return None, ""
