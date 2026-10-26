"""Name similarity for duplicate candidates (code only, no model).

Names are folded (accents, case, punctuation), generic words of the entity type are set aside
("University of", "École", "Inc", "the … project"), and two names are compared by:

- **core**: the same words in any order once generic words are set aside
  ("University of Westbridge" / "Westbridge University", "École Varenne" / "Varenne");
- **acronym**: one name is the initials of the other ("KDC" / "Kestrel Data Cooperative");
- **typo**: a small edit distance between the cores ("Nortwind Labs" / "Northwind Labs");
- **contains**: every core word of one is in the other ("Atlas" / "Atlas Migration");
- **first name**: two person names sharing their first word ("Sam" / "Sam Brennan");
- **shared word**: other names starting with the same distinctive word ("Atlas Migration" /
  "Atlas Reporting"), the weakest signal.

These only *propose* pairs; a typed decision says whether they are the same.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

STOP = frozenset({"the", "of", "and", "a", "an", "at", "for", "in", "on", "de", "la", "le", "du"})
STOP |= frozenset({"des", "del", "der", "van", "von", "da", "do", "dos", "y", "et", "und"})
GENERIC: dict[str, frozenset[str]] = {
    "org": frozenset(
        [
            "university",
            "univ",
            "universite",
            "universidad",
            "universidade",
            "universita",
            "college",
            "school",
            "ecole",
            "institute",
            "institut",
            "inc",
            "ltd",
            "llc",
            "gmbh",
            "sa",
            "sas",
            "bv",
            "co",
            "corp",
            "company",
            "group",
            "plc",
        ]
    ),
    "project": frozenset(["project", "projects", "programme", "program", "initiative"]),
    "person": frozenset(["dr", "prof", "mr", "mrs", "ms", "mx", "sir"]),
    "place": frozenset(["city", "town"]),
    "event": frozenset(),
    "topic": frozenset(),
}
WORD = re.compile(r"[a-z0-9]+")


def fold(text: str) -> str:
    """Lower case, accents stripped, punctuation as spaces."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(WORD.findall(ascii_text.casefold()))


def words(text: str) -> list[str]:
    return fold(text).split()


def core(text: str, kind: str) -> tuple[str, ...]:
    """The name's distinctive words, in order (stop words and the type's generic words out;
    a name made only of them keeps its words)."""
    all_words = [w for w in words(text) if w not in STOP]
    kept = [w for w in all_words if w not in GENERIC.get(kind, frozenset())]
    return tuple(kept or all_words)


def initials(text: str) -> set[str]:
    """Acronyms a name may be written as: initials of all words and of non-stop words."""
    ws = words(text)
    out = {"".join(w[0] for w in ws)}
    out.add("".join(w[0] for w in ws if w not in STOP))
    return {a for a in out if len(a) >= 2}


def edits(a: str, b: str) -> int:
    """Damerau-Levenshtein distance (adjacent transpositions count as one edit)."""
    if a == b:
        return 0
    rows = [list(range(len(b) + 1))]
    for i, ca in enumerate(a, 1):
        row = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            row[j] = min(rows[-1][j] + 1, row[j - 1] + 1, rows[-1][j - 1] + cost)
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                row[j] = min(row[j], rows[-2][j - 2] + 1)
        rows.append(row)
    return rows[-1][-1]


def close(a: str, b: str) -> bool:
    """Two words (or word strings) a typo apart: ≤ 1 edit from 4 letters, ≤ 2 from 9."""
    longest = max(len(a), len(b))
    budget = 2 if longest >= 9 else 1 if longest >= 4 else 0
    return edits(a, b) <= budget


@dataclass(frozen=True)
class NameMatch:
    how: str  # core | acronym | typo | contains | first-name | shared-word
    a: str
    b: str

    @property
    def strength(self) -> int:
        strengths = {"core": 5, "acronym": 4, "typo": 4, "contains": 2}
        return strengths.get(self.how, 1)


def compare(a: str, b: str, kind: str) -> NameMatch | None:
    """How names ``a`` and ``b`` of two entities of type ``kind`` resemble each other."""
    ca, cb = core(a, kind), core(b, kind)
    if not ca or not cb:
        return None
    if sorted(ca) == sorted(cb):
        return NameMatch("core", a, b)
    sa, sb = "".join(ca), "".join(cb)
    for short, long in ((sa, b), (sb, a)):
        acronym = len(short) >= 2 and " " not in short and short in initials(long)
        if acronym and len(words(long)) >= 2:
            return NameMatch("acronym", a, b)
    same_shape = len(ca) == len(cb) and max(len(sa), len(sb)) >= 5
    if same_shape and all(close(x, y) for x, y in zip(ca, cb, strict=True)):
        return NameMatch("typo", a, b)
    if close(" ".join(ca), " ".join(cb)) and max(len(sa), len(sb)) >= 6:
        return NameMatch("typo", a, b)
    if set(ca) < set(cb) or set(cb) < set(ca):
        if kind == "person" and len(ca) >= 2 and len(cb) >= 2:
            return None  # "Sam Okafor" / "Sam Okafor Brennan": different people, at best
        return NameMatch("contains", a, b)
    if kind == "person" and ca[0] == cb[0]:
        return NameMatch("first-name", a, b)
    if kind != "person" and ca[0] == cb[0] and len(ca[0]) >= 4:
        return NameMatch("shared-word", a, b)  # "Atlas Migration" / "Atlas Reporting"
    return None


def best_match(names_a: Iterable[str], names_b: Iterable[str], kind: str) -> NameMatch | None:
    found = [m for x in names_a for y in names_b if (m := compare(x, y, kind))]
    return max(found, key=lambda m: m.strength, default=None)


def different_people(names_a: Iterable[str], names_b: Iterable[str]) -> bool:
    """Both have a full name (first + last) and the last names differ beyond a typo: two
    people, whatever else they share ("Sam Okafor" / "Sam Brennan")."""
    full_a = [core(n, "person") for n in names_a if len(core(n, "person")) >= 2]
    full_b = [core(n, "person") for n in names_b if len(core(n, "person")) >= 2]
    if not full_a or not full_b:
        return False
    return not any(_maybe_same(x, y) for x in full_a for y in full_b)


def _maybe_same(x: tuple[str, ...], y: tuple[str, ...]) -> bool:
    """Same last name (give or take a typo), also when one is written "Last, First"."""
    return sorted(x) == sorted(y) or any(
        close(a, b) for a, b in ((x[-1], y[-1]), (x[0], y[-1]), (x[-1], y[0]))
    )
