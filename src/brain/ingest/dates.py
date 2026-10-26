"""Dates as documents write them, for grounding: which years a line or sentence mentions.

CVs and timelines abbreviate: "Sep 21 – July 22", "April 24 – Present", "09/2021", "2021-09",
"’19". A check that looks only for a four-digit year drops every one of a résumé's short forms.
"""

from __future__ import annotations

import re
from datetime import date

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
OPEN_END = re.compile(r"(?<!\w)(?:present|now|today|current(?:ly)?|ongoing|to date)(?!\w)|∞", re.I)

_MONTH = "|".join(sorted(MONTHS, key=len, reverse=True))
YEAR4 = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
# "Sep 21", "July 22", "Sept. 21", "Sep '21", "Sep-21"; not "May 21, 2024" (a day of the month)
MONTH_YY = re.compile(
    rf"(?<![a-z])(?:{_MONTH})\.?(?:\s*['’]|\s+|-)(\d{{2}})(?!\d)(?!(?:st|nd|rd|th)\b)"
    rf"(?!,?\s*(?:19|20)\d{{2}})",
    re.I,
)
NUMERIC_YY = re.compile(r"(?<![\d/.-])(?:0?[1-9]|1[0-2])/(\d{2})(?![\d/])")  # "09/21"
APOSTROPHE_YY = re.compile(r"(?<![\w'’])['’](\d{2})(?!\d)")  # "’19"


def century(yy: int, today: date | None = None) -> int:
    """A two-digit year as a full year: up to five years ahead of today in this century."""
    now = (today or date.today()).year
    year = now - now % 100 + yy
    return year - 100 if year > now + 5 else year


def years_in(text: str, today: date | None = None) -> set[int]:
    """The years ``text`` mentions, in any of the forms above."""
    years = {int(y) for y in YEAR4.findall(text)}
    for pattern in (MONTH_YY, NUMERIC_YY, APOSTROPHE_YY):
        years |= {century(int(yy), today) for yy in pattern.findall(text)}
    return years


def has_date(text: str) -> bool:
    """``text`` mentions a year (any form) or an open end ("present", "now", "∞")."""
    return bool(years_in(text) or OPEN_END.search(text))


_M = rf"(?<![a-z])(?:{_MONTH})\.?"
_DATE = (
    rf"(?:{_M}\s*)?(?<!\d)(?:19|20)\d{{2}}(?:-\d{{2}})?(?!\d)"  # 2021, Sep 2021, 2021-09
    rf"|{_M}(?:\s*['’]|\s+|-)\d{{2}}(?!\d)"  # Sep 21, Sep '21
    r"|(?<!\d)(?:0?[1-9]|1[0-2])[/.](?:19|20)?\d{2}(?!\d)"  # 09/2021, 09/21
    r"|['’]\d{2}(?!\d)"  # ’21
)
_END = rf"(?:{_DATE}|present|now|today|current|ongoing|∞)"
RANGE_START = re.compile(rf"^(?:{_DATE})\s*(?:[–—-]+|to|until)\s*{_END}", re.I)
DATE_END = re.compile(
    rf"(?:{_DATE}|(?:[–—-]|\bto|\bsince)\s*(?:present|now|today|∞))\)?[.,;]?$", re.I
)


def ends_with_date(line: str) -> bool:
    """The line ends with a date, or a range open to the present: "Org, City, Sep 21 – July 22",
    "Data Scientist, April 24 – Present"."""
    return bool(DATE_END.search(line.rstrip()))


def starts_with_range(line: str) -> bool:
    """The line starts with a date range: a timeline row ("2019 – 2021 · School")."""
    return bool(RANGE_START.match(line.lstrip("-*+ ").strip()))
