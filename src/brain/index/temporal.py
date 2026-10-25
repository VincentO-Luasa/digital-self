"""Valid-time interval logic for partial ISO dates (``YYYY``, ``YYYY-MM``, ``YYYY-MM-DD``).

A partial date stands for the whole period it names: ``2023-02`` covers 2023-02-01..2023-02-31
(day 31 is a string-comparison upper bound, valid for every month). A fact is valid at an as-of
date when its ``[from, to]`` interval overlaps the as-of period; a missing bound is open.
Both bounds are inclusive at their stated precision, so a fact ending ``2023-02`` and its
successor starting ``2023-02`` are both valid "as of 2023-02".
"""

from __future__ import annotations

from datetime import date

_START_PAD = {4: "-01-01", 7: "-01", 10: ""}
_END_PAD = {4: "-12-31", 7: "-31", 10: ""}


def _norm(value: str | date) -> str:
    text = value.isoformat() if isinstance(value, date) else value.strip()
    if len(text) not in _START_PAD:
        raise ValueError(f"expected YYYY, YYYY-MM or YYYY-MM-DD, got {value!r}")
    return text


def period_start(value: str | date) -> str:
    text = _norm(value)
    return text + _START_PAD[len(text)]


def period_end(value: str | date) -> str:
    text = _norm(value)
    return text + _END_PAD[len(text)]


def valid_at(valid_from: str | None, valid_to: str | None, as_of: str | date) -> bool:
    if valid_from and period_start(valid_from) > period_end(as_of):
        return False
    return not (valid_to and period_end(valid_to) < period_start(as_of))


def sql_valid_at(valid_from: str | None, valid_to: str | None, as_of: str | None) -> int:
    """SQLite function wrapper; malformed stored dates count as valid (never hide data)."""
    if as_of is None:
        return 1
    try:
        return int(valid_at(valid_from, valid_to, as_of))
    except ValueError:
        return 1
