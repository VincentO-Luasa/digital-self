"""Pack search hits into the per-turn memory block, gist-first, under a token budget.

Pass 1 adds the short form of every hit in rank order (fact lines, episode gists, the first
sentence of page summaries). Pass 2 spends what is left on the full text of the top two
episodes or summaries (docs/retrieval.md, "Packing").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from brain.index.search import Hit
from brain.tokens import estimate_tokens
from brain.vault.generated import first_sentence

EXPAND_TOP = 2
HEADER = "Memory notes for this message (most relevant first):"
EMPTY = "Memory notes for this message: nothing relevant found."


@dataclass
class MemoryBlock:
    text: str
    hits: list[Hit] = field(default_factory=list)  # every candidate, in rank order
    used: list[str] = field(default_factory=list)  # ids that made it into the block
    tokens: int = 0


def short_form(hit: Hit) -> str:
    if hit.kind == "episode":
        return hit.text.split("\n", 1)[0].strip()
    if hit.kind == "summary":
        return first_sentence(hit.text, limit=240)
    return " ".join(hit.text.split())


def long_form(hit: Hit) -> str:
    return " ".join(hit.text.split())


def _line(hit: Hit, text: str) -> str:
    when = f" ({hit.date[:10]})" if hit.kind in ("episode", "observation") and hit.date else ""
    return f"- [{hit.id}]{when} {text}"


def pack(hits: Sequence[Hit], budget_tokens: int, *, skip_text: str = "") -> MemoryBlock:
    """``skip_text`` (the core card) drops hits whose text it already contains."""
    candidates = [h for h in hits if not (h.kind == "fact" and h.text in skip_text)]
    lines: dict[str, str] = {}
    budget = budget_tokens - estimate_tokens(HEADER)
    for hit in candidates:
        line = _line(hit, short_form(hit))
        cost = estimate_tokens(line) + 1
        if cost <= budget:
            lines[hit.id] = line
            budget -= cost
    expandable = [h for h in candidates if h.id in lines and h.kind in ("episode", "summary")]
    for hit in expandable[:EXPAND_TOP]:
        full = _line(hit, long_form(hit))
        extra = estimate_tokens(full) - estimate_tokens(lines[hit.id])
        if full != lines[hit.id] and extra <= budget:
            lines[hit.id] = full
            budget -= extra
    if not lines:
        return MemoryBlock(EMPTY, list(hits), [], estimate_tokens(EMPTY))
    text = "\n".join([HEADER, *lines.values()])
    return MemoryBlock(text, list(hits), list(lines), estimate_tokens(text))
