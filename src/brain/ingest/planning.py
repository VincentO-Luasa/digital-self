"""Model-planned split of a document too large for one extraction chunk.

The model never sees the full text: it gets a compact outline (one line per block: number,
size, first words) and answers with the blocks where topically coherent parts start. Code
validates the plan (repairs indices, keeps headings with their text, merges tiny parts, splits
parts over the cap at headings/paragraphs) and turns each part into a ``Chunk``. A long outline
is planned in overlapping windows and stitched. If planning fails, the document is split at
headings (``chunk_markdown``), as before.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

from pydantic import BaseModel

from brain.ingest.chunking import (
    CHUNK_TOKENS,
    Block,
    Chunk,
    chunk_markdown,
    join_blocks,
    pack_blocks,
    split_blocks,
)
from brain.ingest.extract import StructuredModel
from brain.llm.base import StructuredOutputError
from brain.prompts import load_prompt
from brain.tokens import estimate_tokens

log = logging.getLogger(__name__)

PREVIEW_WORDS = 16  # first words of a block shown in the outline
WINDOW_TOKENS = 6000  # a longer outline is planned in windows
WINDOW_OVERLAP = 8  # blocks shared by consecutive windows
MIN_PART = 0.25  # a part under this share of the cap joins a neighbour when both fit
TITLE_CHARS = 80
PLAN_MAX_TOKENS = 2048
PLAN_TEMPERATURE = 0.0  # greedy: at 0.2 the model sometimes kept a long stretch whole


class PlannedPart(BaseModel):
    first_block: int
    title: str


class SplitPlan(BaseModel):
    reasoning: str
    parts: list[PlannedPart]


@dataclass(frozen=True)
class Part:
    title: str
    start: int  # first block
    end: int  # one past the last block
    tokens: int


@dataclass
class PlanReport:
    parts: list[Part]
    chunks: list[Chunk]
    planned: bool  # False: the model was not asked (one chunk) or the plan failed
    seconds: float = 0.0
    calls: int = 0
    notes: list[str] = field(default_factory=list)


# --- outline ------------------------------------------------------------------------------------


def outline_line(block: Block) -> str:
    """``[n] # Heading`` or ``[n] (size tok) first words …``."""
    words = block.text.split()
    preview = " ".join(words[:PREVIEW_WORDS]) + (" …" if len(words) > PREVIEW_WORDS else "")
    if block.heading:
        return f"[{block.index}] {preview}"
    return f"[{block.index}] ({block.tokens} tok) {preview}"


def long_stretches(blocks: list[Block], cap: int, lo: int, hi: int) -> list[tuple[int, int, int]]:
    """``(start, end, tokens)`` of each run of blocks in ``[lo, hi)`` between two headings (the
    heading included) that is larger than ``cap``: the model has to split it without a heading
    to go by, and does so only when told (it otherwise keeps a headed section whole)."""
    out: list[tuple[int, int, int]] = []
    start = lo
    for i in range(lo, hi + 1):
        if i == hi or (blocks[i].heading and i > start):
            tokens = sum(b.tokens for b in blocks[start:i])
            if tokens > cap:
                out.append((start, i, tokens))
            start = i
    return out


def windows(blocks: list[Block], budget: int | None = None) -> list[tuple[int, int]]:
    """``[start, end)`` block ranges whose outline fits ``budget`` (default ``WINDOW_TOKENS``)
    tokens; consecutive windows share ``WINDOW_OVERLAP`` blocks (fewer if a window is short)."""
    budget = budget or WINDOW_TOKENS
    sizes = [estimate_tokens(outline_line(b)) + 1 for b in blocks]
    out: list[tuple[int, int]] = []
    start = 0
    while start < len(blocks):
        end, used = start, 0
        while end < len(blocks) and (end == start or used + sizes[end] <= budget):
            used += sizes[end]
            end += 1
        out.append((start, end))
        if end >= len(blocks):
            break
        start = max(start + 1, end - min(WINDOW_OVERLAP, (end - start) // 4))
    return out


def build_messages(
    blocks: list[Block], lo: int, hi: int, *, doc_title: str, total: int, cap: int
) -> list[dict[str, str]]:
    window = blocks[lo:hi]
    tokens = sum(b.tokens for b in window)
    long = []
    for s, e, n in long_stretches(blocks, cap, lo, hi):
        label = blocks[s].text if blocks[s].heading else "no heading"
        long.append(
            f"- blocks {s} to {e - 1} ({n} tok, {label}): "
            f"split into {math.ceil(n / cap)} or more parts where the subject changes"
        )
    user = (
        f"Document: {doc_title}\n"
        f"Token cap per part: {cap}\n"
        f"Outline of blocks {lo} to {hi - 1} of {total} ({tokens} tokens):\n\n"
        + "\n".join(outline_line(b) for b in window)
        + "\n\nToo long for one part, no heading inside:\n"
        + ("\n".join(long) or "- none")
    )
    return [
        {"role": "system", "content": load_prompt("ingest_split")},
        {"role": "user", "content": user},
    ]


# --- validation ---------------------------------------------------------------------------------


def _title(raw: str, block: Block) -> str:
    title = " ".join(raw.split())[:TITLE_CHARS].strip()
    return title or (block.trail[-1] if block.trail else "Part")


def repair_starts(
    plan: SplitPlan, blocks: list[Block], lo: int, hi: int, notes: list[str]
) -> list[tuple[int, str]] | None:
    """Sorted, unique, in-range part starts for the window ``[lo, hi)``, the first at ``lo``;
    ``None`` if the plan is unusable (no parts, or most indices out of range)."""
    raw = [(p.first_block, p.title) for p in plan.parts]
    if not raw:
        return None
    outside = [i for i, _ in raw if not lo <= i < hi]
    if len(outside) * 2 > len(raw):
        return None
    if outside:
        notes.append(f"plan: dropped part starts outside blocks {lo}-{hi - 1}: {outside}")
    inside = [(i, t) for i, t in raw if lo <= i < hi]
    if [i for i, _ in inside] != sorted({i for i, _ in inside}):
        notes.append("plan: part starts were unsorted or repeated; sorted and deduplicated")
    starts: dict[int, str] = {}
    for i, title in sorted(inside, key=lambda r: r[0]):
        starts.setdefault(i, _title(title, blocks[i]))
    if lo not in starts:
        notes.append(f"plan: no part started at block {lo}; added one")
        starts[lo] = _title("", blocks[lo])
    return sorted(starts.items())


def _heading_starts(blocks: list[Block], lo: int, hi: int, cap: int) -> list[tuple[int, str]]:
    return [(g[0].index, _title("", g[0])) for g in pack_blocks(blocks[lo:hi], cap)]


def _keep_headings(starts: list[tuple[int, str]], blocks: list[Block]) -> list[tuple[int, str]]:
    """A heading goes with the text under it: a part start moves back over the headings right
    before it, and a part made of headings only joins the next one (taking its title)."""
    out: list[tuple[int, str]] = []
    for i, title in starts:
        if out and all(b.heading for b in blocks[out[-1][0] : i]):
            out[-1] = (out[-1][0], title)
            continue
        prev = out[-1][0] if out else -1
        while i - 1 > prev and blocks[i - 1].heading:
            i -= 1
        out.append((i, title))
    return out


def _parts(starts: list[tuple[int, str]], blocks: list[Block]) -> list[Part]:
    ends = [s for s, _ in starts[1:]] + [len(blocks)]
    return [
        Part(title, s, e, estimate_tokens(join_blocks(blocks[s:e])))
        for (s, title), e in zip(starts, ends, strict=True)
    ]


def _merge_tiny(parts: list[Part], blocks: list[Block], cap: int, notes: list[str]) -> list[Part]:
    """Join neighbouring parts when one is under ``MIN_PART`` of the cap and both fit the cap:
    each part is an extraction call, and a few lines of context are worth more than a cut."""

    def merged(a: Part, b: Part) -> Part:
        tokens = estimate_tokens(join_blocks(blocks[a.start : b.end]))
        title = f"{a.title}; {b.title}"
        if len(title) > TITLE_CHARS:
            title = max(a, b, key=lambda p: p.tokens).title
        return Part(title, a.start, b.end, tokens)

    out: list[Part] = []
    for part in parts:
        if out and min(part.tokens, out[-1].tokens) < cap * MIN_PART:
            candidate = merged(out[-1], part)
            if candidate.tokens <= cap:
                notes.append(f"plan: joined small parts {out[-1].title!r} and {part.title!r}")
                out[-1] = candidate
                continue
        out.append(part)
    return out


def _split_over_cap(
    parts: list[Part], blocks: list[Block], cap: int, notes: list[str]
) -> list[Part]:
    out: list[Part] = []
    for part in parts:
        if part.tokens <= cap:
            out.append(part)
            continue
        groups = pack_blocks(blocks[part.start : part.end], cap)
        notes.append(f"plan: part {part.title!r} ({part.tokens} tok) split into {len(groups)}")
        for n, group in enumerate(groups, 1):
            title = f"{part.title} ({n}/{len(groups)})"
            text = join_blocks(group)
            out.append(Part(title, group[0].index, group[-1].index + 1, estimate_tokens(text)))
    return out


def to_chunk(index: int, part: Part, blocks: list[Block]) -> Chunk:
    """The part's text; its heading is the headings enclosing the part, then its title."""
    first = blocks[part.start]
    enclosing = first.trail[:-1] if first.heading else first.trail
    trail = enclosing
    if not enclosing or enclosing[-1].casefold() != part.title.casefold():
        trail = (*enclosing, part.title)
    return Chunk(index, trail, join_blocks(blocks[part.start : part.end]).strip())


# --- planning -----------------------------------------------------------------------------------


def _fallback(markdown: str, cap: int, report: PlanReport) -> PlanReport:
    chunks = chunk_markdown(markdown, cap)
    report.chunks = chunks
    report.parts = [Part(c.heading or "(start)", -1, -1, estimate_tokens(c.text)) for c in chunks]
    report.planned = False
    return report


def plan_chunks(
    llm: StructuredModel,
    markdown: str,
    *,
    doc_title: str,
    max_tokens: int = CHUNK_TOKENS,
) -> PlanReport:
    """Chunks for extraction. A document that fits one chunk is returned as is; a larger one is
    split where the model sees topic changes (one call per outline window)."""
    single = chunk_markdown(markdown, max_tokens)
    if len(single) <= 1:
        parts = [Part(c.heading or "(start)", -1, -1, estimate_tokens(c.text)) for c in single]
        return PlanReport(parts, single, planned=False)

    blocks = split_blocks(markdown, max_tokens)
    report = PlanReport([], [], planned=True)
    spans = windows(blocks)
    starts: list[tuple[int, str]] = []
    failed = 0
    clock = time.perf_counter()
    for k, (lo, hi) in enumerate(spans):
        # the part of this window that is kept: from the middle of the overlap with the previous
        # window to the middle of the overlap with the next
        keep_lo = 0 if k == 0 else spans[k - 1][1] - (spans[k - 1][1] - lo) // 2
        keep_hi = len(blocks) if k == len(spans) - 1 else hi - (hi - spans[k + 1][0]) // 2
        messages = build_messages(
            blocks, lo, hi, doc_title=doc_title, total=len(blocks), cap=max_tokens
        )
        report.calls += 1
        window_starts = None
        try:
            plan = llm.structured(
                messages, SplitPlan, temperature=PLAN_TEMPERATURE, max_tokens=PLAN_MAX_TOKENS
            )
            window_starts = repair_starts(plan, blocks, lo, hi, report.notes)
            if window_starts is None:
                report.notes.append(f"plan: unusable plan for blocks {lo}-{hi - 1}")
        except StructuredOutputError as err:
            log.warning("ingest split plan failed for blocks %d-%d: %s", lo, hi - 1, err)
            report.notes.append(f"plan: the model's plan for blocks {lo}-{hi - 1} failed ({err})")
        if window_starts is None:
            failed += 1
            window_starts = _heading_starts(blocks, keep_lo, keep_hi, max_tokens)
        kept = [(i, t) for i, t in window_starts if keep_lo <= i < keep_hi]
        if k > 0 and kept and kept[0][0] == lo:
            kept = kept[1:]  # the window's first block is not a topic change
        starts += kept
    report.seconds = time.perf_counter() - clock

    if failed == len(spans):
        report.notes.append("plan: split at headings instead")
        return _fallback(markdown, max_tokens, report)
    if not starts or starts[0][0] != 0:
        starts.insert(0, (0, _title("", blocks[0])))
    parts = _parts(_keep_headings(starts, blocks), blocks)
    parts = _split_over_cap(parts, blocks, max_tokens, report.notes)
    parts = _merge_tiny(parts, blocks, max_tokens, report.notes)
    report.parts = parts
    report.chunks = [to_chunk(i, p, blocks) for i, p in enumerate(parts)]
    return report
