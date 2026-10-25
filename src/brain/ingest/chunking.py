"""Split a markdown document into heading-bounded chunks of about ``max_tokens`` tokens, or into
ordered blocks (headings, paragraphs, lists, tables, code) for a model-planned split
(``brain.ingest.planning``)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from brain.tokens import estimate_tokens

CHUNK_TOKENS = 500  # a dense CV excerpt yields ~8x its size in extraction JSON
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Section:
    trail: tuple[str, ...]  # heading path, outermost first
    text: str  # includes its own heading line


@dataclass(frozen=True)
class Chunk:
    index: int
    trail: tuple[str, ...]  # heading path of the chunk's first section
    text: str

    @property
    def heading(self) -> str:
        return " > ".join(self.trail)


@dataclass(frozen=True)
class Block:
    index: int
    trail: tuple[str, ...]  # heading path; a heading block's trail ends with itself
    text: str
    heading: bool = False

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


def split_blocks(markdown: str, max_tokens: int = CHUNK_TOKENS) -> list[Block]:
    """Blank-line separated blocks; a heading line is a block of its own and a fenced code block
    stays whole. A block larger than ``max_tokens`` is cut into paragraph/line pieces."""
    raw: list[tuple[tuple[str, ...], str, bool]] = []
    stack: list[tuple[int, str]] = []
    trail: tuple[str, ...] = ()
    lines: list[str] = []
    in_fence = False

    def flush() -> None:
        text = "\n".join(lines).strip("\n")
        lines.clear()
        if text.strip():
            raw.append((trail, text, False))

    for line in markdown.split("\n"):
        if FENCE.match(line):
            if not in_fence:
                flush()
            in_fence = not in_fence
            lines.append(line)
            if not in_fence:
                flush()
            continue
        if in_fence:
            lines.append(line)
            continue
        m = HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            stack = [(lvl, name) for lvl, name in stack if lvl < level] + [(level, m.group(2))]
            trail = tuple(name for _, name in stack)
            raw.append((trail, line.strip(), True))
        elif not line.strip():
            flush()
        else:
            lines.append(line)
    flush()

    blocks: list[Block] = []
    for block_trail, text, is_heading in raw:
        pieces = [text] if estimate_tokens(text) <= max_tokens else _split_large(text, max_tokens)
        for piece in pieces:
            if piece.strip():
                blocks.append(Block(len(blocks), block_trail, piece.strip("\n"), is_heading))
    return blocks


def pack_blocks(blocks: list[Block], max_tokens: int = CHUNK_TOKENS) -> list[list[Block]]:
    """Greedily pack consecutive blocks into groups of at most ``max_tokens``, cutting at
    headings where possible: whole heading-led sections are packed first, an oversized section
    block by block."""
    sections: list[list[Block]] = []
    for block in blocks:
        prev_heading = bool(sections) and sections[-1][-1].heading
        if not sections or (block.heading and not prev_heading):
            sections.append([block])  # consecutive headings stay together
        else:
            sections[-1].append(block)
    groups: list[list[Block]] = []
    current: list[Block] = []

    def size(group: list[Block]) -> int:
        return estimate_tokens(join_blocks(group))

    for section in sections:
        if size(current + section) <= max_tokens:
            current += section
            continue
        if current:
            groups.append(current)
            current = []
        if size(section) <= max_tokens:
            current = list(section)
            continue
        for block in section:
            if current and size([*current, block]) > max_tokens:
                if all(b.heading for b in current):
                    current.append(block)  # never leave a heading on its own
                    continue
                groups.append(current)
                current = []
            current.append(block)
    if current:
        groups.append(current)
    return groups


def join_blocks(blocks: list[Block]) -> str:
    return "\n\n".join(b.text for b in blocks)


def split_sections(markdown: str) -> list[Section]:
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    lines: list[str] = []
    trail: tuple[str, ...] = ()
    in_fence = False

    def flush() -> None:
        text = "\n".join(lines).strip("\n")
        if not text.strip():
            return
        prev = sections[-1] if sections else None
        if prev and "\n" not in prev.text.strip() and HEADING.match(prev.text.strip()):
            # a bare heading belongs with the section that follows it
            sections[-1] = Section(prev.trail, f"{prev.text}\n\n{text}")
        else:
            sections.append(Section(trail, text))

    for line in markdown.split("\n"):
        if FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            stack = [(lvl, name) for lvl, name in stack if lvl < level] + [(level, m.group(2))]
            trail = tuple(name for _, name in stack)
            lines = [line]
        else:
            lines.append(line)
    flush()
    return sections


def _split_large(text: str, max_tokens: int) -> list[str]:
    """Paragraph-wise split of an oversized section; giant paragraphs are cut by lines, then
    hard-wrapped by characters."""
    pieces: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        if estimate_tokens(para) <= max_tokens:
            pieces.append(para)
            continue
        for line in para.split("\n"):
            limit = int(max_tokens * 3.5)
            pieces += [line[i : i + limit] for i in range(0, len(line), limit)] or [""]
    out: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if current and estimate_tokens(candidate) > max_tokens:
            out.append(current)
            current = piece
        else:
            current = candidate
    if current.strip():
        out.append(current)
    return out


def halve(chunk: Chunk) -> list[Chunk]:
    """Split a chunk in two at a paragraph (or line) boundary; [] if it can't be split."""
    parts = _split_large(chunk.text, max(1, estimate_tokens(chunk.text) // 2 + 1))
    if len(parts) < 2:
        return []
    return [Chunk(chunk.index, chunk.trail, part.strip()) for part in parts if part.strip()]


def chunk_markdown(markdown: str, max_tokens: int = CHUNK_TOKENS) -> list[Chunk]:
    """Greedily pack consecutive sections; a chunk always starts at a section boundary unless a
    single section is larger than ``max_tokens``."""
    chunks: list[tuple[tuple[str, ...], str]] = []
    trail: tuple[str, ...] = ()
    current = ""
    for section in split_sections(markdown):
        candidate = f"{current}\n\n{section.text}" if current else section.text
        if estimate_tokens(candidate) <= max_tokens:
            if not current:
                trail = section.trail
            current = candidate
            continue
        if current:
            chunks.append((trail, current))
            current = ""
        if estimate_tokens(section.text) <= max_tokens:
            trail, current = section.trail, section.text
        else:
            chunks += [(section.trail, part) for part in _split_large(section.text, max_tokens)]
    if current:
        chunks.append((trail, current))
    return [Chunk(i, t, text.strip()) for i, (t, text) in enumerate(chunks) if text.strip()]
