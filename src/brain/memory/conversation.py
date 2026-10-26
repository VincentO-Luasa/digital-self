"""How a transcript is shown to the model: numbered turns, the owner's words in full, the
assistant's clipped (facts come from what the owner says; the assistant's turns give context,
such as the question a short "yes" answers)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from brain.vault.models import link_target
from brain.vault.raw import Turn

ASSISTANT_WORDS = 60  # an assistant turn is clipped to this many words
OWNER = "OWNER"
ASSISTANT = "ASSISTANT"
NOTE = "NOTE"
# Interview markers (system turns written by brain.interview): "[interview] region:: … ·
# topic:: … · about:: [[entity]] · mode:: …" before a question, "[interview] privacy:: private
# · …" before a private answer, "[interview] skip · …". Never a source of facts.
MARKER = re.compile(r"^\[interview\]\s*(.*)$", re.S)
FIELD = re.compile(r"^([a-z_]+)::\s*(.*)$")


def marker(turn: Turn) -> dict[str, str] | None:
    """The fields of an interview marker turn (``{"skip": ""}`` for a bare word), else None."""
    if turn.role != "system" or not (m := MARKER.match(turn.text.strip())):
        return None
    fields: dict[str, str] = {}
    for part in m.group(1).split(" · "):
        part = part.strip()
        if f := FIELD.match(part):
            fields[f.group(1)] = f.group(2).strip()
        elif part and " " not in part:
            fields[part] = ""
    return fields


def private_turns(turns: Sequence[Turn], given: Iterable[int] = ()) -> set[int]:
    """Owner turns marked private: ``given`` and the first owner turn after each
    ``privacy:: private`` marker."""
    out = set(given)
    waiting = False
    for turn in turns:
        fields = marker(turn)
        if fields is not None and fields.get("privacy", "").startswith("private"):
            waiting = True
        elif turn.role == "user" and waiting:
            out.add(turn.n)
            waiting = False
    return out


def marker_hints(turns: Sequence[Turn]) -> tuple[list[str], list[str]]:
    """(regions, entity ids) the interview markers among ``turns`` name."""
    regions: list[str] = []
    entities: list[str] = []
    for turn in turns:
        fields = marker(turn) or {}
        if (region := fields.get("region")) and region not in regions:
            regions.append(region)
        target = link_target(fields.get("about", "")) or fields.get("about", "")
        if target and target not in entities:
            entities.append(target)
    return regions, entities


def _note(fields: dict[str, str]) -> str | None:
    if fields.get("privacy", "").startswith("private"):
        return "the owner's next answer is private"
    if "skip" in fields:
        return "the owner skipped the question"
    parts = []
    if topic := fields.get("topic"):
        parts.append(f"topic {topic}")
    if region := fields.get("region"):
        parts.append(f"region {region}")
    if about := fields.get("about"):
        parts.append(f"about {link_target(about) or about}")
    return "next question: " + ", ".join(parts) if parts else None


def _clip(text: str, words: int) -> str:
    parts = text.split()
    return " ".join(parts[:words]) + (" …" if len(parts) > words else "")


def render_turn(turn: Turn, *, assistant_words: int = ASSISTANT_WORDS) -> str | None:
    """One line; None for a system turn that is not an interview marker."""
    if turn.role == "system":
        fields = marker(turn)
        note = _note(fields) if fields is not None else None
        return f"[{turn.n}] {NOTE}: {note}" if note else None
    if turn.role == "user":
        return f"[{turn.n}] {OWNER}: {' '.join(turn.text.split())}"
    return f"[{turn.n}] {ASSISTANT}: {_clip(turn.text, assistant_words)}"


def render(turns: Sequence[Turn], *, assistant_words: int = ASSISTANT_WORDS) -> str:
    """Numbered lines: the owner in full, the assistant clipped, interview markers as NOTEs."""
    lines = (render_turn(t, assistant_words=assistant_words) for t in turns)
    return "\n".join(line for line in lines if line)


def render_plain(turns: Sequence[Turn], *, assistant_words: int = ASSISTANT_WORDS) -> str:
    """Like ``render`` without the turn numbers (the input of typed decisions)."""
    return "\n".join(
        line.split("] ", 1)[1]
        for line in render(turns, assistant_words=assistant_words).split("\n")
        if line
    )


def owner_lines(turns: Sequence[Turn]) -> str:
    """What the owner said, one line per turn."""
    return "\n".join(f"{OWNER}: {' '.join(t.text.split())}" for t in turns if t.role == "user")


def in_range(turns: Sequence[Turn], first: int, last: int) -> list[Turn]:
    return [t for t in turns if first <= t.n <= last]


def owner_turns(turns: Sequence[Turn]) -> list[Turn]:
    return [t for t in turns if t.role == "user" and t.text.strip()]
