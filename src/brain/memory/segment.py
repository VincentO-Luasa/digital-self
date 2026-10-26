"""Segment a finished conversation into topic episodes (docs/architecture.md, Flow 3).

1. **Gate** (``gate``): one yes/no typed decision on what the owner said: does the owner say
   anything about their own life? Under ``SKIP_BELOW`` the conversation is small talk and yields
   no episode (no segmentation call). No answer, or an error, lets it through.
2. **Segment** (``segment``): one structured call (prompt ``memory_segment``, schema
   ``Segmentation``; regions are an enum of the life map's ids). The model labels each episode's
   importance signals; the 1–10 score is computed in code (``importance``).
3. **Repair** (``clean_episodes``): turn ranges are clamped, sorted and made disjoint; an
   episode without an owner turn is dropped; unknown regions dropped; the gist is capped at
   ``GIST_WORDS``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from functools import cache
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, create_model

from brain.llm.decide import Decision, PredicateQuestion, Question
from brain.memory.conversation import in_range, owner_lines, owner_turns, render
from brain.prompts import load_prompt
from brain.tokens import estimate_tokens
from brain.vault.lifemap import LifeMap
from brain.vault.raw import Turn

T = TypeVar("T", bound=BaseModel)

SKIP_BELOW = 0.15  # gate: no episode when p(the owner says anything personal) is under this
GATE_TOKENS = 1500  # the gate sees the owner's lines up to about this many tokens
GIST_WORDS = 22  # a gist is capped at this many words (≤ 30 tokens, docs/vault-schema.md)
MAX_TOKENS = 3072

Emotion = Literal["none", "mild", "strong"]
GoalLink = Literal["none", "related", "central"]
Change = Literal["none", "minor", "major"]
MentionType = Literal["person", "org", "project", "place", "event"]

# Composite importance (1–10) from the labels: a neutral episode scores BASE; feelings, goals,
# life changes, a date ahead and an explicit "remember this" raise it (docs/decisions.md).
BASE = 2
EMOTION = {"none": 0, "mild": 1, "strong": 3}
GOAL = {"none": 0, "related": 1, "central": 2}
CHANGE = {"none": 0, "minor": 1, "major": 3}
FUTURE = 1
REMEMBER = 3

GATE = PredicateQuestion(
    name="personal",
    instructions=(
        "Does the owner say anything about their own life: their work, projects, plans, "
        "people they know, places, past, feelings, preferences or habits?"
    ),
)


class Mention(BaseModel):
    name: str
    type: MentionType


class SegmentedEpisode(BaseModel):
    first_turn: int
    last_turn: int
    topic: str
    regions: list[str]
    mentions: list[Mention]
    gist: str
    summary: str
    emotion: Emotion
    goal: GoalLink
    future: bool
    remember: bool
    change: Change


class Segmentation(BaseModel):
    reasoning: str
    episodes: list[SegmentedEpisode]


@cache
def segmentation_schema(regions: tuple[str, ...]) -> type[Segmentation]:
    """``Segmentation`` with ``regions`` an enum of the life map's region ids (enforced)."""
    if not regions:
        return Segmentation
    region = Literal[regions]  # type: ignore[valid-type]
    episode = create_model(
        "SegmentedEpisode", __base__=SegmentedEpisode, regions=(list[region], ...)
    )
    return create_model("Segmentation", __base__=Segmentation, episodes=(list[episode], ...))


class SegmentModel(Protocol):
    def structured(
        self,
        messages: Sequence[Mapping[str, Any]],
        schema: type[T],
        *,
        temperature: float | None = None,
        max_tokens: int = 2048,
    ) -> T: ...

    def decide(self, text: str, questions: Sequence[Question]) -> list[Decision]: ...


@dataclass
class Segment:
    """One episode as segmented and repaired (not yet an id or a file)."""

    first: int
    last: int
    topic: str
    regions: list[str]
    mentions: list[Mention]
    gist: str
    summary: str
    importance: int
    labels: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentReport:
    segments: list[Segment] = field(default_factory=list)
    p_personal: float | None = None
    skipped: bool = False  # the gate found nothing personal: no episode
    gate_seconds: float = 0.0
    seconds: float = 0.0  # the segmentation call
    notes: list[str] = field(default_factory=list)


def importance(ep: SegmentedEpisode) -> int:
    score = BASE + EMOTION[ep.emotion] + GOAL[ep.goal] + CHANGE[ep.change]
    score += (FUTURE if ep.future else 0) + (REMEMBER if ep.remember else 0)
    return max(1, min(10, score))


def gate_text(turns: Sequence[Turn], max_tokens: int = GATE_TOKENS) -> str:
    """The owner's lines, newest dropped first when over the budget."""
    lines = owner_lines(turns).split("\n")
    while len(lines) > 1 and estimate_tokens("\n".join(lines)) > max_tokens:
        lines.pop()
    return "\n".join(lines)


def gate(llm: SegmentModel, turns: Sequence[Turn]) -> tuple[Decision | None, float, str]:
    """``(decision, seconds, note)``; the decision is None when the call failed."""
    start = time.perf_counter()
    try:
        [decision] = llm.decide(gate_text(turns), [GATE])
    except Exception as err:  # noqa: BLE001 - a failed gate never blocks memory
        return None, time.perf_counter() - start, f"gate: no decision ({err}); segmented"
    seconds = time.perf_counter() - start
    p = decision.p_yes
    if p is None:
        return decision, seconds, "gate: no answer; segmented"
    return decision, seconds, f"gate: p(personal)={p:.3f}"


def build_messages(
    turns: Sequence[Turn],
    *,
    lifemap: LifeMap,
    owner: str,
    channel: str,
    today: date,
    regions_hint: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """The user message starts with what is stable for a vault (regions), then the session."""
    regions = "\n".join(
        f"- {rid}: {region.label or rid}" for rid, region in lifemap.regions.items()
    )
    hint = ""
    if regions_hint:
        hint = f"The owner was interviewed about: {', '.join(regions_hint)}.\n"
    user = (
        f"REGIONS (id: life area):\n{regions}\n\n"
        f"Today is {today.isoformat()}. Owner: {owner}. Channel: {channel}.\n{hint}\n"
        f"CONVERSATION:\n{render(turns)}"
    )
    return [
        {"role": "system", "content": load_prompt("memory_segment")},
        {"role": "user", "content": user},
    ]


def clean_episodes(
    raw: Sequence[SegmentedEpisode], turns: Sequence[Turn], lifemap: LifeMap, notes: list[str]
) -> list[Segment]:
    numbers = [t.n for t in turns]
    lo, hi = (min(numbers), max(numbers)) if numbers else (1, 0)
    out: list[Segment] = []
    last_end = lo - 1
    for ep in sorted(raw, key=lambda e: (e.first_turn, e.last_turn)):
        first, last = max(ep.first_turn, lo, last_end + 1), min(ep.last_turn, hi)
        if (first, last) != (ep.first_turn, ep.last_turn):
            notes.append(
                f"segment: {ep.topic!r} turns {ep.first_turn}-{ep.last_turn} → {first}-{last}"
            )
        if first > last or not owner_turns(in_range(turns, first, last)):
            notes.append(f"segment: dropped {ep.topic!r}: no owner turn in {first}-{last}")
            continue
        regions = [r for r in dict.fromkeys(ep.regions) if lifemap.region(r)]
        if dropped := [r for r in ep.regions if r not in regions]:
            notes.append(f"segment: {ep.topic!r}: unknown regions {dropped} dropped")
        words = " ".join(ep.gist.split()).split(" ")
        gist = " ".join(words[:GIST_WORDS]).rstrip(",;:") if words[0] else ep.topic
        if len(words) > GIST_WORDS:
            notes.append(f"segment: {ep.topic!r}: gist cut to {GIST_WORDS} words")
        labels = ep.model_dump(include={"emotion", "goal", "future", "remember", "change"})
        out.append(
            Segment(
                first=first,
                last=last,
                topic=" ".join(ep.topic.split()) or "conversation",
                regions=regions,
                mentions=[m for m in ep.mentions if m.name.strip()],
                gist=gist,
                summary=" ".join(ep.summary.split()),
                importance=importance(ep),
                labels=labels,
            )
        )
        last_end = last
    return out


def segment(
    llm: SegmentModel,
    turns: Sequence[Turn],
    *,
    lifemap: LifeMap,
    owner: str,
    channel: str,
    today: date,
    regions_hint: Sequence[str] | None = None,
    use_gate: bool = True,
) -> SegmentReport:
    """Gate, then one segmentation call. Raises what ``llm.structured`` raises."""
    report = SegmentReport()
    if not owner_turns(turns):
        report.skipped = True
        report.notes.append("segment: the owner said nothing")
        return report
    if use_gate:
        decision, report.gate_seconds, note = gate(llm, turns)
        report.notes.append(note)
        report.p_personal = decision.p_yes if decision else None
        if report.p_personal is not None and report.p_personal < SKIP_BELOW:
            report.skipped = True
            report.notes.append(f"segment: nothing personal (p < {SKIP_BELOW}): no episode")
            return report
    schema = segmentation_schema(tuple(lifemap.regions))
    messages = build_messages(
        turns,
        lifemap=lifemap,
        owner=owner,
        channel=channel,
        today=today,
        regions_hint=regions_hint,
    )
    start = time.perf_counter()
    result = llm.structured(messages, schema, max_tokens=MAX_TOKENS)
    report.seconds = time.perf_counter() - start
    report.segments = clean_episodes(result.episodes, turns, lifemap, report.notes)
    if not report.segments:
        report.notes.append("segment: no episode")
    return report
