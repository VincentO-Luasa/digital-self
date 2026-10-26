"""Helpers for the memory tests (scripted model answers, transcripts). Fictional persona only."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from brain.llm.decide import Decision, Question
from brain.llm.fake import FakeLLM
from brain.vault.raw import Transcript
from brain.vault.store import Vault


def write_transcript(
    vault: Vault, turns: Sequence[tuple[str, str]], *, at: datetime, channel: str = "chat"
) -> Transcript:
    transcript = Transcript.start(vault, channel=channel, now=at)
    for n, (role, text) in enumerate(turns):
        transcript.append(role, text, at=at + timedelta(minutes=n))  # type: ignore[arg-type]
    transcript.commit()
    return transcript


def predicate(name: str, p: float) -> Decision:
    return Decision(
        name=name,
        type="predicate",
        choice="yes" if p >= 0.5 else "no",
        probabilities={"yes": p, "no": round(1 - p, 4)},
        confidence=abs(2 * p - 1),
    )


def choice(name: str, value: str, p: float, others: Sequence[str]) -> Decision:
    rest = (1 - p) / max(len(others), 1)
    probs = {value: p, **{o: rest for o in others if o != value}}
    return Decision(name=name, type="choice", choice=value, probabilities=probs, confidence=p)


class Decider:
    """Answers the memory questions by name; everything else (ingest's checks) is refused,
    which keeps the extractor's choices."""

    def __init__(
        self,
        *,
        personal: float = 0.99,
        correction: float = 0.99,
        who: dict[str, str] | None = None,
    ) -> None:
        self.personal = personal
        self.correction = correction
        self.who = who or {}  # mention → the choice value to pick
        self.asked: list[tuple[str, str]] = []

    def __call__(self, text: str, questions: Sequence[Question]) -> list[Any]:
        out: list[Any] = []
        for q in questions:
            self.asked.append((q.name, text))
            values = [c.value for c in getattr(q, "choices", [])]
            if q.name == "personal":
                out.append(predicate(q.name, self.personal))
            elif q.name == "correction":
                value = "yes" if self.correction >= 0.5 else "no"
                p = self.correction if value == "yes" else 1 - self.correction
                out.append(choice(q.name, value, p, values))
            elif q.name == "who":
                mention = q.instructions.split('"')[1]
                pick = self.who.get(mention, values[-1])
                out.append(choice(q.name, pick, 0.95, values))
            else:
                out.append(Decision(name=q.name, type=q.type, refusal=True))
        return out


def fake(structured: Sequence[Any], decider: Decider | None = None) -> FakeLLM:
    return FakeLLM(structured_responses=structured, decider=decider or Decider())


def segmentation(*episodes: dict[str, Any], reasoning: str = "r") -> dict[str, Any]:
    base = {
        "topic": "t",
        "regions": [],
        "mentions": [],
        "gist": "g",
        "summary": "s",
        "emotion": "none",
        "goal": "none",
        "future": False,
        "remember": False,
        "change": "none",
    }
    return {"reasoning": reasoning, "episodes": [{**base, **e} for e in episodes]}


def extraction(
    facts: Sequence[dict[str, Any]] = (),
    *,
    entities: Sequence[dict[str, Any]] = (),
    observations: Sequence[str] = (),
    intentions: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    base = {"text": "", "start": "", "end": "", "label": "new", "known": ""}
    return {
        "reasoning": "r",
        "entities": [{"aliases": [], "id": "", **e} for e in entities],
        "facts": [{**base, **f} for f in facts],
        "observations": list(observations),
        "intentions": [{"after": "", "about": [], **i} for i in intentions],
    }
