"""Helpers for the lint tests: a model that replays recorded decisions by request."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from brain.llm.decide import Decision, Question, request_payload
from brain.llm.recorder import load_exchange

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "lint" / "llm"
REPLAY_SEED = 1


def _key(text: str, questions: Sequence[dict[str, Any]]) -> str:
    return json.dumps([text, list(questions)], sort_keys=True)


class Replay:
    """``decider`` for FakeLLM: the recorded ``/v1/decisions`` response for the same input and
    questions (raises LookupError for an unrecorded request)."""

    def __init__(self, directory: Path = FIXTURES) -> None:
        self.table: dict[str, dict[str, Any]] = {}
        for path in sorted(directory.glob("*.json")):
            ex = load_exchange(path)
            if ex.kind == "decide":
                self.table[_key(ex.request["input"], ex.request["questions"])] = ex.response
        self.missing: list[str] = []

    def __call__(self, text: str, questions: Sequence[Question]) -> dict[str, Any] | list[Decision]:
        payload = request_payload("any", text, questions)
        key = _key(payload["input"], payload["questions"])
        if key not in self.table:
            self.missing.append(text[:80])
            raise LookupError("no recorded decision for this request")
        return self.table[key]


Answer = dict[str, float] | Exception


def decision(name: str, probs: dict[str, float], *, fallback: bool = False) -> Decision:
    choice = max(probs, key=lambda k: probs[k])
    return Decision(
        name=name,
        type="choice",
        choice=choice,
        probabilities=probs,
        confidence=probs[choice],
        fallback=fallback,
    )


class Script:
    """``decider`` for FakeLLM: per question name, a function of (input, question) giving the
    probabilities (or None: no answer). Records what was asked."""

    def __init__(self, **answers: Any) -> None:
        self.answers = answers
        self.asked: list[tuple[str, str]] = []
        self.fallback = False

    def __call__(self, text: str, questions: Sequence[Question]) -> list[Decision]:
        out = []
        for q in questions:
            self.asked.append((q.name, text))
            fn = self.answers.get(q.name)
            probs = fn(text, q) if callable(fn) else fn
            if probs is not None:
                out.append(decision(q.name, probs, fallback=self.fallback))
        return out

    def names(self) -> list[str]:
        return [name for name, _ in self.asked]
