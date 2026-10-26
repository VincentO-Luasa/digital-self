"""Shared by the live recording and its replay: one fictional chat, fixed dates."""

from __future__ import annotations

from datetime import date, datetime

FIXTURE_DIR_NAME = "session"
STEM = "chat-"
TODAY = date(2026, 10, 27)
AT = datetime(2026, 10, 27, 19, 5)
TURNS = [
    (
        "user",
        "Evening! Long day. The Atlas go-live got moved to December 1st, which is a relief honestly.",
    ),
    (
        "assistant",
        "That sounds like a relief. Last time you said November 15 felt unrealistic. What changed?",
    ),
    (
        "user",
        "The warehouse team needed more time for the parallel run. Also, small correction: I joined Northwind in January 2023, not February.",
    ),
    ("assistant", "Thanks, noted. Anything else going on?"),
    (
        "user",
        "Sam just started a new job at Contoso as a product manager. We're celebrating on Saturday.",
    ),
    ("assistant", "Congratulations to Sam! Enjoy Saturday."),
]


def ordered_recorder(directory, write: bool):
    """A recorder that numbers the saved exchanges (replay keeps the call order) and keeps each
    call's token usage."""
    from typing import Any

    from brain.llm.recorder import Recorder

    class OrderedRecorder(Recorder):
        def __init__(self) -> None:
            super().__init__(directory)
            self.n = 0
            self.usage: list[tuple[str, dict[str, Any]]] = []

        def save(self, kind: str, request: dict[str, Any], response: dict[str, Any]):
            self.n += 1
            self.usage.append((kind, response.get("usage") or {}))
            if not write:
                return self.directory
            self.prefix = f"{STEM}{self.n:02d}-"
            return super().save(kind, request, response)

    return OrderedRecorder()
