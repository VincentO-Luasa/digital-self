"""Long episodes are extracted window by window; a cut-off window is halved."""

from __future__ import annotations

from datetime import datetime

import pytest

from brain.llm import FakeLLM, OutputTruncated
from brain.memory.extract import MemoryExtraction, extract_windows, windows
from brain.vault.raw import Turn

AT = datetime(2026, 10, 27, 20, 0)


def exchange(n: int, words: int) -> list[Turn]:
    return [
        Turn(n=2 * n - 1, role="assistant", at=AT, text="What is Atlas Migration about?"),
        Turn(n=2 * n, role="user", at=AT, text=" ".join(["data"] * words)),
    ]


def extraction(name: str, fact: str) -> MemoryExtraction:
    return MemoryExtraction.model_validate(
        {
            "reasoning": fact,
            "entities": [{"name": name, "type": "project", "id": "", "aliases": []}],
            "facts": [
                {
                    "subject": "self",
                    "predicate": "interest",
                    "object": fact,
                    "text": fact,
                    "start": "",
                    "end": "",
                    "label": "new",
                    "known": "",
                }
            ],
            "observations": [],
            "intentions": [],
        }
    )


def test_windows_keep_each_question_with_its_answer() -> None:
    turns = [t for n in range(1, 7) for t in exchange(n, 100)]
    turns.append(Turn(n=13, role="assistant", at=AT, text="Thanks, that is all."))
    parts = windows(turns, words=250)
    assert [len(p) for p in parts] == [6, 7]  # the closing assistant turn joins the last window
    assert all(p[0].role == "assistant" and p[-2].role in ("assistant", "user") for p in parts)
    assert parts[0][-1].role == "user"
    assert [t.n for p in parts for t in p] == [t.n for t in turns]


def test_short_episode_is_one_call() -> None:
    llm = FakeLLM(structured_responses=[extraction("Atlas Migration", "a")])
    turns = exchange(1, 20)
    out = extract_windows(llm, turns, lambda w: [{"role": "user", "content": str(len(w))}])
    assert len(llm.calls) == 1 and out.facts[0].object == "a"


def test_cut_off_window_is_halved_and_merged() -> None:
    turns = [t for n in range(1, 5) for t in exchange(n, 50)]  # one window of 4 exchanges
    llm = FakeLLM(
        structured_responses=[
            OutputTruncated("MemoryExtraction: reply cut off", "{"),
            extraction("Atlas Migration", "first half"),
            extraction("atlas migration", "second half"),
        ]
    )
    seen: list[list[int]] = []

    def build(window):
        seen.append([t.n for t in window])
        return [{"role": "user", "content": "x"}]

    out = extract_windows(llm, turns, build)
    assert seen == [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]]
    assert [f.object for f in out.facts] == ["first half", "second half"]
    assert len(out.entities) == 1


def test_a_single_exchange_that_is_cut_off_raises() -> None:
    llm = FakeLLM(structured_responses=[OutputTruncated("cut off", "{")])
    with pytest.raises(OutputTruncated):
        extract_windows(llm, exchange(1, 20), lambda w: [])
