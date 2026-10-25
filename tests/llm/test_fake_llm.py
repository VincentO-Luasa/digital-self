from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel

from brain.llm import ChatResult, FakeLLM, StructuredOutputError, ToolCall

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "llm"


class Answer(BaseModel):
    reasoning: str
    answer: str


def test_scripted_chat_and_default():
    fake = FakeLLM(chat_responses=["first", "second"])
    assert fake.chat([{"role": "user", "content": "a"}]) == "first"
    assert "".join(fake.chat([], stream=True)) == "second"
    assert fake.chat([]) == "ok"
    assert [c[0] for c in fake.calls] == ["chat", "chat", "chat"]


def test_chat_with_tools_returns_chat_result():
    call = ChatResult(None, [ToolCall("c1", "lookup", {"q": "Atlas"})], "tool_calls")
    fake = FakeLLM(chat_responses=[call, "plain"])
    assert fake.chat([], tools=[{}]) is call
    assert fake.chat([], tools=[{}]).content == "plain"


def test_structured_accepts_models_dicts_and_json():
    fake = FakeLLM(
        structured_responses=[
            Answer(reasoning="r", answer="a"),
            {"reasoning": "r", "answer": "b"},
            '{"reasoning": "r", "answer": "c"}',
            '{"reasoning": "r"}',
        ]
    )
    assert [fake.structured([], Answer).answer for _ in range(3)] == ["a", "b", "c"]
    with pytest.raises(StructuredOutputError):
        fake.structured([], Answer)
    with pytest.raises(LookupError):
        fake.structured([], Answer)


def test_embeddings_are_deterministic_and_lexical():
    fake = FakeLLM(dim=128)
    a, b, c = fake.embed(["Sam plays the cello", "the cello Sam plays", "Atlas deadline"])
    assert np.allclose(a, b)
    assert a @ c < 0.5
    assert np.allclose(np.linalg.norm(fake.embed(["x", ""]), axis=1), 1.0)
    assert np.allclose(FakeLLM(dim=128).embed(["Sam plays the cello"])[0], a)
    assert fake.embedded == [
        "Sam plays the cello",
        "the cello Sam plays",
        "Atlas deadline",
        "x",
        "",
    ]
    fake.embed(["query text"], query=True)
    assert "query text" not in fake.embedded  # only document embeddings are tracked


def test_lexical_rerank():
    fake = FakeLLM()
    ranked = fake.rerank("atlas deadline", ["lunch plans", "Atlas deadline moved", "atlas"], 2)
    assert ranked == [(1, 1.0), (2, 0.5)]


def test_health_and_dim():
    assert FakeLLM().health() and not FakeLLM(healthy=False).health()
    assert FakeLLM(dim=16).embed_dim == 16


def test_replays_recorded_fixture():
    paths = sorted(FIXTURES.glob("structured-*.json"))
    assert paths, "expected a recorded structured fixture in tests/fixtures/llm/"
    fake = FakeLLM.from_fixtures(paths)
    out = fake.structured([], Answer)
    assert out.reasoning and out.answer
