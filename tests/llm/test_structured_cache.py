"""StructuredCache replays identical structured requests and passes everything else through."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from brain.llm.cache import StructuredCache
from brain.llm.fake import FakeLLM


class Answer(BaseModel):
    reasoning: str
    answer: str


def test_identical_requests_hit_the_cache(tmp_path: Path) -> None:
    fake = FakeLLM(
        structured_responses=[{"reasoning": "r", "answer": "a"}, {"reasoning": "r", "answer": "b"}]
    )
    llm = StructuredCache(fake, tmp_path / "cache")
    q = [{"role": "user", "content": "q"}]
    assert llm.structured(q, Answer).answer == "a"
    assert llm.structured(q, Answer).answer == "a"  # replayed, the model is not called
    assert llm.hits == 1 and len(fake.calls) == 1
    other = [{"role": "user", "content": "another"}]
    assert llm.structured(other, Answer).answer == "b"
    assert llm.embed_model == fake.embed_model  # passthrough


def test_corrupt_entries_are_refetched(tmp_path: Path) -> None:
    fake = FakeLLM(
        structured_responses=[{"reasoning": "r", "answer": "a"}, {"reasoning": "r", "answer": "b"}]
    )
    llm = StructuredCache(fake, tmp_path)
    q = [{"role": "user", "content": "q"}]
    llm.structured(q, Answer)
    for path in tmp_path.glob("Answer-*.json"):
        path.write_text("{not json")
    assert llm.structured(q, Answer).answer == "b"
