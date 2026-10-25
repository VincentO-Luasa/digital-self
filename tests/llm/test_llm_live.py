"""Smoke test against the real local model server. Run with ``uv run pytest -m llm``."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import BaseModel

from brain.config import load_settings
from brain.llm import LLM

pytestmark = pytest.mark.llm


class Colour(BaseModel):
    reasoning: str
    answer: str


@pytest.fixture(scope="module")
def llm():
    client = LLM(load_settings().llm)
    if not client.health():
        client.close()
        pytest.skip("local model server is not running")
    yield client
    client.close()


def test_embed(llm):
    docs = llm.embed(["The cat sat on the mat.", "Quarterly revenue grew by four percent."])
    query = llm.embed(["Where did the cat sit?"], query=True)
    assert docs.shape == (2, 1024) and docs.dtype == np.float32
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-3)
    sims = docs @ query[0]
    assert sims[0] > sims[1]
    assert llm.embed_dim == 1024


def test_rerank(llm):
    docs = ["Bananas are yellow.", "Paris is the capital of France."]
    ranked = llm.rerank("What is the capital of France?", docs, top_n=2)
    assert ranked[0][0] == 1
    assert ranked[0][1] > ranked[1][1]


def test_structured(llm):
    out = llm.structured(
        [
            {
                "role": "system",
                "content": "Reply in JSON. reasoning: one short sentence; answer: one word.",
            },
            {"role": "user", "content": "Which is a primary colour: red or teal?"},
        ],
        Colour,
    )
    assert out.reasoning
    assert "red" in out.answer.casefold()


def test_chat(llm):
    text = llm.chat([{"role": "user", "content": "Reply with the single word: pong"}], max_tokens=8)
    assert "pong" in text.casefold()
