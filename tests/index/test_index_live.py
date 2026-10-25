"""Reindex + search the fake vault with the real embedder and reranker (``pytest -m llm``)."""

from __future__ import annotations

import pytest

from brain.config import load_settings
from brain.index.reindex import reindex
from brain.index.search import search
from brain.llm import LLM

pytestmark = pytest.mark.llm


@pytest.fixture
def real_llm():
    client = LLM(load_settings().llm)
    if not client.health():
        client.close()
        pytest.skip("local model server is not running")
    yield client
    client.close()


def test_reindex_and_search_with_real_models(vault, db, real_llm, private_regions):
    stats = reindex(vault, db, real_llm, private_regions=private_regions)
    assert stats.embedded == stats.items
    assert db.execute("SELECT dim FROM meta").fetchone()[0] == 1024

    hits = search(db, real_llm, "Which instrument does Sam play?", k=3)
    assert hits[0].id == "f-sam001"
    assert hits[0].rerank is not None

    public = search(db, real_llm, "Which instrument does Sam play?", include_private=False)
    assert all(h.privacy == "normal" for h in public)

    past = search(db, real_llm, "Where did Alex live?", k=3, kinds=["fact"], as_of="2020")
    assert past[0].id == "f-city00"
