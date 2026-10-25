"""End-to-end smoke test of the ingest evaluation on one small document with the real local
model. ``uv run pytest -m llm tests/eval``. Uses the shared extraction cache, like the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.config import load_settings
from brain.eval.ingest_eval import default_cache_dir, load_gold_set, report_json, run_doc
from brain.llm import LLM
from brain.llm.cache import StructuredCache

pytestmark = pytest.mark.llm


def test_eval_one_document_live(tmp_path: Path, git_env: None) -> None:
    client = LLM(load_settings().llm)
    if not client.health():
        client.close()
        pytest.skip("local model server is not running")
    try:
        (gold,) = load_gold_set(only=["speaker"])
        score = run_doc(gold, StructuredCache(client, default_cache_dir()), tmp_path)
    finally:
        client.close()
    data = report_json([score], {})
    print(json.dumps(data["docs"][0]["metrics"], indent=2))
    assert not score.error
    c = score.counts()
    assert c["extracted"] > 0
    assert c["required_matched"] >= c["required"] // 2
