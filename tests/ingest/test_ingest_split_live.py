"""Plan the split of the long fictional page with the real local model. ``uv run pytest -m llm``.

Set ``BRAIN_RECORD_FIXTURES=1`` to (re-)record ``tests/fixtures/llm/ingest/split/`` (needed
whenever the split prompt, the outline or the schema changes).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from conftest import LONG_DOC, split_problems  # type: ignore[import-not-found]

from brain.config import load_settings
from brain.ingest.chunking import CHUNK_TOKENS
from brain.ingest.planning import plan_chunks
from brain.ingest.readers import load_document
from brain.llm import LLM
from brain.llm.recorder import Recorder

pytestmark = pytest.mark.llm

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "llm" / "ingest" / "split"


@pytest.fixture
def llm():
    recorder = None
    if os.environ.get("BRAIN_RECORD_FIXTURES"):
        for old in FIXTURE_DIR.glob("*.json"):
            old.unlink()
        recorder = Recorder(FIXTURE_DIR, prefix="alex-rivera-long-")
    client = LLM(load_settings().llm, recorder=recorder)
    if not client.health():
        client.close()
        pytest.skip("local model server is not running")
    yield client
    client.close()


def test_plan_long_page_live(llm: LLM) -> None:
    doc = load_document(LONG_DOC)
    start = time.perf_counter()
    report = plan_chunks(llm, doc.text, doc_title=doc.title)
    print(f"\nplan: {report.calls} call(s), {time.perf_counter() - start:.1f}s")
    for part in report.parts:
        print(f"  blocks {part.start}-{part.end - 1}: {part.title} ({part.tokens} tok)")
    for note in report.notes:
        print(f"  note: {note}")
    assert split_problems(report, doc.text, CHUNK_TOKENS) == []
    assert len(report.parts) >= 4
    # the model split the long section itself: code did not have to cut a part over the cap
    assert not any("split into" in n for n in report.notes)
