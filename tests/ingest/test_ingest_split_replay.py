"""Replay the recorded split plan of the long fictional page (no model server needed).

The recorded request must equal what ``plan_chunks`` sends today. If this fails after an intended
prompt or outline change, re-record with
``BRAIN_RECORD_FIXTURES=1 uv run pytest -m llm tests/ingest/test_ingest_split_live.py``.
"""

from __future__ import annotations

from pathlib import Path

from conftest import LONG_DOC, split_problems  # type: ignore[import-not-found]

from brain.ingest.chunking import CHUNK_TOKENS
from brain.ingest.planning import PLAN_TEMPERATURE, SplitPlan, plan_chunks
from brain.ingest.readers import load_document
from brain.llm.client import response_format
from brain.llm.fake import FakeLLM
from brain.llm.recorder import load_exchange

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "llm" / "ingest" / "split"


def fixtures() -> list[Path]:
    paths = sorted(FIXTURE_DIR.glob("alex-rivera-long-*.json"))
    assert paths, "no recorded split plan: record it with BRAIN_RECORD_FIXTURES=1"
    return paths


def test_recorded_plan_request_matches() -> None:
    doc = load_document(LONG_DOC)
    fake = FakeLLM.from_fixtures(fixtures())
    plan_chunks(fake, doc.text, doc_title=doc.title)
    recorded = [load_exchange(p).request for p in fixtures()]
    assert [c[1]["messages"] for c in fake.calls] == [r["messages"] for r in recorded]
    assert all(r["response_format"] == response_format(SplitPlan) for r in recorded)
    assert all(r["temperature"] == PLAN_TEMPERATURE for r in recorded)
    assert all(
        r["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False for r in recorded
    )


def test_replayed_plan_is_coherent() -> None:
    doc = load_document(LONG_DOC)
    report = plan_chunks(FakeLLM.from_fixtures(fixtures()), doc.text, doc_title=doc.title)
    assert split_problems(report, doc.text, CHUNK_TOKENS) == []
    assert len(report.parts) >= 4 and not any("split into" in n for n in report.notes)
