"""Replay the recorded gate decisions on fictional parts (no model server needed).

The recorded request must equal what ``gate`` sends today. If this fails after an intended
change of the question or its input, re-record with
``BRAIN_RECORD_FIXTURES=1 uv run pytest -m llm tests/ingest/test_ingest_gate_live.py``.
"""

from __future__ import annotations

from pathlib import Path

from conftest import GATE_PARTS  # type: ignore[import-not-found]

from brain.ingest.decisions import GATE, compact, gate
from brain.llm.fake import FakeLLM
from brain.llm.recorder import load_exchange

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "llm" / "ingest" / "gate"


def test_recorded_gate_requests_match_and_separate_the_parts() -> None:
    recorded = {
        load_exchange(p).request["input"]: load_exchange(p) for p in FIXTURE_DIR.glob("*.json")
    }
    assert recorded, "no recorded gate decisions: record them with BRAIN_RECORD_FIXTURES=1"
    for name, (text, extract) in GATE_PARTS.items():
        exchange = recorded[compact(text)]  # the request gate() sends today
        assert exchange.request["questions"] == [
            GATE.model_dump(exclude={"type"}) | {"type": "predicate"}
        ]
        verdict = gate(FakeLLM(decisions=[exchange.response]), text)
        if extract is not None:
            assert verdict.skip is (not extract), (name, verdict.p_yes)
        else:
            assert not verdict.skip, (name, verdict.p_yes)
