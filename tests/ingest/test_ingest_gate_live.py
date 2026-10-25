"""Measure the extraction gate on fictional parts with the real model. ``uv run pytest -m llm``.

Set ``BRAIN_RECORD_FIXTURES=1`` to (re-)record ``tests/fixtures/llm/ingest/gate/`` (needed when
the gate question or its input changes).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from conftest import GATE_PARTS  # type: ignore[import-not-found]

from brain.config import load_settings
from brain.ingest.decisions import SKIP_BELOW, gate
from brain.llm import LLM
from brain.llm.recorder import Recorder

pytestmark = pytest.mark.llm

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "llm" / "ingest" / "gate"


@pytest.fixture
def llm():
    recorder = None
    if os.environ.get("BRAIN_RECORD_FIXTURES"):
        for old in FIXTURE_DIR.glob("*.json"):
            old.unlink()
        recorder = Recorder(FIXTURE_DIR, prefix="gate-")
    client = LLM(load_settings().llm, recorder=recorder)
    if not client.health():
        client.close()
        pytest.skip("local model server is not running")
    yield client
    client.close()


def test_gate_separates_parts_live(llm: LLM) -> None:
    for name, (text, extract) in GATE_PARTS.items():
        start = time.perf_counter()
        verdict = gate(llm, text)
        print(f"\n{name:14s} p={verdict.p_yes:.3f} {time.perf_counter() - start:.2f}s", end="")
        if extract is not None:
            assert verdict.skip is (not extract), name
        else:
            assert not verdict.skip, name
    print(f"\nskip below {SKIP_BELOW}")
