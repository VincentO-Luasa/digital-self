"""``process_session`` on a fictional chat with the real local model. ``uv run pytest -m llm``.

Set ``BRAIN_RECORD_FIXTURES=1`` to (re-)record ``tests/fixtures/llm/memory/session/`` (needed
whenever a memory prompt, schema or decision question changes); the replay test then checks
that today's requests equal the recorded ones.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from live import (  # type: ignore[import-not-found]
    AT,
    FIXTURE_DIR_NAME,
    STEM,
    TODAY,
    TURNS,
    ordered_recorder,
)
from memtools import write_transcript  # type: ignore[import-not-found]

from brain.config import load_settings
from brain.llm import LLM
from brain.memory import extract, process_session, resolve
from brain.vault.store import Vault

pytestmark = pytest.mark.llm

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "llm" / "memory" / FIXTURE_DIR_NAME


def test_live_session(mvault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    # embeddings are not recorded: candidates by embedding and paraphrase merging stay off so
    # that the replay asks exactly the recorded questions
    monkeypatch.setattr(resolve, "EMBEDDINGS", False)
    monkeypatch.setattr(extract, "DEDUP", False)
    write = bool(os.environ.get("BRAIN_RECORD_FIXTURES"))
    if write:
        for old in FIXTURE_DIR.glob(f"{STEM}[0-9]*.json"):
            old.unlink()
    llm = LLM(load_settings().llm, recorder=ordered_recorder(FIXTURE_DIR, write))
    if not llm.health():
        llm.close()
        pytest.skip("local model server is not running")
    transcript = write_transcript(mvault, TURNS, at=AT)
    start = time.perf_counter()
    try:
        report = process_session(
            mvault, llm, transcript, channel="chat", today=TODAY, refresh=False
        )
    finally:
        llm.close()
    print(f"\n{report.summary()} in {time.perf_counter() - start:.1f}s")
    for note in report.notes:
        print("  ·", note)
    for ep in report.episodes:
        print(f"{ep.id} imp {ep.importance} {ep.regions} {ep.seconds:.1f}s: {ep.gist}")
        for note in ep.notes:
            print("    ·", note)
    for kind, usage in llm.recorder.usage:  # type: ignore[union-attr]
        print(f"  {kind}: {usage}")
    assert report.status == "processed"
    assert 1 <= len(report.episodes) <= 3
    assert report.total("superseded") >= 1
    # only the owner's own correction may be applied without review
    for ep in report.episodes:
        assert ep.result is not None
        assert set(ep.result.corrected) <= {"f-k3x7q2"}
