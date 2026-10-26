"""Replay the recorded live session (no model server): the requests must equal the recorded
ones (the prompts are pinned), and the outcome is checked. Re-record after an intended change
with ``BRAIN_RECORD_FIXTURES=1 uv run pytest -m llm tests/memory/test_memory_live.py``."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from live import AT, FIXTURE_DIR_NAME, STEM, TODAY, TURNS  # type: ignore[import-not-found]
from memtools import write_transcript  # type: ignore[import-not-found]

from brain.config import LLMSettings
from brain.llm.decide import request_payload
from brain.llm.fake import FakeLLM
from brain.llm.recorder import load_exchange
from brain.memory import extract, process_session, resolve
from brain.vault.store import Vault

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "llm" / "memory" / FIXTURE_DIR_NAME


@pytest.fixture(autouse=True)
def recorded_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resolve, "EMBEDDINGS", False)
    monkeypatch.setattr(extract, "DEDUP", False)


SEED_IDS = re.compile(r"\^?f-[a-z2-7]{6}")


def masked(value: object, seed: set[str]) -> object:
    """Fact ids the run created are random: mask those not in the seed vault."""
    text = str(value)
    return SEED_IDS.sub(lambda m: m.group(0) if m.group(0).lstrip("^") in seed else "f-NEW", text)


def fixtures() -> list[Path]:
    paths = sorted(FIXTURE_DIR.glob(f"{STEM}[0-9]*.json"))
    assert paths, "no recorded memory fixture: record it with BRAIN_RECORD_FIXTURES=1"
    return paths


def test_recorded_requests_match_the_pinned_prompts(mvault: Vault) -> None:
    paths = fixtures()
    fake = FakeLLM.from_fixtures(paths)
    seed = {f.id for p in mvault.pages() for f in (*p.facts, *p.history)}
    transcript = write_transcript(mvault, TURNS, at=AT)
    report = process_session(mvault, fake, transcript, channel="chat", today=TODAY, refresh=False)
    assert report.status == "processed", report.notes
    exchanges = [load_exchange(p) for p in paths]
    sent = [masked(c[1]["messages"], seed) for c in fake.calls if c[0] == "structured"]
    recorded = [masked(e.request["messages"], seed) for e in exchanges if e.kind == "structured"]
    assert sent == recorded
    asked = [
        request_payload(LLMSettings().chat_model, c[1]["text"], c[1]["questions"])
        for c in fake.calls
        if c[0] == "decide"
    ]
    decided = [e.request for e in exchanges if e.kind == "decide"]
    assert [masked(a["input"], seed) for a in asked] == [masked(r["input"], seed) for r in decided]
    assert [a["questions"] for a in asked] == [r["questions"] for r in decided]


def test_recorded_session_outcome(mvault: Vault) -> None:
    fake = FakeLLM.from_fixtures(fixtures())
    transcript = write_transcript(mvault, TURNS, at=AT)
    report = process_session(mvault, fake, transcript, channel="chat", today=TODAY, refresh=False)
    episodes = [e for e in mvault.episodes() if e.session == transcript.session_id]
    assert 1 <= len(episodes) <= 3 and all(e.status == "extracted" for e in episodes)
    assert all(len(e.gist.split()) <= 22 for e in episodes)
    # the explicit start-date correction was applied, nothing else without review
    assert {f for ep in report.episodes for f in ep.result.corrected} == {"f-k3x7q2"}
    _, old, in_history = mvault.find_fact("f-k3x7q2")
    assert in_history and old.by
    _, new, _ = mvault.find_fact(old.by)
    assert new.valid_from == "2023-01" and new.o == "[[orgs/northwind-labs]]"
    # the project's new deadline is a supersession of a project fact: queued
    assert mvault.find_fact("f-q5r6s7")[2] is False
    assert report.total("queued") >= 1
