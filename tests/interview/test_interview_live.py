"""First questions from the real local model on the fictional vault. ``uv run pytest -m llm``.

Re-record the replayed fixtures with
``BRAIN_RECORD_FIXTURES=1 uv run pytest -m llm tests/interview/test_interview_live.py``.
"""

from __future__ import annotations

import os
import time

import pytest
from interview_helpers import LATER, Clock, at  # type: ignore[import-not-found]
from interview_replay import (  # type: ignore[import-not-found]
    FIXTURE_DIR,
    REGIONS,
    REPLY,
    SeqRecorder,
)

from brain.config import Settings, load_settings
from brain.interview.planner import build_plan
from brain.interview.session import InterviewSession
from brain.llm import LLM

pytestmark = pytest.mark.llm
RECORD = os.environ.get("BRAIN_RECORD_FIXTURES") == "1"


@pytest.fixture
def client():
    llm = LLM(load_settings().llm)
    if not llm.health():
        llm.close()
        pytest.skip("local model server is not running")
    yield llm
    llm.close()


def test_first_questions_are_grounded_single_and_natural(env_factory, client: LLM) -> None:
    env = env_factory(LATER)
    report = env.report(LATER)
    for region in REGIONS:
        session = InterviewSession(
            env.vault,
            client,
            report,
            region,
            build_plan(report, region),
            Settings(vault=env.vault.root),
            clock=Clock(at(LATER)),
        )
        client.recorder = SeqRecorder(FIXTURE_DIR, prefix=f"{region}-1-") if RECORD else None
        start = time.perf_counter()
        reply = session.opening()
        print(f"\n[{region} {time.perf_counter() - start:.1f}s] {reply.message}")
        assert reply.message and reply.message.count("?") == 1
        assert not any(what.startswith("fallback") for _, what in session.trace), session.trace
        if region in REPLY:
            client.recorder = SeqRecorder(FIXTURE_DIR, prefix=f"{region}-2-") if RECORD else None
            start = time.perf_counter()
            follow = session.reply(REPLY[region])
            print(f"  you: {REPLY[region]}\n  [{time.perf_counter() - start:.1f}s] {follow}")
            assert follow.message.count("?") == 1
            assert session.trace[-1][1] in ("follow_up", "confirm", "ask", "more", "discover")
    client.recorder = None
