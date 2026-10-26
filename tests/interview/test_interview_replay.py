"""Replay the recorded first questions (and one answer) on the fictional vault, no server.

The recorded requests must equal what the session sends today. If this fails after an intended
change of a prompt or the plan, re-record with
``BRAIN_RECORD_FIXTURES=1 uv run pytest -m llm tests/interview/test_interview_live.py``.
"""

from __future__ import annotations

import json

import pytest
from interview_helpers import LATER, Clock, at  # type: ignore[import-not-found]
from interview_replay import FIXTURE_DIR, REGIONS, REPLY  # type: ignore[import-not-found]

from brain.config import Settings
from brain.interview.planner import build_plan
from brain.interview.session import InterviewSession, one_question
from brain.llm.fake import FakeLLM
from brain.llm.recorder import load_exchange


@pytest.mark.parametrize("region", REGIONS)
def test_recorded_interview_turns_replay(env_factory, region: str) -> None:
    # <region>-<step>-<kind>-<digest>.json (dots → dashes): sorted by name = call order
    paths = sorted(FIXTURE_DIR.glob(f"{region.replace('.', '-')}-*.json"))
    assert paths, "no recorded interview turns: record them with BRAIN_RECORD_FIXTURES=1"
    exchanges = [load_exchange(p) for p in paths]
    env = env_factory(LATER)
    report = env.report(LATER)
    llm = FakeLLM.from_fixtures(paths)
    session = InterviewSession(
        env.vault,
        llm,
        report,
        region,
        build_plan(report, region),
        Settings(vault=env.vault.root),
        clock=Clock(at(LATER)),
    )
    replies = [session.opening().message]
    if region in REPLY:
        replies.append(session.reply(REPLY[region]).message)

    sent = [c for c in llm.calls if c[0] in ("structured", "decide")]
    assert [kind for kind, _ in sent] == [e.kind for e in exchanges]
    for (kind, call), exchange in zip(sent, exchanges, strict=True):
        if kind == "structured":
            assert call["messages"] == exchange.request["messages"]
        else:
            assert call["text"] == exchange.request["input"]
            names = [q["name"] for q in exchange.request["questions"]]
            assert [q.name for q in call["questions"]] == names
    recorded = [json.loads(e.content)["message"] for e in exchanges if e.kind == "structured"]
    assert replies == [one_question(m) for m in recorded]
    assert not any(what.startswith("fallback") for _, what in session.trace)
