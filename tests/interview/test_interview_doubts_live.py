"""End-to-end, against the local model: an interview after ``brain lint`` on a fictional vault
(``doubts_scenario``), with a scripted owner. ``uv run pytest -m llm
tests/interview/test_interview_doubts_live.py -s``.

Per sample: questions asked, repeated questions (something the owner already told), declined
information asked again, whether the "tell me more" got the vault's descriptions, lint doubts
settled correctly, wrong graph changes (must be 0), seconds per turn and per linking decision.
``BRAIN_INTERVIEW_SAMPLES`` (default 2) sets the number of samples; ``BRAIN_INTERVIEW_MEMORY=1``
also runs the memory pipeline at close (slow).
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import pytest
from doubts_scenario import (  # type: ignore[import-not-found]
    ASK_BACK,
    FRAUD,
    ING,
    REGION,
    SP,
    TM,
    TODAY,
    Owner,
    build,
    plan_for,
)

from brain.config import Settings, load_settings
from brain.interview.session import InterviewSession
from brain.lint.graph import Graph
from brain.llm import LLM
from brain.vault.ops import AddFact, DemoteEntity, MergeEntities, RefileFact

pytestmark = pytest.mark.llm
SAMPLES = int(os.environ.get("BRAIN_INTERVIEW_SAMPLES", "2"))
MEMORY = os.environ.get("BRAIN_INTERVIEW_MEMORY") == "1"
MAX_TURNS = 24


@pytest.fixture
def client():
    llm = LLM(load_settings().llm)
    if not llm.health():
        llm.close()
        pytest.skip("local model server is not running")
    yield llm
    llm.close()


def _no_memory(vault, llm, transcript, **kwargs):  # noqa: ANN001, ANN003, ANN202
    return {"episodes": 0}


def _right(op) -> bool:  # noqa: ANN001
    """The changes the owner's answers call for."""
    if isinstance(op, AddFact):
        return op.subject == SP and op.p == "part_of" and op.o == "[[orgs/fabrikam-analytics]]"
    if isinstance(op, DemoteEntity):
        return op.id == ING
    if isinstance(op, RefileFact):  # the demotion re-files Alex's worked_on Ingestion
        return op.p == "has_skill"
    if isinstance(op, MergeEntities):
        return {op.keep, op.merge} == {FRAUD, TM}
    return False


def run_sample(root: Path, client: LLM) -> dict:
    env = build(root)
    plan, infos = plan_for(env)
    owner = Owner()
    session = InterviewSession(
        env.vault,
        client,
        env.report(TODAY),
        REGION,
        plan,
        Settings(vault=env.vault.root),
        minutes=30,
        infos=infos,
    )
    lines: list[str] = []
    turn_seconds: list[float] = []
    context_ok: bool | None = None
    reply = session.opening()
    lines.append(f"brain › {reply.message}")
    for _ in range(MAX_TURNS):
        if reply.done:
            break
        answer = "Let's stop here." if reply.event == "continue" else owner.answer(reply.message)
        lines.append(f"you   › {answer}")
        start = time.perf_counter()
        reply = session.reply(answer)
        turn_seconds.append(time.perf_counter() - start)
        lines.append(f"brain › {reply.message}  [{reply.event}, {turn_seconds[-1]:.1f}s]")
        if answer == ASK_BACK:
            text = reply.message.casefold()
            context_ok = reply.event == "context" and "rules engine" in text and "screens" in text
    if not reply.done:
        session.reply("stop")
    result = session.close(process=None if MEMORY else _no_memory, db=env.db if MEMORY else None)
    graph = Graph(env.vault, TODAY)
    ops = [op for s in result.settled for op in s.ops]
    wrong = [op for op in ops if not _right(op)]
    sp_part_of = [r.fact.o for r in graph.facts_of(SP) if r.fact.p == "part_of"]
    settled_ok = {
        "stream processing stays a project": SP in graph.pages
        and graph.pages[SP].type == "project",
        "stream processing at Fabrikam": any(isinstance(op, AddFact) and _right(op) for op in ops),
        "ingestion demoted": any(isinstance(op, DemoteEntity) and op.id == ING for op in ops),
        "fraud alerts = transaction monitoring": any(isinstance(op, MergeEntities) for op in ops),
    }
    links = [s for what, s in session.timings if what == "link"]
    classify = [s for what, s in session.timings if what == "classify"]
    compose = [s for what, s in session.timings if what == "compose"]
    return {
        "lines": lines,
        "trace": session.trace,
        "settled": [(s.doubt, s.action, round(s.p, 2), s.detail) for s in result.settled],
        "asked": session.asked,
        "repeats": owner.repeats,
        "declined_reasked": owner.declined_reasked,
        "context_ok": context_ok,
        "settled_ok": settled_ok,
        "wrong": wrong,
        "sp_part_of": sp_part_of,
        "after": {
            "ingestion page": ING in graph.pages,
            "transaction monitoring page": TM in graph.pages,
            "pending reviews": len(list(env.vault.reviews_pending_dir.glob("*.md"))),
        },
        "turn_s": turn_seconds,
        "link_s": links,
        "classify_s": classify,
        "compose_s": compose,
    }


def test_interview_after_lint_live(tmp_path: Path, git_env: None, client: LLM) -> None:
    results = []
    for n in range(SAMPLES):
        r = run_sample(tmp_path / f"vault-{n}", client)
        results.append(r)
        print(f"\n=== sample {n + 1} ===")
        print("\n".join(r["lines"]))
        for key, what in r["trace"]:
            if what.startswith(("skipped", "follow-up dropped", "fallback", "context")):
                print(f"  trace {key}: {what}")
        for s in r["settled"]:
            print(f"  settled {s}")
        ok = sum(r["settled_ok"].values())
        print(
            f"  asked {r['asked']} · repeats {len(r['repeats'])} {r['repeats']} · declined "
            f"re-asked {len(r['declined_reasked'])} {r['declined_reasked']} · ask-back "
            f"context {r['context_ok']} · doubts settled right {ok}/4 {r['settled_ok']} · "
            f"wrong changes {len(r['wrong'])} {r['wrong']}"
        )
        print(f"  graph after close: part_of of stream processing {r['sp_part_of']} · {r['after']}")
        turns, links = r["turn_s"], r["link_s"]
        print(
            f"  per turn: mean {statistics.mean(turns):.2f}s max {max(turns):.2f}s · linking "
            f"decisions {len(links)}, mean {statistics.mean(links or [0]):.2f}s · classify mean "
            f"{statistics.mean(r['classify_s'] or [0]):.2f}s · compose mean "
            f"{statistics.mean(r['compose_s'] or [0]):.2f}s"
        )
    for r in results:
        assert r["wrong"] == []
