"""Ending an interview on content: dry answers (nothing added) in a row, most of the last few, or
a dry block; a talkative owner is never stopped early. Scripted owners, fake model."""

from __future__ import annotations

from typing import Any

import pytest
from interview_helpers import TODAY, Clock, ScriptedLLM, at  # type: ignore[import-not-found]

from brain.config import Settings
from brain.interview.planner import PlanItem, build_plan
from brain.interview.questions import read_questions
from brain.interview.session import InterviewSession, is_dry

ATLAS = "projects/atlas-migration"
DASH = "projects/legacy-dashboards"


def dec(label: str, p_thread: float = 0.0) -> list[dict[str, Any]]:
    return [
        {"name": "answer", "type": "choice", "choice": label, "probabilities": {label: 1.0}},
        {
            "name": "new_thread",
            "type": "predicate",
            "choice": "yes" if p_thread >= 0.5 else "no",
            "probabilities": {"yes": p_thread, "no": 1 - p_thread},
        },
    ]


def step(entity: str | None, slot: str, hint: str = "Tell me more?") -> PlanItem:
    topic = f"Projects: {entity.rpartition('/')[2]}" if entity else "Projects"
    return PlanItem("projects", "ask", hint, topic, entity=entity, slot=slot)


BLOCKS = [
    step(ATLAS, "status"),
    step(ATLAS, "role"),
    step(ATLAS, "deadline"),
    step(DASH, "status"),
    step(DASH, "role"),
    PlanItem("projects", "discover", "Any other projects?", "Projects", slot="more"),
]


@pytest.fixture
def make(env):
    def _make(llm: ScriptedLLM, region: str = "career.current_job", **kw) -> InterviewSession:
        report = env.report(TODAY)
        plan = kw.pop("plan", None) or build_plan(report, region, read_questions(env.vault))
        settings = Settings(vault=env.vault.root)
        return InterviewSession(
            env.vault, llm, report, region, plan, settings, clock=Clock(at(TODAY)), **kw
        )

    return _make


def scripted(labels: list[str], p_thread: float = 0.0) -> ScriptedLLM:
    llm = ScriptedLLM(decisions=[dec(label, p_thread) for label in labels])
    words = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliet",
        "kilo",
        "lima",
    ]
    llm.structured_queue.extend({"reasoning": "r", "message": f"and {w}?"} for w in words * 3)
    return llm


def play(session: InterviewSession, answers: list[str]) -> list[str]:
    events = [session.opening().event]
    for answer in answers:
        reply = session.reply(answer)
        events.append(reply.event)
        if reply.done:
            break
    return events


def test_dry_answers() -> None:
    assert is_dry("declined", "I have no idea, finance handles the budget")
    assert is_dry("refused", "I'd rather not say")
    assert is_dry("unclear", "banana")
    assert is_dry("partial", "Not much, really.")
    assert is_dry("partial", "It's fine I guess")
    assert not is_dry("partial", "We use Spark, mostly")  # names something
    assert not is_dry("answered", "Yes.")  # a confirmation adds something
    assert not is_dry("answered", "Theo.")


TALKATIVE = [
    "Theo Marchetti, since the reorg in spring; we meet every Monday.",
    "Mostly the nightly batch pipelines, plus on-call one week a month.",
    "Honestly no idea, finance keeps that to themselves.",  # one decline: no stop
    "I'm moving the warehouse to the new cluster with Sam's team, it's half done.",
    "Maya Lindqvist joined last month; she runs the data quality checks now.",
    "Not sure, I'd have to look it up.",
    "We picked Kiln for the scheduler after a long debate, I pushed for it.",
    "It's busy but I like the people, and the Atlas work is the fun part.",
]


def test_a_talkative_owner_is_not_stopped(make) -> None:
    labels = ["answered", "answered", "declined", "answered", "answered", "declined"]
    session = make(scripted(labels + ["answered"] * 10))
    session.plan = session.plan * 3
    events = play(session, TALKATIVE)
    assert not session.done, events
    assert session.dry == 0 and sum(session.yields) == 2


def test_a_terse_owner_is_stopped_on_content(make) -> None:
    session = make(scripted(["partial", "declined", "partial"]))
    session.plan = session.plan * 3
    events = play(session, ["Fine.", "Dunno.", "Not much really."])
    # a vague answer made of filler gets no "tell me more?" follow-up
    assert events == ["question", "question", "question", "content"]
    assert session.done and session.answers == 3
    last = session.history[-1]["content"]
    assert last == "I think I've got what I need on current job for now. Thanks!"


def test_the_next_region_is_offered(make, env) -> None:
    session = make(scripted(["declined"] * 3))
    session.plan = session.plan * 3
    session.opening()
    session.reply("No idea.")
    session.reply("No idea.")
    reply = session.reply("No idea either.")
    assert reply.done and reply.event == "declined"
    assert reply.next_region and reply.next_region != "career.current_job"
    private = {r.id for r in env.report(TODAY).regions.values() if r.private}
    assert reply.next_region not in private
    assert "Next time" not in reply.message  # the CLI prints it on its own line


def test_most_of_the_last_answers_dry_ends_it(make) -> None:
    labels = ["declined", "declined", "answered", "declined", "declined"]
    session = make(scripted(labels))
    session.plan = session.plan * 3
    events = play(session, ["no idea", "no", "Theo Marchetti.", "no idea", "nope"])
    assert events[-1] == "content" and session.answers == 5


def test_two_dry_answers_leave_the_block(make) -> None:
    session = make(scripted(["declined", "declined", "answered"]), region="projects")
    session.plan = list(BLOCKS)
    session.opening()
    session.reply("No idea.")
    nxt = session.reply("Not sure.")  # two dry answers in the Atlas block: its last step is left
    assert nxt.event == "question" and session.item.entity == DASH
    assert (f"projects/deadline/{ATLAS}", "skipped: block dry") in session.trace
    session.reply("I set them up years ago and still maintain them.")
    assert not session.done and session.dry == 0  # the streak was broken by an answer


def test_skips_count_as_dry(make) -> None:
    session = make(scripted([]))
    session.plan = session.plan * 3
    events = play(session, ["skip", "skip", "skip"])
    assert events[-1] == "declined" and session.done
