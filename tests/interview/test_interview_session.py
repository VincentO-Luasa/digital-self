"""A scripted interview with a fake model and a fake memory pipeline."""

from __future__ import annotations

import subprocess
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from interview_helpers import (  # type: ignore[import-not-found]
    LATER,
    TODAY,
    Clock,
    ScriptedLLM,
    at,
)

from brain.config import Settings
from brain.interview.planner import build_plan
from brain.interview.questions import add_question, read_questions
from brain.interview.session import (
    FOLLOW_UP_HINT,
    PRIVATE_ACK,
    PRIVATE_MARKER,
    PRIVATE_PLACEHOLDER,
    SKIP_MARKER,
    START,
    InterviewSession,
    date_words,
    fallback,
    invented_claim,
    one_question,
    parse_command,
    ungrounded,
)
from brain.prompts import load_prompt
from brain.vault.raw import parse_transcript


def msg(text: str) -> dict[str, str]:
    return {"reasoning": "r", "message": text}


def dec(label: str, p_thread: float = 0.0) -> list[dict[str, Any]]:
    return [
        {
            "name": "answer",
            "type": "choice",
            "choice": label,
            "probabilities": {label: 1.0},
            "confidence": 1.0,
        },
        {
            "name": "new_thread",
            "type": "predicate",
            "choice": "yes" if p_thread >= 0.5 else "no",
            "probabilities": {"yes": p_thread, "no": 1 - p_thread},
        },
    ]


class FakeProcess:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[tuple, dict]] = []
        self.error = error

    def __call__(self, vault, llm, transcript, *, channel, regions_hint=None, today=None):
        self.calls.append(((vault, llm, transcript), {"channel": channel, "regions": regions_hint}))
        self.calls[-1][1]["today"] = today
        if self.error:
            raise self.error
        return {"episodes": 1}


class ReportingProcess(FakeProcess):
    def __init__(self, report) -> None:
        super().__init__()
        self.report = report

    def __call__(self, vault, llm, transcript, **kwargs):
        self.calls.append(((vault, llm, transcript), kwargs))
        return self.report


class PrivateAwareProcess(FakeProcess):
    def __call__(self, vault, llm, transcript, **kwargs):  # accepts private_turns
        self.calls.append(((vault, llm, transcript), kwargs))
        return {"episodes": 1}


def structured_calls(llm: ScriptedLLM) -> list[list[dict]]:
    return [c[1]["messages"] for c in llm.calls if c[0] == "structured"]


@pytest.fixture
def make(env):
    def _make(
        llm: ScriptedLLM, region: str = "career.current_job", today=TODAY, step: float = 5.0, **kw
    ) -> InterviewSession:
        report = env.report(today)
        plan = kw.pop("plan", None) or build_plan(report, region, read_questions(env.vault))
        settings = Settings(vault=env.vault.root)
        clock = Clock(at(today), step)
        return InterviewSession(env.vault, llm, report, region, plan, settings, clock=clock, **kw)

    return _make


def turns(session: InterviewSession) -> list[tuple[str, str]]:
    assert session.transcript is not None
    fm, parsed = parse_transcript(session.transcript.path.read_text(encoding="utf-8"))
    assert fm["channel"] == "interview" and fm["region"] == session.region_id
    return [(t.role, t.text) for t in parsed]


# --- pure helpers ------------------------------------------------------------------------------


def test_commands() -> None:
    assert parse_command(" Skip. ") == ("skip", "")
    assert parse_command("/stop") == ("stop", "")
    assert parse_command("private") == ("private", "")
    assert parse_command("Private: we split up last year") == ("private", "we split up last year")
    assert parse_command("privately I think so") == (None, "privately I think so")
    assert parse_command("skip the details: yes") == (None, "skip the details: yes")


def test_grounding_check() -> None:
    allowed = "Atlas go-live is planned for 2026-11-15. Alex Rivera works at Northwind Labs."
    assert (
        ungrounded("Is Atlas still on track? Northwind Labs's plan was November 15th.", allowed)
        == []
    )
    assert ungrounded("Since 2026, how is Atlas?", allowed) == []
    assert ungrounded("Do you still enjoy Berlin and Spark?", allowed) == ["Berlin", "Spark"]
    assert ungrounded("Is the deadline still March 3?", allowed) == ["March", "3"]
    assert ungrounded("Where did you grow up? I'm curious.", "") == []  # sentence starts
    assert {"2026", "november", "nov", "11", "15"} <= date_words(allowed)


def test_one_question_and_fallbacks(make) -> None:
    assert one_question("Got it. Who is your manager? And since when?") == (
        "Got it. Who is your manager?"
    )
    assert one_question("Nice.\n Who  is it?") == "Nice. Who is it?"
    session = make(ScriptedLLM(), region="projects", today=LATER)
    item = session.plan[0]
    assert (
        fallback(item, "confirm")
        == "I have noted: Atlas is in the testing phase. Is that still right?"
    )
    assert fallback(item, "follow_up") == "Could you tell me a bit more about that?"
    assert fallback(item, "ask") == item.hint


# --- the conversation --------------------------------------------------------------------------


def test_opening_prompt_layout_and_transcript(make) -> None:
    llm = ScriptedLLM(
        structured_responses=[msg("You're on the data platform team: who's your manager?")]
    )
    session = make(llm)
    reply = session.opening()
    assert reply.message == "You're on the data platform team: who's your manager?"
    assert not reply.done
    [(system, user)] = structured_calls(llm)
    assert system["role"] == "system"
    assert system["content"].startswith(load_prompt("interview_system"))
    assert "Name: Alex Rivera" in system["content"]  # core card in the cached prefix
    assert user["role"] == "user"
    assert user["content"].startswith(
        f"{START}\n\nPLAN\ntopic: Current job\nmode: ask\nhint: Who do you report to?\n"
        "about:\n- (nothing)\nknown:\n- (nothing yet)\ncontext:\n- "
    )
    assert "today: 2026-10-26" in user["content"]
    assert turns(session) == [
        (
            "system",
            "[interview] region:: career.current_job · topic:: Current job · slot:: manager · "
            "mode:: ask",
        ),
        ("assistant", reply.message),
    ]


def test_follow_up_then_next_step_with_a_stable_prefix(make) -> None:
    llm = ScriptedLLM(
        structured_responses=[
            msg("Who do you report to?"),
            msg("How long has that been?"),
            msg("And your week?"),
        ],
        decisions=[dec("answered", 0.9), dec("answered", 0.1)],
    )
    session = make(llm)
    session.opening()
    first = session.reply("Theo Marchetti, since the reorg.")
    assert first.event == "follow_up" and first.message == "How long has that been?"
    second = session.reply("About a year.")
    assert second.event == "question" and second.message == "And your week?"
    calls = structured_calls(llm)
    assert f"mode: follow_up\nhint: {FOLLOW_UP_HINT}\n" in calls[1][-1]["content"]
    assert calls[1][-1]["content"].startswith("Theo Marchetti, since the reorg.\n\nPLAN")
    assert (
        "mode: more\nhint: What does a typical week of work involve, besides"
        in calls[2][-1]["content"]
    )
    assert calls[2][:3] == calls[1][:3]  # earlier turns never change: the prefix stays cached
    assert calls[2][1:3] == [
        {"role": "user", "content": START},
        {"role": "assistant", "content": "Who do you report to?"},
    ]
    assert calls[2][3] == {"role": "user", "content": "Theo Marchetti, since the reorg."}
    decide = [c[1] for c in llm.calls if c[0] == "decide"][0]
    assert "Interviewer: Who do you report to?\nOwner: Theo Marchetti" in decide["text"]
    assert [q.name for q in decide["questions"]] == ["answer", "new_thread"]


def test_partial_answers_get_at_most_two_follow_ups(make) -> None:
    llm = ScriptedLLM(decisions=[dec("partial")] * 3, default_reply="unused")
    llm.structured_queue.extend(msg(f"Question {n}?") for n in range(4))
    session = make(llm)
    session.opening()
    events = [session.reply(f"some {n}").event for n in range(3)]
    assert events == ["follow_up", "follow_up", "question"]
    assert session.item.slot == "responsibilities"


def test_unclear_is_clarified_once_and_declines_end_the_region(make) -> None:
    llm = ScriptedLLM(
        decisions=[
            dec("unclear"),
            dec("unclear"),
            dec("declined"),
            dec("declined"),
            dec("declined"),
        ]
    )
    llm.structured_queue.extend(msg(f"Question {n}?") for n in range(5))
    session = make(llm)
    session.plan = session.plan * 2  # long enough for three declines
    session.opening()
    assert session.reply("banana").event == "clarify"
    assert session.reply("banana again").event == "question"
    # a third answer in a row that adds nothing: the session ends on content
    last = session.reply("no idea")
    assert last.done and last.event == "content"
    assert last.message == "I think I've got what I need on current job for now. Thanks!"

    llm = ScriptedLLM(decisions=[dec("answered"), *[dec("declined")] * 3])
    llm.structured_queue.extend(msg(f"Question {n}?") for n in range(5))
    session = make(llm)
    session.plan = session.plan * 2
    session.opening()
    assert session.reply("Theo Marchetti.").event == "question"
    assert session.reply("no idea").event == "question"
    assert session.reply("rather not").event == "question"
    last = session.reply("pass on that")
    assert last.done and last.event == "declined"
    assert "Let's leave current job there for today" in last.message


def test_skip_and_stop(make) -> None:
    llm = ScriptedLLM(
        structured_responses=[msg("Who do you report to?"), msg("What's a usual week?")]
    )
    session = make(llm)
    session.opening()
    reply = session.reply("skip")
    assert reply.event == "skip" and reply.message == "What's a usual week?"
    assert not [c for c in llm.calls if c[0] == "decide"]
    assert session.skipped == ["career.current_job/manager"]
    stop = session.reply("stop")
    assert stop.done and stop.event == "stop"
    assert session.reply("hello?").done
    roles = turns(session)
    assert ("user", "skip") in roles and ("system", SKIP_MARKER) in roles
    assert roles[-2:] == [("user", "stop"), ("assistant", stop.message)]
    assert session.answers == 0


def test_private_marks_the_answers_of_the_step(make) -> None:
    llm = ScriptedLLM(decisions=[dec("answered", 0.9), dec("answered")])
    llm.structured_queue.extend(msg(f"Question {n}?") for n in range(4))
    session = make(llm)
    session.opening()
    ack = session.reply("private")
    assert (ack.event, ack.message) == ("private", PRIVATE_ACK)
    assert len(structured_calls(llm)) == 1  # no model call for the command
    # private, and a new thread (p=0.9): still no follow-up on a private answer
    nxt = session.reply("Theo Marchetti; he is struggling lately.")
    assert nxt.event == "question"
    session.reply("Pipelines and on-call.")  # next step: not private
    t = turns(session)
    private = [
        n for n, (role, text) in enumerate(t, 1) if role == "system" and text == PRIVATE_MARKER
    ]
    assert len(private) == 2  # after the command, then before the private answer
    assert session.private_turns == [private[1] + 1]
    assert t[session.private_turns[0] - 1] == ("user", "Theo Marchetti; he is struggling lately.")
    assert t[-3] == ("user", "Pipelines and on-call.")
    assert t[-4] != ("system", PRIVATE_MARKER)
    # later questions are written without the private answer
    seen = "\n".join(m["content"] for m in session.history)
    assert "struggling" not in seen and PRIVATE_PLACEHOLDER in seen

    inline = make(
        ScriptedLLM(decisions=[dec("answered")], structured_responses=[msg("Q?"), msg("Q2?")])
    )
    inline.opening()
    inline.reply("private: Theo Marchetti")
    assert inline.private_turns and turns(inline)[inline.private_turns[0] - 1] == (
        "user",
        "Theo Marchetti",
    )


def test_private_region_makes_every_answer_private(make) -> None:
    llm = ScriptedLLM(decisions=[dec("answered"), dec("answered")])
    llm.structured_queue.extend(msg(f"Question {n}?") for n in range(3))
    session = make(llm, region="relationships.friends")
    session.opening()
    session.reply("Sam, and Maya from work.")
    assert len(session.private_turns) == 1


def test_ungrounded_or_failed_messages_fall_back_to_the_hint(make) -> None:
    llm = ScriptedLLM(
        structured_responses=[
            msg("Do you still report to Ingrid in Berlin?"),
            RuntimeError("server hiccup"),
            msg("   "),
        ],
        decisions=[dec("answered"), dec("answered")],
    )
    session = make(llm)
    assert session.opening().message == "Who do you report to?"
    assert session.reply("Theo.").message == (
        "So far I know: Keeps the nightly batch pipelines running. "
        "What does a typical week of work involve, besides what I already know?"
    )
    assert any("ungrounded Ingrid, Berlin" in what for _, what in session.trace)
    assert any("fallback: RuntimeError" in what for _, what in session.trace)
    last = session.reply("Pipelines.")
    assert last.message.startswith("Is there anything else about your current job")


def test_a_repeated_question_falls_back_to_the_hint(make) -> None:
    llm = ScriptedLLM(
        structured_responses=[msg("Who is your manager?"), msg("Who is your manager?")],
        decisions=[dec("answered")],
    )
    session = make(llm)
    assert session.opening().message == "Who is your manager?"
    nxt = session.reply("skip")
    assert nxt.message != "Who is your manager?"
    assert any("fallback: repeats" in what for _, what in session.trace)


def test_time_is_soft_and_the_owner_decides_to_go_on(make) -> None:
    # at the limit, between steps: "keep going?" instead of a hard stop
    llm = ScriptedLLM(decisions=[dec("answered", 0.9)], structured_responses=[msg("Q?")])
    session = make(llm, step=40, minutes=1)
    session.opening()
    offer = session.reply("Theo.")  # over time: no follow-up despite the new thread
    assert offer.event == "continue" and not offer.done
    assert offer.message == "We've covered current job. Keep going, or stop here?"
    unclear = session.reply("hmm, the weather is nice")  # anything unclear stops
    assert unclear.done and unclear.event == "time" and "about 1 minute," in unclear.message
    assert session.answers == 1  # the reply to the offer is no answer

    stop = make(
        ScriptedLLM(decisions=[dec("answered")], structured_responses=[msg("Q?")]),
        step=40,
        minutes=1,
    )
    stop.opening()
    stop.reply("Theo.")
    assert stop.reply("no, that's enough").event == "stop"

    llm = ScriptedLLM(decisions=[dec("answered")] * 2)
    llm.structured_queue.extend(msg(f"Q{n}?") for n in range(3))
    more = make(llm, step=40, minutes=1)
    more.opening()
    more.reply("Theo.")
    more.clock.step = 1  # type: ignore[attr-defined]
    again = more.reply("Yes, keep going")
    assert again.event == "question" and "typical week" in again.message
    assert more.deadline > more.minutes  # half the time again, from now
    assert ("career.current_job/responsibilities", "keep going") in more.trace


def test_time_limit_finishes_the_entity_block_first(make, env) -> None:
    for text, doubt in (
        ("Is Atlas a project of yours, or a topic?", "type projects/atlas-migration"),
        ("Is Atlas part of your work at Northwind Labs?", "located_in projects/atlas-migration"),
    ):
        add_question(
            env.vault,
            text,
            region="projects",
            origin="lint",
            about="projects/atlas-migration",
            doubt=doubt,
        )
    llm = ScriptedLLM(decisions=[dec("answered")] * 2)
    llm.structured_queue.extend(msg(f"Q{n}?") for n in range(3))
    session = make(llm, region="projects", step=40, minutes=1)
    assert [p.block for p in session.plan[:2]] == ["projects/atlas-migration"] * 2
    session.opening()
    assert session.reply("A project.").event == "question"  # same block: finished first
    assert session.reply("Yes.").event == "continue"


def test_question_limit(make) -> None:
    llm = ScriptedLLM(decisions=[dec("answered", 0.9)] * 3)
    llm.structured_queue.extend(msg(f"Q{n}?") for n in range(3))
    capped = make(llm, max_questions=2)
    capped.opening()
    capped.reply("a")
    assert capped.reply("b").event == "limit"


def test_empty_plan(make) -> None:
    session = make(ScriptedLLM(), plan=None)
    session.plan = []
    reply = session.opening()
    assert reply.done and reply.event == "empty"
    assert session.close(FakeProcess()).transcript is None


# --- closing -----------------------------------------------------------------------------------


def test_close_runs_the_memory_pipeline_and_updates_questions(make, env) -> None:
    item = add_question(
        env.vault, "Who mentors you at work?", region="career.current_job", origin="lint"
    )
    llm = ScriptedLLM(decisions=[dec("answered"), dec("declined")])
    llm.structured_queue.extend(msg(f"Q{n}?") for n in range(3))
    session = make(llm)
    assert session.plan[0].question_id == item.id
    session.opening()
    session.reply("Theo Marchetti does.")
    session.reply("No idea.")
    process = FakeProcess()
    result = session.close(process, db=env.db)
    [(args, kwargs)] = process.calls
    assert args == (env.vault, llm, session.transcript)
    assert kwargs == {"channel": "interview", "regions": ["career.current_job"], "today": TODAY}
    assert result.report == {"episodes": 1} and result.error is None
    assert result.transcript and result.transcript.startswith("raw/conversations/2026/10/conv-")
    assert result.answered == [item.id]
    assert next(i for i in read_questions(env.vault) if i.id == item.id).done
    assert result.questions_changed  # coverage items written
    log = subprocess.run(
        ["git", "log", "--format=%s"], cwd=env.vault.root, capture_output=True, text=True
    ).stdout
    assert "brain: update questions.md after interview" in log
    assert f"brain: transcript {result.transcript}" in log


def test_close_passes_private_turns_when_the_pipeline_takes_them(make, env) -> None:
    llm = ScriptedLLM(decisions=[dec("answered")], structured_responses=[msg("Q?"), msg("Q2?")])
    session = make(llm)
    session.opening()
    session.reply("private: Theo")
    process = PrivateAwareProcess()
    session.close(process, db=env.db)
    assert process.calls[0][1]["private_turns"] == session.private_turns
    assert process.calls[0][1]["db"] is env.db

    plain = FakeProcess()  # no private_turns parameter: the transcript markers carry it
    session2 = make(
        ScriptedLLM(decisions=[dec("answered")], structured_responses=[msg("Q?"), msg("Q2?")])
    )
    session2.opening()
    session2.reply("private: Theo")
    session2.close(plain)
    assert "private_turns" not in plain.calls[0][1]


def test_close_without_answers_or_with_a_failing_pipeline(make, env) -> None:
    session = make(ScriptedLLM(structured_responses=[msg("Q?")]))
    session.opening()
    session.reply("stop")
    process = FakeProcess()
    result = session.close(process)
    assert process.calls == [] and result.transcript

    item = add_question(
        env.vault, "Who mentors you at work?", region="career.current_job", origin="lint"
    )
    failing = make(
        ScriptedLLM(decisions=[dec("answered")], structured_responses=[msg("Q?"), msg("Q2?")])
    )
    failing.opening()
    failing.reply("Theo.")
    result = failing.close(FakeProcess(RuntimeError("boom")))
    assert result.error == "RuntimeError: boom" and result.transcript
    assert not next(i for i in read_questions(env.vault) if i.id == item.id).done  # retry later
    assert timedelta(0) < failing.clock() - at(TODAY)


def test_questions_stay_open_when_an_episode_failed(make, env) -> None:
    item = add_question(
        env.vault, "Who mentors you at work?", region="career.current_job", origin="lint"
    )
    session = make(
        ScriptedLLM(decisions=[dec("answered")], structured_responses=[msg("Q?"), msg("Q2?")])
    )
    session.opening()
    session.reply("Theo Marchetti does.")
    failed = SimpleNamespace(
        status="processed",
        error=None,
        episodes=[SimpleNamespace(error=None), SimpleNamespace(error="extraction failed: cut off")],
    )
    result = session.close(ReportingProcess(failed))
    assert result.error is None and result.answered == []
    assert not next(i for i in read_questions(env.vault) if i.id == item.id).done


def test_invented_provenance_is_caught() -> None:
    said = "We went live in December, a few weeks late."
    assert invented_claim("You listed Stream processing as a project: is it one?", "")
    assert invented_claim("You mentioned Atlas entered testing in October. Still so?", said)
    assert invented_claim("You went live in December: what caused the delay?", said) is None
    assert invented_claim("You mentioned going live in December: what was late?", said) is None
    assert invented_claim("Your document “cv” mentions Kiln: is it yours?", "") is None
