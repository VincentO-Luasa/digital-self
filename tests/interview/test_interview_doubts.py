"""Lint doubts in an interview, with a scripted model (fictional persona: Alex Rivera, Northwind
Labs now, Fabrikam Analytics before): one combined question per entity, answer linking, asking
back, declined information, settling doubts from the answers, and the budget."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from doubts_scenario import (  # type: ignore[import-not-found]
    ASK_BACK,
    DECLINE,
    FRAUD,
    ING,
    REGION,
    SCRAPER,
    SP,
    SP_ANSWER,
    TM,
    TODAY,
    build,
    plan_for,
)
from interview_helpers import Clock, Env, ScriptedLLM, at, choice  # type: ignore[import-not-found]

from brain.config import Settings
from brain.interview.planner import build_plan
from brain.interview.questions import GapItem, read_questions
from brain.interview.session import PRIVATE_PLACEHOLDER, InterviewSession
from brain.interview.settle import kind_question, part_of_question, same_question
from brain.lint.graph import Graph
from brain.vault.raw import parse_transcript


@pytest.fixture
def denv(tmp_path: Path, git_env: None) -> Env:
    env = build(tmp_path / "vault")
    yield env
    env.db.close()  # type: ignore[attr-defined]


def msg(text: str) -> dict[str, str]:
    return {"reasoning": "r", "message": text}


def label(value: str, p_thread: float = 0.0) -> list:
    return [
        choice(
            "answer", value, 1.0, ["answered", "partial", "declined", "refused", "unclear", "asked"]
        ),
        choice("new_thread", "yes" if p_thread >= 0.5 else "no", 1.0),
    ]


def settle_with(**answers: tuple[str, float]):
    """decider for the settle decisions: per question name, (value, p)."""

    def decide(_text: str, questions):  # noqa: ANN001, ANN202
        out = []
        for q in questions:
            if q.name in answers:
                value, p = answers[q.name]
                out.append(choice(q.name, value, p, [c.value for c in q.choices]))
        return out

    return decide


def session(env: Env, llm: ScriptedLLM, plan=None, infos=None, **kw) -> InterviewSession:
    if plan is None:
        plan, infos = plan_for(env)
    return InterviewSession(
        env.vault,
        llm,
        env.report(TODAY),
        REGION,
        plan,
        Settings(vault=env.vault.root),
        clock=Clock(at(TODAY), kw.pop("step", 5.0)),
        infos=infos,
        **kw,
    )


def step_of(plan, entity: str) -> int:
    return next(n for n, p in enumerate(plan) if p.entity == entity and p.doubts)


class NoPipeline:
    def __call__(self, vault, llm, transcript, **kwargs):  # noqa: ANN001, ANN003, ANN204
        return {"episodes": 0}


# --- A: one block, one question per entity -----------------------------------------------------


def test_doubts_about_one_entity_are_one_question(denv: Env) -> None:
    plan, infos = plan_for(denv)
    sp = [p for p in plan if p.entity == SP]
    assert len(sp) == 1  # type + part_of: one combined question, both items ticked by it
    assert sp[0].hint == (
        "What is Stream processing to you: a project (at one of your jobs, or on your own), "
        "or a topic or tool you work with?"
    )
    assert set(sp[0].doubts) == {f"type {SP}", f"part_of {SP}"} and len(sp[0].question_ids) == 2
    merge = next(p for p in plan if p.entity == FRAUD)
    assert merge.hint == (
        "Are Fraud alerts and Transaction monitoring one and the same project, or two "
        "different ones?"
    )
    assert merge.brief == (
        "Fraud alerts: A rules engine that flags suspicious card payments "
        "(from your document “homepage”)",
        "Transaction monitoring: Screens payments for fraud patterns in real time "
        "(from your document “homepage”)",
    )
    # pages that may disappear get no coverage questions this time
    assert not [p for p in plan if p.entity in (ING, TM, SP, FRAUD) and not p.doubts]
    # doubt blocks alternate with coverage steps
    kinds = ["doubt" if p.doubts else p.mode for p in plan]
    assert kinds[0] == "doubt" and kinds[1] != "doubt" and kinds[-1] == "discover"
    # without infos (no vault access): each item as written, still one block per entity
    bare = build_plan(denv.report(TODAY), REGION, read_questions(denv.vault))
    assert [p.block for p in bare if p.entity == SP] == [SP, SP]


def test_at_most_max_doubts_entities_per_session(denv: Env) -> None:
    items = read_questions(denv.vault)
    plan = build_plan(denv.report(TODAY), REGION, items, max_doubts=1)
    assert len({p.block for p in plan if p.doubts}) == 1
    many = items + [
        GapItem(f"Is Thing {n} a project?", REGION, origin="lint", doubt=f"type projects/t{n}")
        for n in range(6)
    ]
    plan = build_plan(denv.report(TODAY), REGION, many)
    assert len({p.block for p in plan if p.doubts}) == 4


def test_doubts_get_only_part_of_the_time(denv: Env) -> None:
    llm = ScriptedLLM(decisions=[label("answered")] * 12)
    llm.structured_queue.extend(msg(f"Question number {w}?") for w in "abcdefghijkl")
    s = session(denv, llm, minutes=10, step=1)
    s.opening()
    s.clock.now += timedelta(minutes=5)  # type: ignore[attr-defined]  # past 40% of the time
    while not s.done and not any(w == "skipped: doubt budget" for _, w in s.trace):
        s.reply("Nothing more on that.")
    assert any(what == "skipped: doubt budget" for _, what in s.trace)
    asked = [s.plan[i] for i in s.asked_steps]
    assert len([p for p in asked if p.doubts]) == 1  # the first block only


# --- B: answer linking -------------------------------------------------------------------------


def test_an_answered_step_is_skipped_and_settled_at_close(denv: Env) -> None:
    plan, infos = plan_for(denv)
    sp, ing = step_of(plan, SP), step_of(plan, ING)
    plan = [plan[sp], plan[ing], plan[-1]]  # stream processing, then ingestion

    def link(_text: str, questions):  # noqa: ANN001, ANN202
        return ["answered" if "Ingestion" in q.instructions else "open" for q in questions]

    llm = ScriptedLLM(
        decisions=[label("answered")],
        link=link,
        decider=settle_with(kind=("project", 0.97), part_of=("Fabrikam Analytics", 0.95)),
    )
    llm.structured_queue.extend([msg("What is stream processing to you?"), msg("Anything else?")])
    s = session(denv, llm, plan, infos)
    s.opening()
    reply = s.reply(SP_ANSWER)
    assert reply.message == "Anything else?"
    assert ("projects/q/" + plan[1].question_ids[0], "skipped: answered earlier") in [
        (k.split("/", 1)[1] if k.startswith(REGION + "/") else k, w) for k, w in s.trace
    ] or any(w == "skipped: answered earlier" for _, w in s.trace)
    [(text, instructions)] = llm.linked
    assert "Owner: It was a project I built at Fabrikam" in text
    assert any("What is Ingestion to you" in i for i in instructions)

    # at close: stream processing gets part_of Fabrikam (stated, from the transcript),
    # ingestion is demoted (the owner said "not a project")
    llm.decider = settle_with(kind=("project", 0.97), part_of=("Fabrikam Analytics", 0.95))

    def by_question(text: str, questions):  # noqa: ANN001, ANN202
        title = questions[0].instructions
        if "Ingestion" in title:
            return [choice("kind", "topic", 0.95, [c.value for c in questions[0].choices])]
        return settle_with(kind=("project", 0.97), part_of=("Fabrikam Analytics", 0.95))(
            text, questions
        )

    llm.decider = by_question
    result = s.close(NoPipeline())
    actions = {x.doubt: x.action for x in result.settled}
    assert actions == {
        f"type {SP}": "kept",
        f"part_of {SP}": "applied",
        f"type {ING}": "applied",
    }
    g = Graph(denv.vault, TODAY)
    [part_of] = [r.fact for r in g.facts_of(SP) if r.fact.p == "part_of"]
    assert part_of.o == "[[orgs/fabrikam-analytics]]" and part_of.kind == "stated"
    assert part_of.src == [result.transcript]
    assert ING not in g.pages  # demoted; Alex's worked_on became has_skill "Ingestion"
    assert any(r.fact.p == "has_skill" and r.fact.o == "Ingestion" for r in g.facts_of("self"))
    done = {i.id for i in read_questions(denv.vault) if i.done}
    assert set(plan[0].question_ids) | set(plan[1].question_ids) <= done
    log = (denv.vault.root / "log.md").read_text(encoding="utf-8")
    assert "user | " + result.transcript in log


def test_a_follow_up_already_answered_is_dropped(denv: Env) -> None:
    plan, infos = plan_for(denv)
    plan = [plan[step_of(plan, SP)], plan[-1]]

    def link(_text: str, questions):  # noqa: ANN001, ANN202
        return ["answered" if "what was it" in q.instructions else "open" for q in questions]

    llm = ScriptedLLM(decisions=[label("answered", 0.9)], link=link)
    llm.structured_queue.extend(
        [
            msg("What is stream processing to you?"),
            msg("You built something for the media client: what was it?"),
            msg("Are you working on any other projects?"),
        ]
    )
    s = session(denv, llm, plan, infos)
    s.opening()
    reply = s.reply(SP_ANSWER + " " + SCRAPER)
    assert reply.message == "Are you working on any other projects?"
    assert any(w.startswith("follow-up dropped: answered earlier") for _, w in s.trace)
    # the follow-up was checked against what the owner said, and never asked
    [(text, [instruction])] = llm.linked
    assert "reporting scraper" in text and "media client: what was it" in instruction
    assert "what was it" not in "\n".join(m["content"] for m in s.history)
    # earlier answers go into later PLAN blocks ("already said"; the latest one is the message)
    assert s.said_lines(s.item) == []
    s.reply("No, that's all.")
    block = s.plan_block(s.item, "follow_up", TODAY)
    assert "already said:\n- It was a project I built at Fabrikam" in block


def test_a_declined_kind_of_information_is_not_asked_again(denv: Env) -> None:
    plan, infos = plan_for(denv)
    plan = [plan[step_of(plan, SP)], plan[step_of(plan, FRAUD)], plan[-1]]
    seen: list[str] = []

    def link(text: str, questions):  # noqa: ANN001, ANN202
        seen.append(text)
        return ["declined" if "client" in q.instructions else "open" for q in questions]

    llm = ScriptedLLM(decisions=[label("refused"), label("answered", 0.9)], link=link)
    llm.structured_queue.extend(
        [
            msg("Which client was stream processing for?"),
            msg("Are fraud alerts and transaction monitoring the same?"),
            msg("Who was the client for that rules engine?"),  # same kind: dropped
            msg("Are you working on any other projects?"),
        ]
    )
    s = session(denv, llm, plan, infos)
    s.opening()
    s.reply(DECLINE)
    final = s.reply("Yes, the same rules engine.")
    assert final.message == "Are you working on any other projects?"
    assert s.declined_lines() == [
        "- asked “Which client was stream processing for?”, the owner said: “I'd rather not "
        "say, it's confidential.”"
    ]
    assert "declined:\n- asked “Which client" in s.plan_block(s.item, "ask", TODAY)
    assert any(w.startswith("follow-up dropped: declined") for _, w in s.trace)
    assert "What the owner would not tell:\n- asked “Which client was" in seen[-1]


def test_linking_never_sees_a_private_answer(denv: Env) -> None:
    plan, infos = plan_for(denv)
    plan = [plan[step_of(plan, SP)], plan[step_of(plan, FRAUD)], plan[-1]]
    llm = ScriptedLLM(decisions=[label("answered"), label("answered")])
    llm.structured_queue.extend([msg("Q one?"), msg("Q two?"), msg("Q three?")])
    s = session(denv, llm, plan, infos)
    s.opening()
    s.reply("private: it was for a bank in trouble")
    s.reply("Two different ones.")
    assert all("bank" not in text for text, _ in llm.linked)
    assert all(PRIVATE_PLACEHOLDER not in text for text, _ in llm.linked)
    # and nothing is settled from it
    result = s.close(NoPipeline())
    assert not [x for x in result.settled if x.doubt.endswith(SP)]


# --- C: asking back ----------------------------------------------------------------------------


def test_asking_back_gets_what_the_vault_has_then_the_question_again(denv: Env) -> None:
    plan, infos = plan_for(denv)
    plan = [plan[step_of(plan, FRAUD)], plan[-1]]
    llm = ScriptedLLM(
        decisions=[label("asked"), label("answered")],
        decider=settle_with(same=("same", 0.96)),
    )
    question = "Are fraud alerts and transaction monitoring the same project?"
    llm.structured_queue.extend([msg(question), msg("Anything else?")])
    s = session(denv, llm, plan, infos)
    s.opening()
    reply = s.reply(ASK_BACK)
    assert reply.event == "context"
    assert reply.message == (
        "Here's what I have: Fraud alerts: A rules engine that flags suspicious card payments "
        "(from your document “homepage”); Transaction monitoring: Screens payments for fraud "
        "patterns in real time (from your document “homepage”). " + question
    )
    assert len([c for c in llm.calls if c[0] == "structured"]) == 1  # no model call for it
    assert s.reply("Yes, same thing, it was renamed.").message == "Anything else?"
    result = s.close(NoPipeline())
    [merge] = result.settled
    assert merge.action == "applied" and merge.ops[0].keep == FRAUD
    assert TM not in Graph(denv.vault, TODAY).pages


def test_asking_back_twice_moves_on(denv: Env) -> None:
    plan, infos = plan_for(denv)
    plan = [plan[step_of(plan, FRAUD)], plan[-1]]
    llm = ScriptedLLM(decisions=[label("asked"), label("asked")])
    llm.structured_queue.extend([msg("Same project?"), msg("Anything else?")])
    s = session(denv, llm, plan, infos)
    s.opening()
    assert s.reply("What do you mean?").event == "context"
    assert s.reply("Why do you ask?").message == "Anything else?"


# --- E: settling -------------------------------------------------------------------------------


def _answered(denv: Env, entity: str, answer: str, **settle: tuple[str, float]):
    plan, infos = plan_for(denv)
    plan = [plan[step_of(plan, entity)], plan[-1]]
    llm = ScriptedLLM(decisions=[label("answered")], decider=settle_with(**settle))
    llm.structured_queue.extend([msg("What is it to you?"), msg("Anything else?")])
    s = session(denv, llm, plan, infos)
    s.opening()
    s.reply(answer)
    return s.close(NoPipeline()), plan[0]


def test_a_doubtful_mapping_is_queued_for_review(denv: Env) -> None:
    result, item = _answered(
        denv,
        SP,
        "A project, I think at Fabrikam?",
        kind=("project", 0.93),
        part_of=("Fabrikam Analytics", 0.7),
    )
    by = {x.doubt: x for x in result.settled}
    assert by[f"part_of {SP}"].action == "queued"
    [fact] = by[f"part_of {SP}"].ops
    assert fact.kind == "inferred" and fact.evidence[0] == result.transcript
    assert list((denv.vault.reviews_pending_dir).glob("*.md"))
    assert not any(r.fact.p == "part_of" for r in Graph(denv.vault, TODAY).facts_of(SP))


def test_a_non_answer_leaves_the_items_open(denv: Env) -> None:
    result, item = _answered(
        denv, SP, "Hard to say.", kind=("unclear", 0.8), part_of=("unclear", 0.9)
    )
    assert {x.action for x in result.settled} == {"open"}
    open_ids = {i.id for i in read_questions(denv.vault) if i.open}
    assert set(item.question_ids) <= open_ids


def test_a_part_of_answer_is_ignored_when_it_is_not_a_project(denv: Env) -> None:
    result, _ = _answered(
        denv,
        SP,
        "More a topic, I do it at Northwind.",
        kind=("topic", 0.7),
        part_of=("Northwind Labs", 0.95),
    )
    by = {x.doubt: x for x in result.settled}
    assert by[f"part_of {SP}"].action == "open"
    assert by[f"type {SP}"].action == "kept"  # has a fact of its own: not demoted, logged
    assert SP in Graph(denv.vault, TODAY).pages


def test_settle_questions_put_the_escape_option_first() -> None:
    assert kind_question("X").choices[0].value == "unclear"
    assert [c.value for c in part_of_question("X", ["A", "B"]).choices][:2] == [
        "unclear",
        "on their own",
    ]
    assert same_question("A", "B", "project").choices[0].value == "unclear"


def test_the_transcript_marks_the_context_reply_as_the_brain(denv: Env) -> None:
    plan, infos = plan_for(denv)
    plan = [plan[step_of(plan, FRAUD)], plan[-1]]
    llm = ScriptedLLM(decisions=[label("asked")])
    llm.structured_queue.extend([msg("Same project?")])
    s = session(denv, llm, plan, infos)
    s.opening()
    s.reply(ASK_BACK)
    assert s.transcript is not None
    _, turns = parse_transcript(s.transcript.path.read_text(encoding="utf-8"))
    assert [t.role for t in turns][-2:] == ["user", "assistant"]
    assert turns[-1].text.startswith("Here's what I have:")


def test_nothing_to_add_is_no_refusal(denv: Env) -> None:
    """Only a refusal ("I'd rather not say") makes later questions candidates for "declined";
    "nothing more on that" does not (the refusal predicates are not even asked)."""
    plan, infos = plan_for(denv)
    plan = [plan[step_of(plan, SP)], plan[step_of(plan, FRAUD)], plan[-1]]
    llm = ScriptedLLM(decisions=[label("declined"), label("refused")])
    llm.structured_queue.extend([msg("Q one?"), msg("Q two?"), msg("Q three?")])
    s = session(denv, llm, plan, infos)
    s.opening()
    s.reply("Nothing more on that for now.")
    [link] = [c[1]["questions"] for c in llm.calls if c[0] == "link"]
    assert [q.name for q in link] == ["q0"]  # no refusal yet: no "asks again?" predicate
    assert s.declined_lines() == []
    s.reply(DECLINE)
    assert len(s.declined_lines()) == 1 and s.declines == 2


def test_the_plan_questions_are_not_in_the_profile_card(denv: Env) -> None:
    assert "To ask:" in denv.vault.core_path.read_text(encoding="utf-8")
    s = session(denv, ScriptedLLM())
    assert "To ask:" not in s.system  # they are this plan's steps, asked in order


def test_a_project_placed_at_a_job_settles_its_type_doubt(denv: Env) -> None:
    result, item = _answered(
        denv, SP, SP_ANSWER, kind=("project", 0.7), part_of=("Fabrikam Analytics", 0.95)
    )
    by = {x.doubt: x.action for x in result.settled}
    assert by == {f"type {SP}": "kept", f"part_of {SP}": "applied"}
    assert set(item.question_ids) <= {i.id for i in read_questions(denv.vault) if i.done}
