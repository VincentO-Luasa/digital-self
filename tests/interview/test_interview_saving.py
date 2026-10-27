"""Saving an interview as it goes: finished blocks become memory during the interview (the real
memory pipeline with a scripted model), the close handles only the rest; idempotent, private
answers stay private, a crash is recovered by ``--pending``. Fictional persona only."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from interview_helpers import TODAY, Clock, ScriptedLLM, at  # type: ignore[import-not-found]

from brain.config import Settings
from brain.interview.planner import PlanItem
from brain.interview.saver import MAX_INTERRUPTS, BlockSaver, Gate, Job, JobResult, Yielding
from brain.interview.session import InterviewSession
from brain.llm import CallCancelled
from brain.llm.decide import Decision
from brain.llm.fake import FakeLLM
from brain.memory import is_processed, pending_transcripts, process_session
from brain.memory.pipeline import segmented_through

ATLAS = "projects/atlas-migration"
DASH = "projects/legacy-dashboards"
REGION = "projects"


def step(entity: str | None, slot: str, hint: str) -> PlanItem:
    return PlanItem(REGION, "ask", hint, "Projects", entity=entity, slot=slot)


PLAN = [
    step(ATLAS, "status", "Where does Atlas stand?"),
    step(ATLAS, "role", "What is your part in Atlas?"),
    step(ATLAS, "deadline", "When is Atlas due?"),
    step(DASH, "status", "What happened to the dashboards?"),
    step(DASH, "role", "Who looks after them now?"),
    PlanItem(REGION, "discover", "Any other projects?", "Projects", slot="more"),
]
ANSWERS = [
    "Atlas is in the parallel run until mid-November; the scheduler moved to Kiln.",
    "I lead the cut-over of the nightly batch.",
    "The go-live is planned for the end of November.",
    "The dashboards were retired last year.",
    "Nobody, they are gone.",
    "Nothing else right now.",
]


def decider(text: str, questions) -> list[Decision]:  # noqa: ANN001
    out = []
    for q in questions:
        if q.name == "personal":
            out.append(
                Decision(name=q.name, type="predicate", choice="yes", probabilities={"yes": 0.99})
            )
        else:
            out.append(Decision(name=q.name, type=q.type, refusal=True))
    return out


def segmentation(first: int, last: int, topic: str) -> dict[str, Any]:
    episode = {
        "first_turn": first,
        "last_turn": last,
        "topic": topic,
        "regions": [REGION],
        "mentions": [],
        "gist": f"Alex on {topic}",
        "summary": f"Alex talks about {topic}.",
        "emotion": "none",
        "goal": "none",
        "future": False,
        "remember": False,
        "change": "none",
    }
    return {"reasoning": "r", "episodes": [episode]}


def extraction(subject: str, obj: str, predicate: str = "status") -> dict[str, Any]:
    fact = {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "text": "",
        "start": "",
        "end": "",
        "label": "new",
        "known": "",
    }
    return {"reasoning": "r", "entities": [], "facts": [fact], "observations": [], "intentions": []}


ATLAS_PART = [segmentation(1, 9, "atlas status"), extraction("Atlas Migration", "Kiln", "uses")]
REST = [segmentation(10, 99, "dashboards"), extraction("Legacy Dashboards", "retired")]


def interviewer(n: int = len(ANSWERS)) -> ScriptedLLM:
    answered = [
        {
            "name": "answer",
            "type": "choice",
            "choice": "answered",
            "probabilities": {"answered": 1},
        },
        {"name": "new_thread", "type": "predicate", "choice": "no", "probabilities": {"yes": 0}},
    ]
    llm = ScriptedLLM(decisions=[answered] * n)
    llm.structured_queue.extend({"reasoning": "r", "message": item.hint} for item in PLAN)
    return llm


def memory(*responses: Any) -> FakeLLM:
    return FakeLLM(structured_responses=list(responses), decider=decider)


@pytest.fixture
def make(env):
    def _make(mem: FakeLLM, **kw: Any) -> InterviewSession:
        report = env.report(TODAY)
        return InterviewSession(
            env.vault,
            kw.pop("llm", None) or interviewer(),
            report,
            REGION,
            list(PLAN),
            Settings(vault=env.vault.root),
            clock=Clock(at(TODAY)),
            process=process_session,
            memory_llm=mem,
            background=kw.pop("background", False),
            **kw,
        )

    return _make


def structured(llm: FakeLLM) -> int:
    return sum(c[0] == "structured" for c in llm.calls)


def episodes(env, session: InterviewSession) -> list:
    assert session.transcript is not None
    sid = session.transcript.session_id
    return [e for e in env.vault.episodes() if e.session == sid]


def test_a_finished_block_is_saved_during_the_interview(make, env) -> None:
    mem = memory(*ATLAS_PART, *REST)
    session = make(mem)
    session.opening()
    for answer in ANSWERS[:3]:
        session.reply(answer)
    # the Atlas block ended after three answers: saved before the next question was answered
    [first] = episodes(env, session)
    assert first.turns == (1, 9) and first.status == "extracted"
    assert first.entities[:1] == [ATLAS]
    assert segmented_through(env.vault, session.transcript) == 9
    assert "segmented turns 1-9: 1 episode(s)" in env.vault.log_path.read_text()
    atlas = env.vault.page(ATLAS)
    assert any(f.p == "uses" and f.o == "Kiln" for f in atlas.facts)
    for answer in ANSWERS[3:]:
        reply = session.reply(answer)
    assert reply.done and reply.event == "end"
    assert structured(mem) == 2  # the Dashboards block is short: carried on to the close

    result = session.close(db=env.db)
    assert result.error is None and len(result.parts) == 1
    assert [len(p.job.turns) for p in result.parts] == [9]
    report = result.report
    assert report.status == "processed" and len(report.episodes) == 2
    assert "2 episodes" in report.summary()
    second = [e for e in episodes(env, session) if e.turns[0] >= 10]
    assert len(second) == 1 and second[0].status == "extracted"
    assert structured(mem) == 4  # each part segmented and extracted once
    assert is_processed(env.vault, session.transcript)

    again = memory()
    rerun = process_session(env.vault, again, session.transcript, channel="interview")
    assert rerun.status == "already processed" and not again.calls


def test_a_crash_after_a_saved_block_is_recovered_by_pending(make, env) -> None:
    session = make(memory(*ATLAS_PART))
    session.opening()
    for answer in ANSWERS[:4]:
        session.reply(answer)
    # the process dies here: no close
    assert session.transcript is not None
    session.transcript.commit()
    assert [t.session_id for t in pending_transcripts(env.vault)] == [session.transcript.session_id]
    mem = memory(*REST)
    report = process_session(env.vault, mem, session.transcript, channel="interview")
    assert report.status == "processed" and len(report.episodes) == 1
    assert report.episodes[0].id != episodes(env, session)[0].id
    assert structured(mem) == 2  # only the part after the saved block
    assert is_processed(env.vault, session.transcript)
    assert not pending_transcripts(env.vault)


def test_a_failed_part_is_retried_at_close(make, env) -> None:
    broken = ValueError("cut off")
    mem = memory(segmentation(1, 9, "atlas status"), broken, *REST, ATLAS_PART[1])
    session = make(mem)
    session.opening()
    for answer in ANSWERS:
        session.reply(answer)
    [ep] = episodes(env, session)
    assert ep.status == "pending"  # extraction failed during the interview
    result = session.close(db=env.db)
    assert all(e.status == "extracted" for e in episodes(env, session))
    assert len(episodes(env, session)) == 2
    assert result.report.status == "processed"
    assert not [e for e in result.report.episodes if e.error]


def test_a_private_answer_inside_a_saved_block(make, env) -> None:
    mem = memory(
        segmentation(1, 10, "atlas status"),
        ATLAS_PART[1],
        *REST,
    )
    session = make(mem)
    session.opening()
    session.reply(ANSWERS[0])
    session.reply("private: " + ANSWERS[1])
    session.reply(ANSWERS[2])
    [ep] = episodes(env, session)
    assert ep.privacy == "private"  # the block holds a private answer
    atlas = env.vault.page(ATLAS)
    fact = next(f for f in atlas.facts if f.o == "Kiln")
    assert fact.extra.get("privacy") == "private"
    # later questions never saw it; the saver got the turn as private
    assert session.private_turns == [7]
    assert "cut-over" not in "\n".join(m["content"] for m in session.history)


def test_the_worker_runs_in_the_background(make, env) -> None:
    threads: list[str] = []

    def process(vault, llm, transcript, **kwargs):  # noqa: ANN001, ANN003, ANN202
        threads.append(threading.current_thread().name)
        return process_session(vault, llm, transcript, **kwargs)

    mem = memory(*ATLAS_PART, *REST)
    session = make(mem, background=True)
    session.process = process
    session.memory_llm.grace = 0.0  # no need to wait for a pause in a test
    session.opening()
    for answer in ANSWERS[:3]:
        session.reply(answer)
    deadline = time.monotonic() + 5  # the owner pauses: the worker saves the block
    while not session.saver.results and time.monotonic() < deadline:
        time.sleep(0.01)
    for answer in ANSWERS[3:]:
        session.reply(answer)
    result = session.close(db=env.db)
    assert threads[0] == "interview-saver" and threads[-1] == threading.current_thread().name
    assert len(result.parts) == 1 and result.parts[0].error is None
    assert len(episodes(env, session)) == 2


def test_a_block_that_never_got_a_pause_is_left_to_the_close(make, env) -> None:
    whole = segmentation(1, 9, "atlas status")
    whole["episodes"].append(segmentation(10, 99, "dashboards")["episodes"][0])
    mem = memory(whole, ATLAS_PART[1], REST[1])
    session = make(mem, background=True)  # the owner answers at once: no pause of 2 s
    session.opening()
    for answer in ANSWERS:
        session.reply(answer)
    result = session.close(db=env.db)
    assert result.parts == []  # the job gave up before its first model call
    assert [e.turns[0] for e in episodes(env, session)] == [1, 10]
    assert structured(mem) == 3  # one segmentation for everything, then the extractions
    assert is_processed(env.vault, session.transcript)


def test_the_worker_yields_while_the_interview_works() -> None:
    gate = Gate()
    fake = FakeLLM(decider=decider)
    worker = Yielding(fake, gate)
    started = threading.Event()
    done = threading.Event()

    def call() -> None:
        started.set()
        worker.decide("text", [])
        done.set()

    with gate.busy():
        thread = threading.Thread(target=call)
        thread.start()
        started.wait()
        time.sleep(0.05)
        assert not done.is_set()  # waiting for the interview
    thread.join(timeout=2)
    assert done.is_set() and worker.calls == 1 and worker.waited > 0


def test_saver_errors_never_reach_the_interview() -> None:
    def boom(job: Job) -> JobResult:
        raise RuntimeError("disk full")

    saver = BlockSaver(boom, background=True)
    saver.submit(Job([], []))
    saver.drain()
    assert saver.errors == ["RuntimeError: disk full"]


def test_a_pipeline_without_turns_saves_at_close_only(env) -> None:
    calls: list[dict] = []

    def old(vault, llm, transcript, *, channel, regions_hint=None, today=None):  # noqa: ANN001, ANN202
        calls.append({"channel": channel})
        return None

    session = InterviewSession(
        env.vault,
        interviewer(),
        env.report(TODAY),
        REGION,
        list(PLAN),
        Settings(vault=env.vault.root),
        clock=Clock(at(TODAY)),
        process=old,
    )
    assert session.saver is None
    session.opening()
    for answer in ANSWERS:
        session.reply(answer)
    session.close()
    assert len(calls) == 1


def test_the_worker_waits_for_a_pause_and_starts_long_calls_early() -> None:
    gate = Gate()
    with gate.busy():
        pass
    start = time.monotonic()
    assert gate.wait(grace=0.2) >= 0.15  # the owner has been idle for less than the grace
    assert time.monotonic() - start < 1
    time.sleep(0.15)
    got: list[float] = []
    thread = threading.Thread(target=lambda: got.append(gate.wait(latest=0.1)))
    thread.start()  # too late in this pause for a long call: waits for the next one
    time.sleep(0.1)
    assert not got
    with gate.busy():
        pass
    thread.join(timeout=2)
    assert got
    blocked = threading.Thread(target=lambda: got.append(gate.wait(latest=0.0)))
    time.sleep(0.05)
    blocked.start()
    gate.release()  # close: no more pauses to wait for
    blocked.join(timeout=2)
    assert len(got) == 2


def test_close_drops_the_blocks_still_queued() -> None:
    running = threading.Event()
    release = threading.Event()

    def work(job: Job) -> JobResult:
        running.set()
        release.wait(2)
        return JobResult(job)

    saver = BlockSaver(work, background=True)
    saver.submit(Job([], [], label="first"))
    saver.submit(Job([], [], label="second"))
    running.wait(2)
    threading.Timer(0.1, release.set).start()
    dropped = saver.drain(cancel=True)
    assert [r.job.label for r in saver.results] == ["first"]
    assert [j.label for j in dropped] == ["second"]


def test_a_long_call_is_stopped_when_the_owner_answers_and_repeated() -> None:
    gate = Gate()
    seen: list[bool] = []

    class Slow:
        def structured(self, messages, schema, *, cancel=None):  # noqa: ANN001, ANN201
            seen.append(cancel is not None)
            if len(seen) == 1:
                with gate.busy():  # the owner answers while the call runs
                    pass
                if cancel.is_set():
                    raise CallCancelled
            return "done"

    worker = Yielding(Slow(), gate)
    assert worker.structured([], None) == "done"
    assert seen == [True, True] and worker.interrupted == 1

    class Long:  # longer than every pause: stopped MAX_INTERRUPTS times, then left to finish
        def structured(self, messages, schema, *, cancel=None):  # noqa: ANN001, ANN201
            seen.append(cancel is not None)
            if cancel is None:
                return "done"
            with gate.busy():
                pass
            raise CallCancelled

    seen.clear()
    worker = Yielding(Long(), gate)
    assert worker.structured([], None) == "done"
    assert seen == [True] * MAX_INTERRUPTS + [False]
    assert worker.interrupted == MAX_INTERRUPTS
