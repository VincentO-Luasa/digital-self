"""Against the local model, with scripted fictional owners: does the interview end on content
(a terse owner is stopped, a talkative one never early), and what does saving as it goes cost?
``uv run pytest -m llm tests/interview/test_interview_saving_live.py -s``.

Modes (talkative owner, real memory pipeline at close):

- ``close``: today's behaviour, every block processed at close; the owner answers at once;
- ``fast``: blocks saved during the interview; the owner answers at once (the worst case: the
  worker never gets a pause);
- ``typing``: blocks saved during the interview; the owner takes 10–20 s to answer;
- ``fast-naive`` / ``typing-naive``: the same, but the worker starts every call as soon as the
  interview is idle (no grace, long calls at any time).

Per run: seconds per turn (mean, p90, max), seconds at close, parts saved during the interview,
how long the worker yielded, facts and episodes. ``BRAIN_SAVING_SAMPLES`` (default 2),
``BRAIN_SAVING_MODES`` (default ``close,fast,typing``), ``BRAIN_SAVING_TYPING`` ("10,20").
"""

from __future__ import annotations

import os
import random
import shutil
import statistics
import time
from pathlib import Path

import pytest
from interview_helpers import TODAY, build_env  # type: ignore[import-not-found]

from brain.config import Settings, load_settings
from brain.index.db import open_index
from brain.index.reindex import reindex
from brain.interview.planner import PlanItem
from brain.interview.session import InterviewSession
from brain.llm import LLM
from brain.memory import is_processed, process_session
from brain.vault import gitops

pytestmark = pytest.mark.llm
SAMPLES = int(os.environ.get("BRAIN_SAVING_SAMPLES", "2"))
MODES = os.environ.get("BRAIN_SAVING_MODES", "close,fast,typing").split(",")
TYPING = tuple(float(x) for x in os.environ.get("BRAIN_SAVING_TYPING", "10,20").split(","))
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REGION = "projects"
ATLAS, DASH = "projects/atlas-migration", "projects/legacy-dashboards"
MAX_TURNS = 30


def ask(block: str, entity: str | None, slot: str, hint: str, mode: str = "ask") -> PlanItem:
    topic = {ATLAS: "Projects: Atlas Migration", DASH: "Projects: Legacy Dashboards"}.get(
        entity or "", f"Projects: {block}"
    )
    return PlanItem(REGION, mode, hint, topic, entity=entity, slot=slot, block=block)  # type: ignore[arg-type]


PLAN = [
    ask("atlas", ATLAS, "status", "Where does Atlas Migration stand right now?", "confirm"),
    ask("atlas", ATLAS, "role", "What is your own part in Atlas Migration?"),
    ask("atlas", ATLAS, "team", "Who else works on Atlas Migration with you?"),
    ask("atlas", ATLAS, "deadline", "Is there a deadline for Atlas Migration?", "confirm"),
    ask("dash", DASH, "what", "What were the Legacy Dashboards for?"),
    ask("dash", DASH, "why", "Why were the Legacy Dashboards retired?"),
    ask("dash", DASH, "lessons", "What did you learn from the Legacy Dashboards?"),
    ask("tidepool", None, "what", "Are you working on anything outside work, a side project?"),
    ask("tidepool", None, "who", "Who works on it with you?"),
    ask("tidepool", None, "next", "What's next for it?"),
    PlanItem(
        REGION, "discover", "Any other projects I should know about?", "Projects", slot="more"
    ),
]

TALK = {
    ("atlas", "status"): "We're in the parallel run now: the old and the new warehouse both get "
    "the nightly batch, and we compare the outputs every morning before standup.",
    ("atlas", "role"): "I own the cut-over plan and the reconciliation jobs, and I'm the one who "
    "gets paged when the comparison fails at six in the morning.",
    ("atlas", "team"): "Sam Okafor's team does the infrastructure, and Maya Lindqvist joined in "
    "September to run the data quality checks; Theo Marchetti sponsors it.",
    ("atlas", "deadline"): "Go-live moved to the first week of December because the finance "
    "reports needed another round of fixes, but everyone agrees it will hold.",
    ("dash", "what"): "They were the reporting dashboards for the sales team, built on an old "
    "BI server that nobody wanted to patch any more.",
    ("dash", "why"): "Licensing got too expensive and the queries were too slow, so we moved "
    "the important views into Atlas and dropped the rest.",
    ("dash", "lessons"): "Mostly that people only use five of fifty charts, so I now ask who "
    "reads a report before building it, and I keep the queries in version control.",
    ("tidepool", "what"): "Yes, Tidepool: a small app that tracks tide times for the beach "
    "near my place, I write it in Rust on weekends.",
    ("tidepool", "who"): "Just me and my friend Sam Okafor, he does the design and I do the "
    "backend, we meet on Sunday mornings.",
    ("tidepool", "next"): "We want to put it on the app store before summer and add surf "
    "forecasts, which means learning a bit of Swift.",
    ("more", "more"): "Not right now, those are the main ones; I'm also reading about stream "
    "processing but that's not a project yet.",
}
FOLLOW = {
    "atlas": [
        "The hardest part is the finance tables: they change shape every quarter, so the "
        "comparison keeps flagging false differences I have to explain.",
        "We keep a list of known differences in a YAML file and the reconciliation job skips "
        "them; it's ugly but it works.",
        "Sam's team moved the scheduler to Kiln in August, which made the nightly runs about "
        "twice as fast.",
        "Theo checks in every Friday; he mostly wants to know whether the finance team is happy.",
        "Maya found three tables where the old warehouse was silently wrong for years.",
        "If it slips again we lose the budget for the second phase, so there's some pressure.",
    ],
    "dash": [
        "The sales team complained for a week and then forgot; the new views in Atlas load in "
        "seconds instead of minutes.",
        "The BI server ran on a machine under someone's desk in the Lisbon office, believe it or "
        "not.",
        "I built most of them in my first year at Northwind Labs, so retiring them felt odd.",
        "We kept two dashboards alive for the board meeting and rebuilt them in a notebook.",
    ],
    "tidepool": [
        "It started because the official tide site is terrible on a phone; about twenty friends "
        "use it already.",
        "The tide data comes from a public port authority feed that updates every hour.",
        "Sam drew the icons on his tablet during a long train ride to Porto.",
        "The surf forecast part needs a paid weather API, so we're looking for a free one first.",
    ],
    None: [
        "That's about it, it's been a busy autumn with all of that going on.",
        "I'd like to write more about what I learn, maybe a small blog, but no time yet.",
    ],
}
TERSE = ["Fine.", "Dunno.", "Not much really.", "Same as before.", "No idea.", "Meh.", "Not sure."]


class Talkative:
    def __init__(self) -> None:
        self.told: dict[str | None, int] = {}

    def answer(self, session: InterviewSession, message: str) -> str:
        if "keep going" in message.casefold():
            return "Yes, keep going."
        item = session.item
        if session.mode in ("follow_up", "clarify") or item.slot not in {k[1] for k in TALK}:
            block = item.block if item.block in FOLLOW else None
            n = self.told.get(block, 0)
            self.told[block] = n + 1
            return FOLLOW[block][n % len(FOLLOW[block])]
        return TALK.get((item.block or "more", item.slot or "more"), FOLLOW[None][0])


class Terse:
    def __init__(self) -> None:
        self.n = 0

    def answer(self, session: InterviewSession, message: str) -> str:
        self.n += 1
        return TERSE[(self.n - 1) % len(TERSE)]


@pytest.fixture
def client():
    llm = LLM(load_settings().llm)
    if not llm.health():
        llm.close()
        pytest.skip("local model server is not running")
    yield llm
    llm.close()


def make_env(root: Path, client: LLM):
    shutil.copytree(FIXTURES / "vault", root)
    gitops.commit_all(root, "fixture vault")
    env = build_env(root, TODAY)
    db = open_index(
        root.parent / f"{root.name}.sqlite", embed_model=client.embed_model, dim=client.embed_dim
    )
    reindex(env.vault, db, client)
    return env, db


def run(root: Path, client: LLM, owner, mode: str, *, memory: bool = True) -> dict:
    env, db = make_env(root, client)
    session = InterviewSession(
        env.vault,
        client,
        env.report(TODAY),
        REGION,
        list(PLAN),
        Settings(vault=env.vault.root),
        minutes=60,
        process=process_session if memory else None,
        save_blocks=mode != "close",
    )
    if mode.endswith("-naive") and session.saver is not None:  # every call as soon as idle
        session.memory_llm.grace, session.memory_llm.latest = 0.0, None
    rng = random.Random(len(str(root)))
    lines, turn_s, collided = [], [], []
    reply = session.opening()
    lines.append(f"brain › {reply.message}")
    for _ in range(MAX_TURNS):
        if reply.done:
            break
        answer = owner.answer(session, reply.message)
        if mode.startswith("typing"):
            time.sleep(rng.uniform(*TYPING))
        lines.append(f"you   › {answer}")
        busy = getattr(session.memory_llm, "active", None)  # a worker call in flight
        mark = len(session.timings)
        start = time.perf_counter()
        reply = session.reply(answer)
        turn_s.append(time.perf_counter() - start)
        collided.append(busy)
        parts = ", ".join(f"{k} {s:.1f}" for k, s in session.timings[mark:])
        lines.append(
            f"brain › {reply.message}  [{reply.event}, {turn_s[-1]:.1f}s: {parts}"
            + (f"; worker in {busy}" if busy else "")
            + "]"
        )
    if not reply.done:
        session.reply("stop")
    worker = session.memory_llm
    out = {
        "lines": lines,
        "event": reply.event,
        "answers": session.answers,
        "asked_steps": len(session.asked_steps),
        "turn_s": turn_s,
        "collided": collided,
        "kinds": {
            k: statistics.mean([s for what, s in session.timings if what == k] or [0])
            for k in ("classify", "link", "compose")
        },
        "waited": getattr(worker, "waited", 0.0),
        "interrupted": getattr(worker, "interrupted", 0),
        "worker_calls": getattr(worker, "calls", 0),
        "worker_s": getattr(worker, "seconds", 0.0),
        "next": reply.next_region,
    }
    if memory:
        result = session.close(db=db)
        out["close_s"] = result.seconds
        out["parts"] = [(len(p.job.turns), round(p.seconds, 1), p.error) for p in result.parts]
        report = result.report
        out["summary"] = report.summary() if report is not None else result.error
        out["episodes"] = len(report.episodes) if report is not None else 0
        out["processed"] = is_processed(env.vault, session.transcript)
    db.close()
    env.db.close()
    return out


def p90(xs: list[float]) -> float:
    return sorted(xs)[max(0, int(round(0.9 * len(xs))) - 1)] if xs else 0.0


def test_ending_on_content_live(tmp_path: Path, git_env: None, client: LLM) -> None:
    """A terse owner is stopped on content; a talkative one is not (no memory at close)."""
    stops = {"terse": [], "talkative": []}
    for n in range(SAMPLES):
        for name, owner in (("terse", Terse()), ("talkative", Talkative())):
            r = run(tmp_path / f"{name}-{n}", client, owner, "close", memory=False)
            stops[name].append((r["event"], r["answers"]))
            print(f"\n=== {name} {n + 1}: {r['event']} after {r['answers']} answers ===")
            print("\n".join(r["lines"]))
    print(f"\nstops: {stops}")
    assert all(event in ("content", "declined") for event, _ in stops["terse"])
    assert all(event in ("end", "limit") for event, _ in stops["talkative"])  # never on content


def test_saving_as_you_go_live(tmp_path: Path, git_env: None, client: LLM) -> None:
    results: dict[str, list[dict]] = {m: [] for m in MODES}
    for n in range(SAMPLES):
        for mode in MODES:
            r = run(tmp_path / f"{mode}-{n}", client, Talkative(), mode)
            results[mode].append(r)
            t = r["turn_s"]
            print(f"\n=== {mode} sample {n + 1}: {r['event']} after {r['answers']} answers ===")
            print("\n".join(r["lines"]))
            print(
                f"  per turn mean {statistics.mean(t):.2f}s p90 {p90(t):.2f}s max {max(t):.2f}s"
                f" · close {r['close_s']:.1f}s · parts {r['parts']} · worker calls "
                f"{r['worker_calls']} ({r['worker_s']:.0f}s, yielded {r['waited']:.0f}s, {r['interrupted']} interrupted) · "
                f"{r['summary']} · processed {r['processed']}"
            )
            hit = [s for s, c in zip(t, r["collided"], strict=True) if c]
            clean = [s for s, c in zip(t, r["collided"], strict=True) if not c]
            kinds = ", ".join(f"{k} {v:.2f}s" for k, v in r["kinds"].items())
            print(
                f"  turns meeting a worker call: {len(hit)} (mean "
                f"{statistics.mean(hit or [0]):.2f}s) · others mean "
                f"{statistics.mean(clean or [0]):.2f}s · {kinds}"
            )
    print("\n=== summary ===")
    for mode, rs in results.items():
        turns = [s for r in rs for s in r["turn_s"]]
        print(
            f"{mode:7} turns {len(turns)} mean {statistics.mean(turns):.2f}s p90 {p90(turns):.2f}s "
            f"max {max(turns):.2f}s · close {[round(r['close_s'], 1) for r in rs]} · episodes "
            f"{[r['episodes'] for r in rs]} · {[r['summary'] for r in rs]}"
        )
    for rs in results.values():
        for r in rs:
            assert r["processed"] and r["event"] in ("end", "limit")
