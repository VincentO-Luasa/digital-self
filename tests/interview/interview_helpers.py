"""Helpers for the interview tests: the fictional vault plus tests/fixtures/interview/ pages,
indexed with a fake model (coverage reads only the graph tables, so no server is needed)."""

from __future__ import annotations

import shutil
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from brain.index.db import open_index
from brain.index.reindex import reindex
from brain.interview.coverage import CoverageReport, coverage_report
from brain.interview.session import LINK_CHOICES
from brain.llm.decide import Decision, refusals
from brain.llm.fake import FakeLLM
from brain.vault import gitops
from brain.vault.generated import refresh_generated
from brain.vault.store import Vault

EXTRA = Path(__file__).resolve().parents[1] / "fixtures" / "interview"
TODAY = date(2026, 10, 26)
LATER = date(2027, 1, 10)
LINK_VALUES = [
    c.value for c in LINK_CHOICES
]  # Atlas status (30 days) and deadline (60 days) are stale by then


class Clock:
    """Advances ``step`` seconds per call, from ``start``."""

    def __init__(self, start: datetime, step: float = 5.0) -> None:
        self.now = start
        self.step = step

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=self.step)
        return self.now


def at(day: date, hour: int = 20) -> datetime:
    return datetime(day.year, day.month, day.day, hour)


@dataclass
class Env:
    vault: Vault
    db: object

    def report(self, today: date = TODAY) -> CoverageReport:
        return coverage_report(self.db, self.vault.lifemap, today)  # type: ignore[arg-type]

    def reindex(self) -> None:
        reindex(self.vault, self.db, FakeLLM())  # type: ignore[arg-type]


def build_env(root: Path, today: date = TODAY) -> Env:
    shutil.copytree(EXTRA / "wiki", root / "wiki", dirs_exist_ok=True)
    gitops.commit_all(root, "interview fixtures")
    vault = Vault(root)
    refresh_generated(vault, today=today)
    llm = FakeLLM()
    db = open_index(vault.db_path, embed_model=llm.embed_model, dim=llm.embed_dim)
    reindex(vault, db, llm)
    return Env(vault, db)


# --- a model for scripted interviews -----------------------------------------------------------


def choice(name: str, value: str, p: float = 1.0, values: list[str] | None = None) -> Decision:
    probs = {v: (p if v == value else (1 - p) / max(1, len(values) - 1)) for v in values or []}
    probs[value] = p
    return Decision(name=name, type="choice", choice=value, probabilities=probs, confidence=p)


class ScriptedLLM(FakeLLM):
    """FakeLLM whose scripted ``decisions`` answer only the answer classification; linking
    decisions (questions ``q0``, ``q1``…) go to ``link`` (default: every step still open) and
    any other decision (settling a doubt) to ``decider``."""

    def __init__(self, *args, link=None, **kwargs) -> None:  # noqa: ANN001, ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.link = link
        self.linked: list[tuple[str, list[str]]] = []  # (text, questions' instructions)

    def decide(self, text: str, questions):  # noqa: ANN001, ANN201
        names = [q.name for q in questions]
        if names and all(n[:1] in "qr" and n[1:].isdigit() for n in names):
            asked = [q for q in questions if q.name[0] == "q"]
            refusals_asked = any(q.name[0] == "r" for q in questions)
            self.calls.append(("link", {"text": text, "questions": list(questions)}))
            self.linked.append((text, [q.instructions for q in asked]))
            statuses = self.link(text, asked) if self.link else ["open"] * len(asked)
            out = []
            for q, st in zip(asked, statuses, strict=True):
                out.append(
                    choice(q.name, "answered" if st == "answered" else "open", 0.95, LINK_VALUES)
                )
                if refusals_asked:
                    p = 0.9 if st == "declined" else 0.05
                    out.append(
                        Decision(
                            name=f"r{q.name[1:]}",
                            type="predicate",
                            choice="yes" if p >= 0.5 else "no",
                            probabilities={"yes": p, "no": 1 - p},
                        )
                    )
            return out
        if names == ["answer", "new_thread"]:
            if self.decision_queue:
                return super().decide(text, questions)
            return super().decide(text, questions) if self.decider else refusals(questions)
        if self.decider is None:
            self.calls.append(("decide", {"text": text, "questions": list(questions)}))
            return refusals(questions)
        queue, self.decision_queue = self.decision_queue, deque()
        try:
            return super().decide(text, questions)
        finally:
            self.decision_queue = queue
