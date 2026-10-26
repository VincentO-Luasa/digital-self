"""Helpers for the interview tests: the fictional vault plus tests/fixtures/interview/ pages,
indexed with a fake model (coverage reads only the graph tables, so no server is needed)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from brain.index.db import open_index
from brain.index.reindex import reindex
from brain.interview.coverage import CoverageReport, coverage_report
from brain.llm.fake import FakeLLM
from brain.vault import gitops
from brain.vault.generated import refresh_generated
from brain.vault.store import Vault

EXTRA = Path(__file__).resolve().parents[1] / "fixtures" / "interview"
TODAY = date(2026, 10, 26)
LATER = date(2027, 1, 10)  # Atlas status (30 days) and deadline (60 days) are stale by then


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
