"""Ingest the fictional homepage with the real local model. ``uv run pytest -m llm``.

Set ``BRAIN_RECORD_FIXTURES=1`` to (re-)record the replay fixture in
``tests/fixtures/llm/ingest/`` (needed whenever the extraction prompt or schema changes).
"""

from __future__ import annotations

import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from conftest import (  # type: ignore[import-not-found]
    DOCS,
    portfolio_problems,
    quality_problems,
)

from brain.config import load_settings
from brain.index.db import open_index
from brain.ingest import ingest
from brain.llm import LLM
from brain.llm.recorder import Recorder
from brain.vault.store import Vault

pytestmark = pytest.mark.llm

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "llm" / "ingest"
TODAY = date(2026, 10, 25)


class OrderedRecorder(Recorder):
    """Keep each call's kind and token usage; when recording, save fixtures prefixed with the
    call number so replay keeps the order."""

    def __init__(self, directory: Path, *, write: bool, prefix: str = "alex-rivera-") -> None:
        super().__init__(directory)
        self.write = write
        self.stem = prefix
        self.n = 0
        self.usage: list[tuple[str, dict[str, Any]]] = []

    def save(self, kind: str, request: dict[str, Any], response: dict[str, Any]) -> Path:
        self.n += 1
        self.usage.append((kind, response.get("usage") or {}))
        if not self.write:
            return self.directory
        self.prefix = f"{self.stem}{self.n:02d}-"
        return super().save(kind, request, response)


def live_llm(directory: Path, stem: str):
    write = bool(os.environ.get("BRAIN_RECORD_FIXTURES"))
    if write:
        for old in directory.glob(f"{stem}[0-9]*.json"):
            old.unlink()
    client = LLM(load_settings().llm, recorder=OrderedRecorder(directory, write=write, prefix=stem))
    if not client.health():
        client.close()
        pytest.skip("local model server is not running")
    return client


@pytest.fixture
def llm():
    client = live_llm(FIXTURE_DIR, "alex-rivera-")
    yield client
    client.close()


def print_report(report: Any, llm: LLM, total: float) -> None:
    """Timings, token counts, the plan, decisions and notes of a live ingest."""
    plan = report.plan
    print(f"\n{len(report.chunks)} chunk(s), total {total:.1f}s, plan {plan.seconds:.1f}s")
    for part in plan.parts:
        print(f"  part: {part.title} ({part.tokens} tok)")
    usage = llm.recorder.usage if isinstance(llm.recorder, OrderedRecorder) else []
    structured = [u for kind, u in usage if kind == "structured"]
    if structured:
        u = structured[0]
        print(
            f"  plan: {u.get('prompt_tokens')} prompt / {u.get('completion_tokens')} output tokens"
        )
    briefed = report.brief is not None
    if briefed:
        u = structured[plan.calls] if len(structured) > plan.calls else {}
        print(
            f"  brief: {report.brief.seconds:.1f}s, {u.get('prompt_tokens')} prompt / "
            f"{u.get('completion_tokens')} output tokens"
        )
    extraction = structured[plan.calls + briefed :]
    for c, u in zip(report.chunks, extraction, strict=False):
        print(
            f"  chunk {c.chunk.index} [{c.chunk.heading}]: {c.seconds:.1f}s, {c.entities} entities, "
            f"{c.facts} facts, {u.get('prompt_tokens')} prompt / {u.get('completion_tokens')} "
            f"output tokens; gate p={c.p_about} {c.gate_seconds:.2f}s, {c.checks} checks "
            f"{c.check_seconds:.2f}s, {c.changed} changed"
        )
    for c in report.skipped:
        print(f"  skipped chunk {c.chunk.index} [{c.chunk.heading}]: p={c.p_about:.3f}")
    result = report.applied
    print(f"applied {result.facts_added} facts, created {result.created}")
    for op, reason in result.queued:
        print(f"  queued {op.op}: {reason}")
    for note in report.normalised.notes:
        print(f"  note: {note}")


def test_ingest_homepage_live(fresh_vault: Vault, llm: LLM) -> None:
    db = open_index(fresh_vault.db_path, embed_model=llm.embed_model, dim=llm.embed_dim)
    start = time.perf_counter()
    report = ingest(DOCS / "alex-rivera.html", fresh_vault, llm, db=db, today=TODAY)
    total = time.perf_counter() - start
    result = report.applied
    assert result is not None and report.normalised is not None and report.plan is not None
    print_report(report, llm, total)

    facts = {(f.p, f.o) for p in fresh_vault.pages() if p.subject == "self" for f in p.facts}
    assert ("works_at", "[[orgs/northwind-labs]]") in facts
    assert ("works_at", "[[orgs/fabrikam-retail]]") in facts
    assert {"Spanish", "English"} <= {o for p, o in facts if p == "speaks"}
    assert {"Python", "SQL"} <= {o for p, o in facts if p == "has_skill"}
    assert any(p == "responsibility" for p, _ in facts)
    assert fresh_vault.page("projects/atlas-migration") is not None
    sam = fresh_vault.page("people/sam-okafor")
    assert sam is not None and sam.privacy == "private"
    assert any(getattr(op, "p", "") == "friend_of" for op, _ in result.queued)
    assert not any(getattr(op, "p", "") == "lives_in" for op, _ in result.queued)
    assert quality_problems(fresh_vault) == []
    # the brief: read once, shared by every part, and the project is placed in its employer
    assert report.brief is not None and report.brief.brief is not None
    assert "Alex Rivera" in report.brief.brief.subject
    atlas = fresh_vault.page("projects/atlas-migration")
    assert any(f.p == "part_of" and "northwind" in f.o for f in atlas.facts)


PORTFOLIO_DIR = FIXTURE_DIR / "portfolio"


@pytest.fixture
def portfolio_llm():
    PORTFOLIO_DIR.mkdir(exist_ok=True)
    client = live_llm(PORTFOLIO_DIR, "alex-rivera-portfolio-")
    yield client
    client.close()


def test_ingest_portfolio_live(fresh_vault: Vault, portfolio_llm: LLM) -> None:
    """The page that exercises the gate and the typed checks (see ``portfolio_problems``)."""
    llm = portfolio_llm
    start = time.perf_counter()
    report = ingest(DOCS / "alex-rivera-portfolio.html", fresh_vault, llm, today=TODAY)
    total = time.perf_counter() - start
    assert report.applied is not None and report.normalised is not None
    print_report(report, llm, total)
    assert portfolio_problems(report, fresh_vault) == []
