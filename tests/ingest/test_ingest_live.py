"""Ingest the fictional homepage with the real local model. ``uv run pytest -m llm``.

Set ``BRAIN_RECORD_FIXTURES=1`` to (re-)record the replay fixture in
``tests/fixtures/llm/ingest/`` (needed whenever the extraction prompt or schema changes).
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from conftest import DOCS, quality_problems  # type: ignore[import-not-found]

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
    """Keep each call's token usage; when recording, save fixtures prefixed with the call
    number so replay keeps chunk order."""

    def __init__(self, directory: Path, *, write: bool) -> None:
        super().__init__(directory)
        self.write = write
        self.n = 0
        self.usage: list[dict[str, Any]] = []

    def save(self, kind: str, request: dict[str, Any], response: dict[str, Any]) -> Path:
        self.n += 1
        self.usage.append(response.get("usage") or {})
        if not self.write:
            return self.directory
        self.prefix = f"alex-rivera-{self.n:02d}-"
        return super().save(kind, request, response)


@pytest.fixture
def llm():
    write = bool(os.environ.get("BRAIN_RECORD_FIXTURES"))
    if write:
        shutil.rmtree(FIXTURE_DIR, ignore_errors=True)
    client = LLM(load_settings().llm, recorder=OrderedRecorder(FIXTURE_DIR, write=write))
    if not client.health():
        client.close()
        pytest.skip("local model server is not running")
    yield client
    client.close()


def test_ingest_homepage_live(fresh_vault: Vault, llm: LLM) -> None:
    db = open_index(fresh_vault.db_path, embed_model=llm.embed_model, dim=llm.embed_dim)
    start = time.perf_counter()
    report = ingest(DOCS / "alex-rivera.html", fresh_vault, llm, db=db, today=TODAY)
    total = time.perf_counter() - start
    result = report.applied
    assert result is not None and report.normalised is not None
    print(f"\n{len(report.chunks)} chunk(s), total {total:.1f}s")
    usage = llm.recorder.usage if isinstance(llm.recorder, OrderedRecorder) else []
    for c, u in zip(report.chunks, usage, strict=False):
        print(
            f"  chunk {c.chunk.index}: {c.seconds:.1f}s, {c.entities} entities, {c.facts} facts, "
            f"{u.get('prompt_tokens')} prompt / {u.get('completion_tokens')} output tokens"
        )
    print(f"applied {result.facts_added} facts, created {result.created}")
    for op, reason in result.queued:
        print(f"  queued {op.op}: {reason}")
    for note in report.normalised.notes:
        print(f"  note: {note}")

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
