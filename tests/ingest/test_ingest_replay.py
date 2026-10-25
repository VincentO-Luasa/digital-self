"""Replay the recorded live extraction of the fictional homepage (no model server needed).

The recorded request must equal what ``ingest`` sends today: the prompt is pinned. If this
fails after an intended prompt or schema change, re-record with
``BRAIN_RECORD_FIXTURES=1 uv run pytest -m llm tests/ingest/test_ingest_live.py``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from conftest import DOCS, quality_problems  # type: ignore[import-not-found]

from brain.ingest import ingest
from brain.ingest.extract import Extraction
from brain.llm.client import response_format
from brain.llm.fake import FakeLLM
from brain.llm.recorder import load_exchange
from brain.vault.store import Vault

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "llm" / "ingest"
TODAY = date(2026, 10, 25)


def fixtures() -> list[Path]:
    paths = sorted(FIXTURE_DIR.glob("alex-rivera-*.json"))
    assert paths, "no recorded ingest fixture: record it with BRAIN_RECORD_FIXTURES=1"
    return paths


def test_recorded_requests_match_the_pinned_prompt(fresh_vault: Vault) -> None:
    paths = fixtures()
    fake = FakeLLM.from_fixtures(paths)
    ingest(DOCS / "alex-rivera.html", fresh_vault, fake, today=TODAY)
    sent = [call[1]["messages"] for call in fake.calls if call[0] == "structured"]
    recorded = [load_exchange(p).request for p in paths]
    assert sent == [r["messages"] for r in recorded]
    assert all(r["response_format"] == response_format(Extraction) for r in recorded)
    assert all(
        r["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False for r in recorded
    )


def test_replayed_extraction_lands_in_the_vault(fresh_vault: Vault) -> None:
    fake = FakeLLM.from_fixtures(fixtures())
    report = ingest(DOCS / "alex-rivera.html", fresh_vault, fake, today=TODAY)
    result = report.applied
    assert result is not None
    self_facts = [f for p in fresh_vault.pages() if p.subject == "self" for f in p.facts]
    by_p: dict[str, set[str]] = {}
    for f in self_facts:
        by_p.setdefault(f.p, set()).add(f.o)
    assert "[[orgs/northwind-labs]]" in by_p["works_at"]
    assert "[[orgs/fabrikam-retail]]" in by_p["works_at"]
    assert {"Python", "SQL"} <= by_p["has_skill"]
    assert {"Spanish", "English"} <= by_p["speaks"]
    assert all(f.kind == "stated" for f in self_facts)
    assert all(f.src == [report.raw_path] for f in self_facts)
    assert not any(f.text.lower().startswith(("he ", "she ")) for f in self_facts)
    sam = fresh_vault.page("people/sam-okafor")
    assert sam is not None and sam.privacy == "private"
    assert any(getattr(op, "p", "") == "friend_of" for op, _ in result.queued)
    assert any(f.p == "responsibility" for f in self_facts)
    assert quality_problems(fresh_vault) == []
    core = fresh_vault.core_path.read_text()
    assert "Name: Alex Rivera" in core
