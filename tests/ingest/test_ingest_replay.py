"""Replay the recorded live extraction of the fictional homepage (no model server needed).

The recorded request must equal what ``ingest`` sends today: the prompt is pinned. If this
fails after an intended prompt or schema change, re-record with
``BRAIN_RECORD_FIXTURES=1 uv run pytest -m llm tests/ingest/test_ingest_live.py``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from conftest import (  # type: ignore[import-not-found]
    DOCS,
    PORTFOLIO,
    portfolio_problems,
    quality_problems,
)

from brain.config import LLMSettings
from brain.ingest import ingest, pipeline
from brain.ingest.brief import Brief
from brain.ingest.extract import Extraction
from brain.ingest.planning import SplitPlan
from brain.llm.client import response_format
from brain.llm.decide import request_payload
from brain.llm.fake import FakeLLM
from brain.llm.recorder import load_exchange
from brain.vault.store import Vault

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "llm" / "ingest"
TODAY = date(2026, 10, 25)


@pytest.fixture(autouse=True)
def no_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixtures hold no embeddings, and the fake's hashed bag-of-words vectors are not the
    embedding model's: paraphrase merging (``dedup``, tested on its own) is left out here."""
    monkeypatch.setattr(pipeline, "DEDUP", False)


def fixtures() -> list[Path]:
    paths = sorted(FIXTURE_DIR.glob("alex-rivera-*.json"))
    assert paths, "no recorded ingest fixture: record it with BRAIN_RECORD_FIXTURES=1"
    return paths


def test_recorded_requests_match_the_pinned_prompt(fresh_vault: Vault) -> None:
    paths = fixtures()
    fake = FakeLLM.from_fixtures(paths)
    ingest(DOCS / "alex-rivera.html", fresh_vault, fake, today=TODAY)
    exchanges = [load_exchange(p) for p in paths]
    sent = [call[1]["messages"] for call in fake.calls if call[0] == "structured"]
    recorded = [e.request for e in exchanges if e.kind == "structured"]
    assert sent == [r["messages"] for r in recorded]
    # the gate and checks: same inputs and questions, in the same order
    asked = [
        request_payload(LLMSettings().chat_model, c[1]["text"], c[1]["questions"])
        for c in fake.calls
        if c[0] == "decide"
    ]
    decided = [e.request for e in exchanges if e.kind == "decide"]
    assert [(a["input"], a["questions"]) for a in asked] == [
        (d["input"], d["questions"]) for d in decided
    ]
    assert decided, "the recording has no gate/check decisions"
    # a split plan first (the page is over one chunk), the brief, then one extraction per part
    assert recorded[0]["response_format"] == response_format(SplitPlan)
    assert recorded[1]["response_format"] == response_format(Brief)
    assert all(r["response_format"] == response_format(Extraction) for r in recorded[2:])
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
    # the brief places the project in its employer (no part states it)
    atlas = fresh_vault.page("projects/atlas-migration")
    assert atlas is not None and any(
        f.p == "part_of" and f.o == "[[orgs/northwind-labs]]" for f in atlas.facts
    )
    core = fresh_vault.core_path.read_text()
    assert "Name: Alex Rivera" in core


PORTFOLIO_DIR = FIXTURE_DIR / "portfolio"


def test_replayed_portfolio_gate_and_checks(fresh_vault: Vault) -> None:
    """The recorded live ingest of the portfolio page: a part that is only the tutorial is
    skipped, the team, tool and topic are values, hobbies are not skills (``portfolio_problems``).
    """
    paths = sorted(PORTFOLIO_DIR.glob("alex-rivera-portfolio-*.json"))
    assert paths, "no recorded portfolio ingest: record it with BRAIN_RECORD_FIXTURES=1"
    fake = FakeLLM.from_fixtures(paths)
    report = ingest(PORTFOLIO, fresh_vault, fake, today=TODAY)
    exchanges = [load_exchange(p) for p in paths]
    sent = [c[1]["messages"] for c in fake.calls if c[0] == "structured"]
    assert sent == [e.request["messages"] for e in exchanges if e.kind == "structured"]
    asked = [(c[1]["text"], c[1]["questions"]) for c in fake.calls if c[0] == "decide"]
    decided = [e.request for e in exchanges if e.kind == "decide"]
    assert [(d["input"], d["questions"]) for d in decided] == [
        (r["input"], r["questions"])
        for r in (request_payload(LLMSettings().chat_model, t, q) for t, q in asked)
    ]
    assert portfolio_problems(report, fresh_vault) == []
    assert all(c.p_about is not None for c in [*report.chunks, *report.skipped])  # all gated
    assert any(n.startswith("check: ") for n in report.normalised.notes)
