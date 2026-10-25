"""``ingest`` end to end with a scripted model: raw copy, apply, review queue, generated files,
reindex, and idempotent re-ingest."""

from __future__ import annotations

from datetime import date

from conftest import DOCS, homepage_extraction  # type: ignore[import-not-found]

from brain.index.db import open_index
from brain.index.search import search
from brain.ingest import ingest
from brain.llm.fake import FakeLLM
from brain.vault import gitops, pending_reviews
from brain.vault.store import Vault

TODAY = date(2026, 10, 25)


def run(vault: Vault, llm: FakeLLM):
    db = open_index(vault.db_path, embed_model=llm.embed_model, dim=llm.embed_dim)
    return ingest(DOCS / "alex-rivera.html", vault, llm, db=db, today=TODAY), db


def test_ingest_homepage(fresh_vault: Vault) -> None:
    llm = FakeLLM(structured_responses=[homepage_extraction()])
    report, db = run(fresh_vault, llm)

    assert report.raw_path == "raw/docs/2026-10-25-alex-rivera.html"
    assert (fresh_vault.root / report.raw_path).exists()
    assert len(report.chunks) == 1
    # the prompt carries the chunk and the (empty) known entities
    messages = llm.calls[0][1]["messages"]
    assert "Fabrikam Retail" in messages[1]["content"]
    assert "- self: the owner (name unknown)" in messages[1]["content"]

    result = report.applied
    assert result is not None and result.facts_added >= 8
    assert "people/sam-okafor" in result.created
    queued = {(op.op, getattr(op, "p", None)) for op, _ in result.queued}
    assert ("add_fact", "friend_of") in queued  # relationships always go to review
    reviews = pending_reviews(fresh_vault)
    assert [r.id for r in reviews] == [result.review_id]
    assert reviews[0].src == [report.raw_path]

    sam = fresh_vault.page("people/sam-okafor")
    assert sam is not None and sam.privacy == "private"
    profile = fresh_vault.page("self/profile")
    assert profile is not None and profile.aliases == ["Alex Rivera", "Alex"]

    core = fresh_vault.core_path.read_text()
    assert "Name: Alex Rivera" in core and "- Is a senior data engineer (since 2025-03)" in core
    assert "[[orgs/northwind-labs]]" in fresh_vault.index_path.read_text()
    assert not gitops.has_staged_changes(fresh_vault.root)

    assert report.reindexed is not None and report.reindexed.added > 0
    hits = search(db, llm, "Fabrikam Retail", k=5)
    assert any("Fabrikam" in h.text for h in hits)


def test_reingest_is_idempotent(fresh_vault: Vault) -> None:
    llm = FakeLLM(structured_responses=[homepage_extraction(), homepage_extraction()])
    first, _ = run(fresh_vault, llm)
    second, _ = run(fresh_vault, llm)
    assert second.raw_path == first.raw_path and not second.raw_created
    result = second.applied
    assert result is not None
    assert result.facts_added == 0 and not result.created
    assert all("duplicate" in reason or "exists" in reason for _, reason in result.skipped)
    assert len(list((fresh_vault.raw_dir / "docs").iterdir())) == 1


def empty_extraction() -> dict:
    return {"reasoning": "Nothing new here.", "owner_names": [], "entities": [], "facts": []}


def test_truncated_chunk_is_split_and_retried(fresh_vault: Vault) -> None:
    from brain.ingest.chunking import chunk_markdown, halve
    from brain.ingest.loaders import load_document
    from brain.llm import OutputTruncated

    halves = halve(chunk_markdown(load_document(DOCS / "alex-rivera.html").text)[0])
    assert len(halves) >= 2
    llm = FakeLLM(
        structured_responses=[
            OutputTruncated("cut off", '{"reasoning"'),
            homepage_extraction(),
            *[empty_extraction() for _ in halves[1:]],
        ]
    )
    report, _ = run(fresh_vault, llm)
    assert not report.failed
    assert len(report.chunks) == len(halves)
    assert report.applied is not None and report.applied.facts_added >= 8


def test_failed_chunk_is_skipped_not_fatal(fresh_vault: Vault) -> None:
    from brain.llm import StructuredOutputError

    llm = FakeLLM(structured_responses=[StructuredOutputError("bad json", "{")])
    report, _ = run(fresh_vault, llm)
    assert len(report.failed) == 1 and "bad json" in report.failed[0][1]
    assert report.chunks == []
    assert (fresh_vault.root / report.raw_path).exists()  # the raw copy is kept for a re-run
