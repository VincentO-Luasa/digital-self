"""``ingest`` end to end with a scripted model: raw copy, apply, review queue, generated files,
reindex, idempotent re-ingest, and web pages fetched through ``httpx.MockTransport``."""

from __future__ import annotations

import subprocess
from datetime import date

import httpx
import pytest
from conftest import (  # type: ignore[import-not-found]
    DOCS,
    homepage_extractions,
    homepage_plan,
)

from brain.index.db import open_index
from brain.index.search import search
from brain.ingest import FetchError, ingest, ingest_url
from brain.ingest.chunking import halve
from brain.ingest.planning import plan_chunks
from brain.ingest.readers import load_document
from brain.llm import OutputTruncated, StructuredOutputError
from brain.llm.fake import FakeLLM
from brain.vault import gitops, pending_reviews
from brain.vault.store import Vault

TODAY = date(2026, 10, 25)
HOMEPAGE = DOCS / "alex-rivera.html"


def homepage_chunks():
    doc = load_document(HOMEPAGE)
    return plan_chunks(
        FakeLLM(structured_responses=[homepage_plan()]), doc.text, doc_title=""
    ).chunks


N_CHUNKS = len(homepage_chunks())


def empty_extraction() -> dict:
    return {"reasoning": "Nothing new here.", "owner_names": [], "entities": [], "facts": []}


def homepage_responses() -> list:
    """The scripted split plan, then the extraction of each of the two parts."""
    return [homepage_plan(), *homepage_extractions()]


def run(vault: Vault, llm: FakeLLM):
    db = open_index(vault.db_path, embed_model=llm.embed_model, dim=llm.embed_dim)
    return ingest(HOMEPAGE, vault, llm, db=db, today=TODAY), db


def test_ingest_homepage(fresh_vault: Vault) -> None:
    assert N_CHUNKS == 2
    llm = FakeLLM(structured_responses=homepage_responses())
    report, db = run(fresh_vault, llm)

    assert report.raw_path == "raw/docs/2026-10-25-alex-rivera.html"
    assert report.source == "alex-rivera.html"
    assert (fresh_vault.root / report.raw_path).exists()
    assert len(report.chunks) == N_CHUNKS
    assert report.plan is not None and report.plan.planned and report.plan.calls == 1
    assert [p.title for p in report.plan.parts] == [
        "Profile and about me",
        "Career, projects and education",
    ]
    # the plan sees an outline; each extraction prompt carries its part's title and text and
    # the known entities
    structured = [c[1] for c in llm.calls if c[0] == "structured"]
    plan_user = structured[0]["messages"][1]["content"]
    assert "] ## Experience" in plan_user and "moved it from spreadsheets" not in plan_user
    first, second = (structured[i]["messages"][1]["content"] for i in (1, 2))
    assert "Section: Profile and about me" in first and "I grew up in Valencia" in first
    # labelled by its first heading and the title (never an earlier section's trail)
    assert "Section: Alex Rivera > Experience > Career, projects and education" in second
    assert "Fabrikam Retail" in second and "Fabrikam Retail" not in first
    assert "- self: the owner (name unknown)" in first

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
    identity = next(line for line in core.splitlines() if line.startswith("Name: "))
    assert "Alex Rivera" in identity and "Senior Data Engineer at Northwind Labs" in identity
    story = core.split("## Life story", 1)[1].split("\n## ", 1)[0]
    assert "Northwind Labs" in story and "Fabrikam Retail" in story
    assert "[[orgs/northwind-labs]]" in fresh_vault.index_path.read_text()
    assert not gitops.has_staged_changes(fresh_vault.root)

    assert report.reindexed is not None and report.reindexed.added > 0
    hits = search(db, llm, "Fabrikam Retail", k=5)
    assert any("Fabrikam" in h.text for h in hits)


def test_reingest_is_idempotent(fresh_vault: Vault) -> None:
    llm = FakeLLM(structured_responses=[*homepage_responses(), *homepage_responses()])
    first, _ = run(fresh_vault, llm)
    second, _ = run(fresh_vault, llm)
    assert second.raw_path == first.raw_path and not second.raw_created
    result = second.applied
    assert result is not None
    assert result.facts_added == 0 and not result.created
    assert all("duplicate" in reason or "exists" in reason for _, reason in result.skipped)
    assert len(list((fresh_vault.raw_dir / "docs").iterdir())) == 1


def test_truncated_chunk_is_split_and_retried(fresh_vault: Vault) -> None:
    halves = halve(homepage_chunks()[1])
    assert len(halves) >= 2
    profile, career = homepage_extractions()
    llm = FakeLLM(
        structured_responses=[
            homepage_plan(),
            profile,
            OutputTruncated("cut off", '{"reasoning"'),
            *[career for _ in halves],  # each half keeps the facts its own text names
        ]
    )
    report, _ = run(fresh_vault, llm)
    assert not report.failed
    assert len(report.chunks) == len(halves) + N_CHUNKS - 1
    assert report.applied is not None and report.applied.facts_added >= 8


def test_failed_chunk_is_skipped_not_fatal(fresh_vault: Vault) -> None:
    llm = FakeLLM(
        structured_responses=[
            homepage_plan(),
            StructuredOutputError("bad json", "{"),
            *[empty_extraction() for _ in range(N_CHUNKS - 1)],
        ]
    )
    report, _ = run(fresh_vault, llm)
    assert len(report.failed) == 1 and "bad json" in report.failed[0][1]
    assert len(report.chunks) == N_CHUNKS - 1
    assert (fresh_vault.root / report.raw_path).exists()  # the raw copy is kept for a re-run


# --- web pages ---------------------------------------------------------------------------------

INDEX = b"""<html><head><title>Alex</title></head><body>
<nav><a href="/cv">CV</a> <a href="/cv#top">CV again</a> <a href="/notes.md">Notes</a>
<a href="/photo.jpg">Photo</a> <a href="https://elsewhere.example/">Elsewhere</a>
<a href="mailto:alex@example.com">Mail</a></nav>
<h1>Alex Rivera</h1><p>Data engineer in Porto.</p></body></html>"""


def site(requests: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/":
            return httpx.Response(200, content=INDEX, headers={"content-type": "text/html"})
        if request.url.path == "/cv":
            html = HOMEPAGE.read_bytes()
            return httpx.Response(200, content=html, headers={"content-type": "text/html"})
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_ingest_url_follows_same_origin_links(fresh_vault: Vault) -> None:
    requests: list[httpx.Request] = []
    llm = FakeLLM(structured_responses=[empty_extraction(), *homepage_responses()])
    pages: list[str] = []
    with site(requests) as http:
        report = ingest_url(
            "https://alex.example/",
            fresh_vault,
            llm,
            follow=5,
            http=http,
            today=TODAY,
            on_page=lambda url, n, total: pages.append(f"{n}/{total} {url}"),
        )
    # /cv#top is the same page as /cv; the image, other origins and mailto are not followed
    assert pages == [
        "1/3 https://alex.example/",
        "2/3 https://alex.example/cv",
        "3/3 https://alex.example/notes.md",
    ]
    assert [p.raw_path for p in report.pages] == [
        "raw/docs/2026-10-25-alex-example.html",
        "raw/docs/2026-10-25-alex-example-cv.html",
    ]
    assert report.failed == [
        ("https://alex.example/notes.md", "https://alex.example/notes.md: HTTP 404")
    ]
    cv = report.pages[1]
    assert cv.source == "https://alex.example/cv"
    assert (fresh_vault.root / cv.raw_path).read_bytes() == HOMEPAGE.read_bytes()
    assert cv.applied is not None and "orgs/northwind-labs" in cv.applied.created
    assert fresh_vault.page("orgs/northwind-labs") is not None
    history = subprocess.run(
        ["git", "log", "--format=%s"], cwd=fresh_vault.root, capture_output=True, text=True
    ).stdout
    assert f"brain: raw doc {cv.raw_path} from https://alex.example/cv" in history


def test_ingest_url_without_follow_and_failures(fresh_vault: Vault) -> None:
    requests: list[httpx.Request] = []
    llm = FakeLLM(structured_responses=homepage_responses())
    with site(requests) as http:
        report = ingest_url("https://alex.example/cv", fresh_vault, llm, http=http, today=TODAY)
        assert len(report.pages) == 1 and not report.failed
        assert [str(r.url) for r in requests] == ["https://alex.example/cv"]
        with pytest.raises(FetchError, match="HTTP 404"):
            ingest_url("https://alex.example/missing", fresh_vault, llm, http=http)
        with pytest.raises(FetchError, match="only http"):
            ingest_url("ftp://alex.example/cv", fresh_vault, llm, http=http)
