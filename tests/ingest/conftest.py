"""Scripted extractions (fictional persona) for ingest tests that run without the model."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from brain.ingest.chunking import split_blocks
from brain.ingest.readers import load_document
from brain.vault.init import init_vault
from brain.vault.store import Vault

DOCS = Path(__file__).resolve().parents[1] / "fixtures" / "docs"


def fact(subject: str, p: str, o: str, text: str, start: str = "", end: str = "") -> dict:
    return {
        "subject": subject,
        "predicate": p,
        "object": o,
        "text": text,
        "start": start,
        "end": end,
    }


def entity(name: str, type_: str, id_: str = "", aliases: list[str] | None = None) -> dict:
    return {"name": name, "type": type_, "id": id_, "aliases": aliases or []}


def homepage_plan() -> dict[str, Any]:
    """A split plan for tests/fixtures/docs/alex-rivera.html: profile, then the career."""
    blocks = split_blocks(load_document(DOCS / "alex-rivera.html").text)
    experience = next(b.index for b in blocks if b.heading and b.text == "## Experience")
    return {
        "reasoning": "A homepage: profile and about, then experience, projects and education.",
        "parts": [
            {"first_block": 0, "title": "Profile and about me"},
            {"first_block": experience, "title": "Career, projects and education"},
        ],
    }


def homepage_extraction() -> dict[str, Any]:
    """What a good extraction of tests/fixtures/docs/alex-rivera.html looks like."""
    return {
        "reasoning": "A personal homepage: career, projects, skills and education of Alex.",
        "owner_names": ["Alex Rivera", "Alex"],
        "entities": [
            entity("Northwind Labs", "org"),
            entity("Fabrikam Retail", "org"),
            entity("Atlas Migration", "project", aliases=["Atlas"]),
            entity("Sam Okafor", "person"),
            entity("Airflow", "tool"),
            entity("Lisbon", "place"),
        ],
        "facts": [
            fact("self", "name", "Alex Rivera", "Is called Alex Rivera"),
            fact("self", "lives_in", "Porto", "He lives in Porto", "2024"),
            fact("self", "works_at", "Northwind Labs", "Works at Northwind Labs", "2025-03"),
            fact(
                "self",
                "works_at",
                "Northwind Labs",
                "Worked at Northwind Labs",
                "2023-02",
                "2025-03",
            ),
            fact(
                "self",
                "role",
                "Data Engineer",
                "Was a data engineer",
                "February 2023",
                "March 2025",
            ),
            fact(
                "self",
                "role",
                "Senior Data Engineer",
                "Is a senior data engineer",
                "2025-03",
                "present",
            ),
            fact(
                "self",
                "works_at",
                "Fabrikam Retail",
                "Worked at Fabrikam Retail",
                "2019-09",
                "2023-01",
            ),
            fact(
                "self",
                "studied_at",
                "Westbridge University",
                "Studied at Westbridge",
                "2018",
                "2019",
            ),
            fact("self", "has_skill", "Airflow", "Has skill in Airflow"),
            fact("self", "owner", "Atlas Migration", "Leads the Atlas Migration", "2026"),
            fact("self", "owner", "self", "Owns itself"),
            fact("Atlas Migration", "uses", "Airflow", "It uses Airflow"),
            fact("Sam Okafor", "friend_of", "self", "Is a friend of Alex"),
            fact("Somebody Unknown", "knows", "self", "Knows Alex"),
            fact("self", "born", "self", "Was born"),
        ],
    }


def homepage_extractions() -> tuple[dict[str, Any], dict[str, Any]]:
    """``homepage_extraction`` split over the two parts of ``homepage_plan``: each fact goes with
    the part whose text names its entities (ingest drops facts about entities a part never
    names)."""
    full = homepage_extraction()
    profile_preds = {"name", "lives_in", "born", "knows"}
    profile = {**full, "entities": [], "facts": []}
    career = {**full, "facts": []}
    for f in full["facts"]:
        (profile if f["predicate"] in profile_preds else career)["facts"].append(f)
    return profile, career


@pytest.fixture
def fresh_vault(tmp_path: Path, git_env: None) -> Vault:
    """A brand-new vault from ``brain init-vault`` (no pages except self/profile)."""
    return Vault(init_vault(tmp_path / "vault").root)


# --- quality checks shared by the replay and live tests (problems seen on a real homepage) ----

TOOLS = ("codeharbor", "pebble", "lumen", "triplog", "tl4521", "python", "airflow", "dbt")
# pronouns that can stand for the owner ("they"/"their" only as the leading subject: in "help
# analysts trust their numbers" it is the analysts' numbers)
PRONOUN = re.compile(r"\b(?:he|she|his|her|i|my)\b|^(?:they|their)\b", re.I)
OWNER = re.compile(r"\balex\b|\brivera\b", re.I)
ZERO = re.compile(r"(?<![\w.,])0\+?\s+[a-z]", re.I)


def quality_problems(vault: Vault) -> list[str]:
    """What is wrong with an ingest of tests/fixtures/docs/alex-rivera.html ([] if nothing)."""
    problems: list[str] = []
    pages = {p.id: p for p in vault.pages()}
    for page_id, page in pages.items():
        slug = page_id.rsplit("/", 1)[-1]
        if any(t in slug for t in TOOLS) or "team" in slug:
            problems.append(f"page for a tool or team: {page_id}")
        for f in page.facts:
            if ZERO.search(f.text) or ZERO.search(f.o):
                problems.append(f"zero counter: {f.text}")
            if PRONOUN.search(f.text):
                problems.append(f"pronoun: {f.text}")
            if page.subject == "self" and f.p != "name" and OWNER.search(f.text):
                problems.append(f"owner's name: {f.text}")
            if f.p == "works_at" and not (f.object_id or "").startswith("orgs/"):
                problems.append(f"works_at a non-org: {f.o}")
            if page.subject == "self" and f.p == "works_at" and "team" in f.text.lower():
                problems.append(f"works_at a team: {f.text}")
    current = [
        f.o
        for p in pages.values()
        if p.subject == "self"
        for f in p.facts
        if f.p == "lives_in" and not f.valid_to
    ]
    if any("porto" not in o.lower() for o in current):
        problems.append(f"past city as current lives_in: {current}")
    return problems


# --- the long fictional "about me" page (tests/fixtures/docs/alex-rivera-long.md) -------------

LONG_DOC = DOCS / "alex-rivera-long.md"
# lines that belong to one topic and must land in the same part
LONG_TOPICS = {
    "Northwind Labs": ("I joined Northwind Labs", "promoted to Senior", "writing culture"),
    "Fabrikam Retail": ("From September 2019", "stock-forecasting dashboard"),
    "Atlas Migration": ("Since 2026 I have been leading", "The hard part is not the code"),
    "Tidepool": ("Tidepool is a small open-source",),
}


def split_problems(report: Any, text: str, cap: int) -> list[str]:
    """What is wrong with a plan for the long page ([] if nothing)."""
    from brain.ingest.chunking import join_blocks

    problems: list[str] = []
    if not report.planned:
        problems.append(f"not planned: {report.notes}")
    if "\n\n".join(c.text for c in report.chunks) != join_blocks(split_blocks(text, cap)).strip():
        problems.append("the chunks do not cover every block exactly once, in order")
    for part in report.parts:
        if part.tokens > cap:
            problems.append(f"part over the cap: {part.title} ({part.tokens} tok)")

    def where(line: str) -> int:
        return next(i for i, c in enumerate(report.chunks) if line in c.text)

    for topic, lines in LONG_TOPICS.items():
        if len({where(line) for line in lines}) > 1:
            problems.append(f"{topic} is split across parts")
    if where("Bouldering is my main sport") == where("I play the guitar"):
        problems.append("the long hobbies section was not split")
    for chunk in report.chunks:
        if chunk.text.rstrip().split("\n")[-1].startswith("#"):
            problems.append(f"a part ends with a heading: {chunk.heading}")
    return problems


# --- the portfolio page (tests/fixtures/docs/alex-rivera-portfolio.html): gate and checks ------

PORTFOLIO = DOCS / "alex-rivera-portfolio.html"
HOBBIES = ("bouldering", "sourdough", "chess")


def portfolio_problems(report: Any, vault: Vault) -> list[str]:
    """What is wrong with an ingest of the portfolio page ([] if nothing)."""
    problems: list[str] = []
    pages = {p.id: p for p in vault.pages()}
    for page_id, page in pages.items():
        slug = page_id.rsplit("/", 1)[-1]
        if any(t in slug for t in ("data-platform", "lakehouse", "causal", "python", "spark")):
            problems.append(f"page for a team, tool or topic: {page_id}")
        if re.match(r"\d", page.title):
            problems.append(f"page title glued to a section number: {page.title}")
    for name in ("projects/tidepool", "orgs/northwind-labs", "orgs/fabrikam-retail"):
        if name not in pages:
            problems.append(f"missing page {name}")
    self_facts = [f for p in pages.values() if p.subject == "self" for f in p.facts]
    for f in self_facts:
        if f.p == "has_skill" and any(h in f.o.lower() for h in HOBBIES):
            problems.append(f"hobby as a skill: {f.o}")
        if f.p == "works_at" and "northwind" not in f.o and "fabrikam" not in f.o:
            problems.append(f"works_at {f.o}")
        if f.p == "speaks" and f.o.lower() not in ("english", "spanish", "portuguese"):
            problems.append(f"speaks a non-language: {f.o}")
    if not any(f.p == "has_skill" and "python" in f.o.lower() for f in self_facts):
        problems.append("Python is not a skill")
    for chunk in report.plan.chunks:
        job = "Since March 2025" in chunk.text
        if job and ("Side projects" in chunk.heading or "reading-list" in chunk.heading):
            problems.append(f"the job's part is labelled as a side project: {chunk.heading}")
    # a part that is only the generic tutorial and the privacy notice must be skipped (the plan
    # varies: when it puts the contact block or the education in the same part, it is kept)
    personal = ("Westbridge", "Get in touch", "Northwind", "Tidepool")
    junk = [
        c
        for c in report.plan.chunks
        if c.text.count("partition") > 5 and not any(p in c.text for p in personal)
    ]
    skipped = {c.chunk.index for c in report.skipped}
    if any(c.index not in skipped for c in junk):
        problems.append("the generic tutorial part was extracted, not skipped")
    if any("Northwind" in c.chunk.text for c in report.skipped):
        problems.append("a part about the owner was skipped")
    return problems


# --- fictional parts for the extraction gate (tests/ingest/test_ingest_gate_*.py) ---------------

GATE_PARTS = {  # name → (text, should be extracted)
    "cv": (
        "## Experience\n\n### Northwind Labs — Porto\n\n- **Senior Data Engineer**, March 2025 – "
        "present. I lead the Atlas Migration and run the platform team's on-call rotation.\n"
        "- **Data Engineer**, February 2023 – March 2025. Built the ingestion framework for the "
        "product analytics team.",
        True,
    ),
    "hobbies": (
        "## Life outside work\n\nAlex has done calisthenics every morning for ten years. On "
        "weekends he goes bouldering with Sam Okafor and bakes sourdough bread.",
        True,
    ),
    "opinion": (
        "## Why I prefer boring pipelines\n\nI have come to value boring, predictable pipelines "
        "over clever ones. After years of on-call I care more about clear documentation than "
        "about the newest framework.",
        True,
    ),
    "cookie banner": (
        "## Privacy and cookies\n\nThis site uses cookies to measure traffic and remember your "
        "preferences. Analytics cookies are only set if you accept them. You can withdraw consent "
        "at any time from the settings page. Data is stored for 13 months and never sold to "
        "third parties. For details, see the full privacy policy.",
        False,
    ),
    "footer": ("© 2026 · Built with a static site generator · Last updated October 2026", False),
    "tutorial": (
        "## Quick guide: partitioning tables\n\nPartitioning splits a large table into smaller "
        "pieces by a key such as the event date. Queries that filter on the key read only the "
        "matching partitions, which cuts scan time and cost. Choose a key with a few hundred to a "
        "few thousand distinct values; too many tiny partitions slow the planner down. "
        "Re-cluster the table after large backfills.",
        False,
    ),
    # a contact block says little but is not junk either: it must not be skipped by mistake
    "contact": (
        "## Get in touch\n\nThe best way to reach me is by email. I try to answer within a week. "
        "Press and speaking requests are welcome.\n\nalex@example.com",
        None,
    ),
}
