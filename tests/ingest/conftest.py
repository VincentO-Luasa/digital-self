"""Scripted extractions (fictional persona) for ingest tests that run without the model."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

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


@pytest.fixture
def fresh_vault(tmp_path: Path, git_env: None) -> Vault:
    """A brand-new vault from ``brain init-vault`` (no pages except self/profile)."""
    return Vault(init_vault(tmp_path / "vault").root)


# --- quality checks shared by the replay and live tests (problems seen on a real homepage) ----

TOOLS = ("codeharbor", "pebble", "lumen", "triplog", "tl4521", "python", "airflow", "dbt")
PRONOUN = re.compile(r"\b(?:he|she|his|her|they|their|i|my)\b", re.I)
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
