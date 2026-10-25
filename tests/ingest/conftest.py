"""Scripted extractions (fictional persona) for ingest tests that run without the model."""

from __future__ import annotations

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


def entity(
    name: str, type_: str, id_: str = "", aliases: list[str] | None = None, summary: str = ""
) -> dict:
    return {"name": name, "type": type_, "id": id_, "aliases": aliases or [], "summary": summary}


def homepage_extraction() -> dict[str, Any]:
    """What a good extraction of tests/fixtures/docs/alex-rivera.html looks like."""
    return {
        "reasoning": "A personal homepage: career, projects, skills and education of Alex.",
        "owner_names": ["Alex Rivera", "Alex"],
        "entities": [
            entity("Northwind Labs", "org", summary="Company where Alex works."),
            entity("Fabrikam Retail", "org", summary="Retailer in Lisbon."),
            entity(
                "Atlas Migration",
                "project",
                aliases=["Atlas"],
                summary="Move of nightly batch pipelines to a managed warehouse.",
            ),
            entity("Sam Okafor", "person", summary="Friend who co-maintains Tidepool."),
            entity("Airflow", "topic"),
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
