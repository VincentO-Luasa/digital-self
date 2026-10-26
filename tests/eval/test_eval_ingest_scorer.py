"""The ingest scorer on hand-built extractions (no model, no vault unless stated)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from brain import cli
from brain.config import Settings
from brain.eval.ingest_eval import (
    EVAL_DIR,
    Extracted,
    GoldDoc,
    GoldEntity,
    GotEntity,
    GotFact,
    collect,
    error_kinds,
    names_literal,
    parse_fact,
    report_json,
    score_doc,
    similarity,
    tokens,
)
from brain.ingest.extract import Extraction
from brain.llm.base import StructuredOutputError
from brain.llm.fake import FakeLLM
from brain.vault.ops import AddFact, CreateEntity
from brain.vault.store import Vault


def gold_doc(required=(), optional=(), acceptable=(), entities=(), literals=()) -> GoldDoc:
    facts = [parse_fact(f, "required") for f in required]
    facts += [parse_fact(f, "optional") for f in optional]
    facts += [parse_fact(f, "acceptable") for f in acceptable]
    return GoldDoc(
        name="test",
        path=Path("test.md"),
        kind="stated",
        owner=("Alex Rivera", "Alex"),
        entities=[GoldEntity(n, t, tuple(a)) for n, t, a in entities],
        literals=list(literals),
        facts=facts,
    )


def test_tokens_fold_case_accents_punctuation_and_suffixes() -> None:
    assert tokens("Hallström School of Design") == tokens("hallstrom school design")
    assert tokens("Urban sketching") == tokens("urban sketcher")
    assert tokens("Human-Computer Interaction") == tokens("human computer interaction")
    assert tokens("the Data Platform team") == ["data", "platform", "team"]


@pytest.mark.parametrize(
    ("gold", "got", "free", "ok"),
    [
        ("Swedish", "Swedish (native)", False, True),
        ("Lead Product Designer", "lead product designer", False, True),
        ("Lead Product Designer", "Senior Product Designer", False, False),
        ("Senior Product Designer", "Product Designer", False, False),
        ("Kungsholm Institute of Technology", "Kungsholm Institute", False, False),
        ("Data Platform team", "Data Platform", False, False),
        ("accessibility baseline", "Sets the accessibility baseline for every app", True, True),
        ("accessibility baseline", "Sets the accessibility baseline for every app", False, False),
        ("1991", "1991", False, True),
    ],
)
def test_similarity(gold: str, got: str, free: bool, ok: bool) -> None:
    assert bool(similarity(gold, got, free_text=free)) is ok


def test_team_suffix_and_literal_pages() -> None:
    assert similarity("Data Platform team", "Data Platform", drop=frozenset({"team"}))
    assert names_literal("Design Systems team", ["Design Systems"])
    assert not names_literal("Design Systems team", ["Fjord", "Fjord design system"])
    assert names_literal("Airflow", ["Apache Airflow"])


def extracted() -> Extracted:
    return Extracted(
        entities=[
            GotEntity("orgs/northwind-labs", "org", ("Northwind Labs", "Northwind")),
            GotEntity("orgs/fabrikam-retail", "org", ("Fabrikam Retail",)),
            GotEntity("projects/atlas-migration", "org", ("Atlas Migration", "Atlas")),
            GotEntity("places/porto", "place", ("Porto",)),
            GotEntity("places/lisbon", "place", ("Lisbon",)),
            GotEntity("projects/airflow", "project", ("Airflow",)),
            GotEntity("people/alex-rivera", "person", ("Alex Rivera",)),
        ],
        facts=[
            GotFact("self", "works_at", "[[orgs/northwind-labs]]", "2023"),
            GotFact("self", "role", "Senior Data Engineer", "2025-03"),
            GotFact("self", "role", "Data Engineer", "2023-02", "2025-03", history=True),
            GotFact("self", "lives_in", "[[places/porto]]", "2024", "2025"),
            GotFact("self", "lives_in", "[[places/lisbon]]", "2019"),
            GotFact("self", "speaks", "Spanish (native)"),
            GotFact("self", "responsibility", "Data Platform team"),
            GotFact("projects/atlas-migration", "owner", "[[self]]", "2026"),
            GotFact("self", "has_skill", "[[projects/airflow]]"),
            GotFact("people/sam-okafor", "friend_of", "[[self]]", queued=True),
        ],
    )


GOLD_ENTITIES = [
    ("Northwind Labs", "org", ["Northwind"]),
    ("Fabrikam Retail", "org", []),
    ("Atlas Migration", "project", ["Atlas"]),
    ("Porto", "place", []),
    ("Sam Okafor", "person", ["Sam"]),
    ("Valencia", "place", []),
]


def scored():
    gold = gold_doc(
        required=[
            {"s": "self", "p": "works_at", "o": "Northwind Labs", "from": "2023-02"},
            {"s": "self", "p": "works_at", "o": "Northwind Labs", "from": "2025-03"},
            {"s": "self", "p": "role", "o": "Senior Data Engineer", "from": "2025-03"},
            {"s": "self", "p": "role", "o": "Data Engineer", "from": "2023-02", "to": "2025-03"},
            {"s": "self", "p": "lives_in", "o": "Porto", "from": 2024, "current": True},
            {"s": "self", "p": "speaks", "o": "Spanish"},
            {"s": "self", "p": "speaks", "o": "English"},
        ],
        optional=[
            {"s": "self", "p": "team", "o": "Data Platform team"},
            {
                "s": "self",
                "p": "worked_on",
                "o": "Atlas Migration",
                "alt": [{"s": "Atlas Migration", "p": "owner", "o": "self"}],
            },
            {"s": "Sam Okafor", "p": ["friend_of", "knows"], "o": "self"},
            {"s": "Valencia", "p": "located_in", "o": "Spain"},
        ],
        acceptable=[{"s": "self", "p": "has_skill", "o": "Airflow"}],
        entities=GOLD_ENTITIES,
        literals=["Airflow", "Data Platform team"],
    )
    return score_doc(gold, extracted(), seconds=12.5)


def test_score_doc_matches_one_to_one_and_checks_dates() -> None:
    s = scored()
    c = s.counts()
    assert (c["required"], c["required_matched"]) == (7, 5)
    assert (c["optional"], c["optional_matched"]) == (4, 2)
    # 10 extracted: 5 required + 2 optional + 1 acceptable match; Lisbon and the duty are extras
    assert (c["extracted"], c["extracted_matched"], c["unverified"]) == (10, 8, 2)
    problems = {m.gold.show(): m.date_problems for m in s.matches}
    works = next(k for k in problems if "works_at" in k)
    assert problems[works] == ["less precise start (expected 2023-02, got 2023)"]
    porto = next(k for k in problems if "Porto" in k)
    assert problems[porto] == ["ended but current (got end 2025)"]
    assert (c["dates_checked"], c["dates_ok"]) == (4, 2)
    assert c["queued"] == 1


def test_misses_are_diagnosed() -> None:
    s = scored()
    kinds = {m.gold.show(): (m.kind, m.detail) for m in s.misses}
    assert kinds["self · speaks · English"] == ("not extracted", "")
    team = kinds["self · team · Data Platform team"]
    assert team[0] == "wrong predicate" and "responsibility" in team[1]
    assert kinds["Valencia · located_in · Spain"] == ("subject has no page", "Valencia")
    # two stints, one extracted fact: it goes to the better-dated gold fact, the other is merged
    stint = next(v for k, v in kinds.items() if "works_at" in k)
    assert stint[0] == "merged into another fact"


def test_entities_literals_and_unexpected_pages() -> None:
    s = scored()
    by_name = {e.gold.name: e for e in s.entities}
    assert by_name["Northwind Labs"].type_ok
    assert by_name["Atlas Migration"].found is not None and not by_name["Atlas Migration"].type_ok
    assert by_name["Sam Okafor"].found is None  # only in a queued fact, no page
    assert s.literal_pages == [("Airflow", "projects/airflow")]
    assert {e.id for e in s.unexpected_pages} == {"places/lisbon", "people/alex-rivera"}
    c = s.counts()
    # Fabrikam is in no fact, so no page is expected for it; its page still has its type checked
    assert (c["entities"], c["entities_found"]) == (5, 3)
    assert (c["entities_typed"], c["entity_types_ok"]) == (4, 3)

    kinds = error_kinds([s])
    assert kinds["page for the owner"] == 1
    assert kinds["page not in gold"] == 1
    assert kinds["literal turned into a page"] == 1
    assert kinds["entity type project → org"] == 1
    assert kinds["date: ended but current"] == 1
    assert kinds["unverified lives_in"] == 1


def test_degree_school_link_is_checked_apart_from_matching() -> None:
    gold = gold_doc(
        optional=[
            {"s": "self", "p": "degree", "o": "MSc Data Science", "at": "Westbridge University"},
            {"s": "self", "p": "degree", "o": "BSc Mathematics", "at": "Westbridge University"},
            {"s": "self", "p": "degree", "o": "Preparatory classes", "at": "Calder College"},
        ],
        entities=[
            ("Westbridge University", "org", ["Westbridge"]),
            ("Calder College", "org", []),
        ],
    )
    x = Extracted(
        entities=[
            GotEntity("orgs/westbridge", "org", ("Westbridge",)),
            GotEntity("orgs/calder", "org", ("Calder College",)),
        ],
        facts=[
            GotFact("self", "degree", "MSc Data Science", at="[[orgs/westbridge]]"),
            GotFact("self", "degree", "BSc Mathematics", at="[[orgs/calder]]"),
            GotFact("self", "degree", "Preparatory classes"),
        ],
    )
    score = score_doc(gold, x)
    assert [m.link_problem for m in score.matches] == [
        "",
        "wrong school (expected Westbridge University, got Calder College)",
        "no school (expected Calder College)",
    ]
    assert score.counts()["optional_matched"] == 3  # matching ignores the link
    assert (score.counts()["links_checked"], score.counts()["links_ok"]) == (3, 1)
    assert error_kinds([score])["degree: wrong school"] == 1


def test_report_json_is_serialisable() -> None:
    data = report_json([scored()], {"model": "test"})
    text = json.dumps(data)
    assert data["overall"]["required_recall"] == round(5 / 7, 3)
    assert data["docs"][0]["metrics"]["seconds"] == 12.5
    assert "unverified" in text and data["error_kinds"]


def test_collect_reads_pages_history_and_queued_ops(tmp_vault: Vault) -> None:
    queued = [
        (CreateEntity(id="people/jordan-lee", type="person", title="Jordan Lee"), "relation"),
        (
            AddFact(
                subject="people/jordan-lee", p="friend_of", o="[[self]]", text="x", kind="stated"
            ),
            "relation",
        ),
    ]
    x = collect(tmp_vault, queued)
    assert any(e.id == "orgs/northwind-labs" and e.type == "org" for e in x.entities)
    assert any(e.id == "people/jordan-lee" and e.queued for e in x.entities)
    assert any(f.subject == "people/sam-okafor" and f.p == "friend_of" for f in x.facts)
    assert any(f.queued and f.subject == "people/jordan-lee" for f in x.facts)
    assert x.show(next(f for f in x.facts if f.queued)) == "Jordan Lee · friend_of · self (queued)"


class ScriptedIngestLLM(FakeLLM):
    """Answers every extraction with one scripted reply; any other call (a split plan) fails,
    so the pipeline falls back to splitting at headings."""

    def structured(self, messages, schema, *, temperature=None, max_tokens=2048):
        if schema is not Extraction:
            raise StructuredOutputError("no plan in this test", "")
        return Extraction.model_validate(
            {
                "reasoning": "A speaker bio.",
                "owner_names": ["Maya Lindqvist", "Maya"],
                "entities": [{"name": "Fernhill Health", "type": "org", "id": "", "aliases": []}],
                "facts": [
                    {
                        "subject": "self",
                        "predicate": "name",
                        "object": "Maya Lindqvist",
                        "text": "",
                        "start": "",
                        "end": "",
                    },
                    {
                        "subject": "self",
                        "predicate": "works_at",
                        "object": "Fernhill Health",
                        "text": "",
                        "start": "2021",
                        "end": "",
                    },
                    {
                        "subject": "self",
                        "predicate": "role",
                        "object": "Design Lead",
                        "text": "",
                        "start": "",
                        "end": "",
                    },
                ],
            }
        )


def test_cli_eval_ingest_with_a_scripted_model(
    tmp_path: Path, git_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(vault=tmp_path / "never-used")
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "make_llm", lambda _s: ScriptedIngestLLM())
    out = tmp_path / "run.json"
    result = CliRunner().invoke(
        cli.app, ["eval", "ingest", "--docs", "speaker", "--no-cache", "--json", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "interfaces-north-speaker.html" in result.output
    data = json.loads(out.read_text())
    assert [d["doc"] for d in data["docs"]] == ["interfaces-north-speaker.html"]
    assert data["docs"][0]["counts"]["required_matched"] == 3
    assert not settings.vault.exists()  # the configured vault is never touched
    assert EVAL_DIR.is_dir()
