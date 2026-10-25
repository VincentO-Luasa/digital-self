"""From flat extractions to vault operations: resolution, dates, merging, direction, regions."""

from __future__ import annotations

from conftest import entity, fact, homepage_extraction  # type: ignore[import-not-found]

from brain.ingest.extract import Extraction
from brain.ingest.normalise import build_changeset
from brain.vault.ops import AddAlias, AddFact, CreateEntity
from brain.vault.store import Vault

SRC = "raw/docs/2026-10-25-alex-rivera.html"


def build(vault: Vault, *extractions: dict):
    return build_changeset([Extraction.model_validate(e) for e in extractions], vault, src=SRC)


def facts_of(ops) -> list[AddFact]:
    return [op for op in ops if isinstance(op, AddFact)]


def test_homepage_on_a_fresh_vault(fresh_vault: Vault) -> None:
    out = build(fresh_vault, homepage_extraction())
    cs = out.changeset
    assert (cs.origin, cs.src) == ("ingest", [SRC])
    creates = {op.id: op for op in cs.ops if isinstance(op, CreateEntity)}
    # Westbridge comes from a literal works/studied object; Airflow and Lisbon are dropped
    # (a tool is a literal value; nothing points at Lisbon).
    assert set(creates) == {
        "orgs/northwind-labs",
        "orgs/fabrikam-retail",
        "orgs/westbridge-university",
        "projects/atlas-migration",
        "people/sam-okafor",
    }
    assert creates["orgs/northwind-labs"].regions == ["career.current_job"]
    assert creates["orgs/fabrikam-retail"].regions == ["career.history"]
    assert creates["orgs/westbridge-university"].regions == ["education"]
    assert creates["people/sam-okafor"].regions == ["relationships.friends"]
    assert creates["projects/atlas-migration"].aliases == ["Atlas"]
    aliases = [op.alias for op in cs.ops if isinstance(op, AddAlias)]
    assert aliases == ["Alex Rivera", "Alex"]

    facts = {(f.subject, f.p, f.o): f for f in facts_of(cs.ops)}
    assert facts[("self", "name", "Alex Rivera")].o == "Alex Rivera"  # a value, not [[self]]
    works = facts[("self", "works_at", "[[orgs/northwind-labs]]")]
    assert (works.valid_from, works.valid_to) == ("2023-02", None)  # union of both stints
    role = facts[("self", "role", "Data Engineer")]
    assert (role.valid_from, role.valid_to) == ("2023-02", "2025-03")
    assert facts[("self", "has_skill", "Airflow")]  # literal, no page for a tool
    assert facts[("self", "lives_in", "Porto")].text == "Lives in Porto"
    # (self, owner, project) is turned around to (project, owner, self)
    assert ("projects/atlas-migration", "owner", "[[self]]") in facts
    assert ("people/sam-okafor", "friend_of", "[[self]]") in facts
    assert all(f.kind == "stated" for f in facts.values())
    # current role first among single-valued role facts
    roles = [f.o for f in facts_of(cs.ops) if f.p == "role"]
    assert roles == ["Senior Data Engineer", "Data Engineer"]

    notes = "\n".join(out.notes)
    assert "unknown subject 'Somebody Unknown'" in notes
    assert "subject is its object" in notes
    assert "merged duplicate works_at" in notes
    assert "dropped entity places/lisbon" in notes


def test_reuses_known_entities_and_owner_names(tmp_vault: Vault) -> None:
    out = build(
        tmp_vault,
        {
            "reasoning": "r",
            "owner_names": ["Alex Rivera"],
            "entities": [
                entity("Northwind", "org", "orgs/northwind-labs"),
                entity("Atlas", "project"),  # alias of an existing page, id left empty
            ],
            "facts": [
                fact("Alex Rivera", "works_at", "Northwind", "Works at Northwind"),
                fact("Atlas", "status", "testing", "Is in testing"),
                fact("Sam", "knows", "Alex", "Knows Alex"),
            ],
        },
    )
    ops = out.changeset.ops
    assert not [op for op in ops if isinstance(op, (CreateEntity, AddAlias))]
    triples = [(f.subject, f.p, f.o) for f in facts_of(ops)]
    assert triples == [
        ("self", "works_at", "[[orgs/northwind-labs]]"),
        ("projects/atlas-migration", "status", "testing"),
        ("people/sam-okafor", "knows", "[[self]]"),
    ]


def test_observed_kind_and_bad_dates(fresh_vault: Vault) -> None:
    extraction = Extraction.model_validate(
        {
            "reasoning": "r",
            "owner_names": [],
            "entities": [],
            "facts": [
                fact("self", "speaks", "Spanish", "Speaks Spanish", "spring 2020"),
                fact("self", "lives_in", "Porto", "Lives in Porto", "2024", "2019"),
            ],
        }
    )
    out = build_changeset([extraction], fresh_vault, src=SRC, kind="observed")
    speaks, lives = facts_of(out.changeset.ops)
    assert speaks.kind == "observed" and speaks.valid_from is None
    assert (lives.valid_from, lives.valid_to) == ("2024", None)
    assert any("unreadable start date 'spring 2020'" in n for n in out.notes)
    assert any("end 2019 before start 2024" in n for n in out.notes)
