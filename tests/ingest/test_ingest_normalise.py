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


def slipped_extraction() -> dict:
    """The slips a ~3B-active model made on a real homepage, replayed on the fictional persona."""
    return {
        "reasoning": "r",
        "owner_names": ["Alex Rivera", "Alex"],
        "entities": [
            entity("Northwind Labs", "org"),
            entity("Atlas Migration", "project"),
            entity("Pebble", "org"),  # an AI assistant typed as an organisation
            entity("CodeHarbor", "tool"),
            entity("Data Platform team", "org"),
            entity("TL4521", "project"),  # a dataset record id
            entity("Seville", "place"),
            entity("Porto", "place"),
            entity("Madrid", "place"),
        ],
        "facts": [
            fact("self", "works_at", "Northwind Labs", "Works at Northwind Labs", "2023-02"),
            # verb-led duties filed under the employer would merge into it and be lost
            fact(
                "self",
                "works_at",
                "Northwind Labs",
                "Works on pipelines that help analysts trust their numbers",
            ),
            fact("self", "works_at", "Northwind Labs", "Experiments with streaming ingestion."),
            fact("self", "works_at", "Data Platform team", "Is part of the Data Platform team"),
            fact("self", "works_at", "Atlas Migration", "Works on the Atlas Migration", "2026"),
            fact("self", "works_at", "Madrid", "Worked in Madrid"),
            fact("self", "worked_on", "Pebble", "Built plugins for Pebble"),
            fact("Pebble", "mentor_of", "self", "Was supervised by the owner"),
            fact("CodeHarbor", "uses", "Python", "Hosts code"),
            fact("self", "uses", "CodeHarbor", "Uses CodeHarbor in their toolkit"),
            fact("Atlas Migration", "uses", "TL4521", "Uses record TL4521"),
            # counters read before their script ran
            fact("self", "has_skill", "0 years of ML", "Has 0 years of ML"),
            fact("self", "degree", "MSc", "Has 0 master's degrees"),
            # pronouns and the owner's name in the text
            fact("self", "habit", "calisthenics", "Has calisthenics as his longest-running habit"),
            fact("self", "has_skill", "Python", "Uses Python in their toolkit"),
            fact("self", "role", "Data engineer", "Alex works as a data engineer", "2023-02"),
            fact("self", "values", "clear documentation", "Cares about clear documentation"),
            # a past city without dates, next to the current one
            fact("self", "lives_in", "Porto", "", "2024"),
            fact("self", "lives_in", "Seville", "Lived in Seville before"),
            fact("self", "lives_in", "Valencia", "Grew up in Valencia"),
        ],
    }


def test_guards_repair_or_drop_what_breaks_the_schema(fresh_vault: Vault) -> None:
    out = build(fresh_vault, slipped_extraction())
    creates = {op.id for op in out.changeset.ops if isinstance(op, CreateEntity)}
    # no pages for tools, products, record ids or teams; Madrid and Seville end up unused
    assert creates == {"orgs/northwind-labs", "projects/atlas-migration", "places/porto"}
    facts = [(f.subject, f.p, f.o, f.text) for f in facts_of(out.changeset.ops)]
    by_key = {(s, p, o): t for s, p, o, t in facts}

    assert ("self", "works_at", "[[orgs/northwind-labs]]") in by_key
    duties = sorted(o for s, p, o, _ in facts if p == "responsibility")
    assert duties == [
        "Experiments with streaming ingestion",
        "Works on pipelines that help analysts trust their numbers",
    ]
    assert by_key[("self", "team", "Data Platform team")] == "Is part of the Data Platform team"
    assert ("self", "worked_on", "[[projects/atlas-migration]]") in by_key
    assert by_key[("self", "worked_on", "Pebble")] == "Built plugins for Pebble"  # a literal
    assert by_key[("self", "has_skill", "CodeHarbor")] == "Skilled in CodeHarbor"
    assert ("projects/atlas-migration", "uses", "TL4521") in by_key
    assert not [f for f in facts if "0 " in f[2] or "0 " in f[3]]
    assert by_key[("self", "habit", "calisthenics")] == "Does calisthenics regularly"
    assert by_key[("self", "has_skill", "Python")] == "Skilled in Python"
    assert by_key[("self", "role", "Data engineer")] == "Works as Data engineer"
    assert by_key[("self", "values", "clear documentation")] == "Cares about clear documentation"
    assert by_key[("self", "lives_in", "[[places/porto]]")] == "Lives in Porto"
    assert by_key[("self", "note", "Lived in Seville")] == "Lived in Seville"
    assert by_key[("self", "hometown", "Valencia")] == "Grew up in Valencia"
    assert [o for s, p, o, _ in facts if p == "lives_in"] == ["[[places/porto]]"]
    assert not [f for f in facts if f[0] in ("Pebble", "CodeHarbor")]

    notes = "\n".join(out.notes)
    for expected in (
        "works_at + responsibility (describes the work): Works on pipelines",
        "works_at + responsibility (describes the work): Experiments with",
        "works_at → team ('Data Platform team' is a team)",
        "works_at → worked_on ('Atlas Migration' is a project)",
        "dropped works_at fact: 'Madrid' is a place",
        "worked_on: 'Pebble' is a org, kept as a literal value",
        "dropped mentor_of fact: its subject 'Pebble' is a org",
        "dropped uses fact about 'CodeHarbor' (a tool)",
        "uses → has_skill (the subject is not a project): CodeHarbor",
        "dropped has_skill fact with a zero count",
        "dropped degree fact with a zero count",
        "rewrote habit text (a pronoun)",
        "rewrote role text (the owner's name)",
        "past lives_in 'Seville' without an end date: kept as a note",
        "lives_in → hometown (grew up there): Valencia",
        "dropped entity places/seville: no fact refers to it",
    ):
        assert expected in notes, expected


def test_template_text() -> None:
    from brain.ingest.normalise import template_text, text_problem

    assert template_text("works_at", "Fabrikam Retail", past=True) == "Worked at Fabrikam Retail"
    assert template_text("team", "Platform", past=False) == "Is on the Platform team"
    assert template_text("habit", "go bouldering", past=False) == "Has the habit: go bouldering"
    assert template_text("responsibility", "weekly sales reporting", past=False) == (
        "Responsible for weekly sales reporting"
    )
    assert template_text("responsibility", "runs the on-call rotation", past=False) == (
        "Runs the on-call rotation"
    )
    lint = {"o_name": "Python", "owner_names": ["Alex Rivera", "Alex"], "about_self": True}
    assert text_problem("Writes Python daily", **lint) == ""
    assert text_problem("Writes code daily", **lint) == "no object"
    assert text_problem("Alex writes Python", **lint) == "the owner's name"
    assert text_problem("Writes Python " + "x" * 200, **lint) == "too long"


def test_vague_duplicates_and_mentions_dont_become_contradictions(fresh_vault: Vault) -> None:
    """Slips seen on a real homepage: a summary line with a vague year beside the career
    section's exact date, the team named as an employer, a country beside the city, a city as
    a school, and languages listed as skills."""
    extraction = {
        "reasoning": "Header summary and career section.",
        "owner_names": ["Alex Rivera"],
        "entities": [
            entity("Northwind Labs", "org"),
            entity("Data Platform", "org"),
            entity("Porto", "place"),
            entity("Portugal", "place"),
            entity("Lisbon", "place"),
        ],
        "facts": [
            fact("self", "works_at", "Northwind Labs", "", "2022"),
            fact("self", "works_at", "Northwind Labs", "", "2024-04"),
            fact("self", "role", "Data engineer", "", "2024-04"),
            fact("self", "role", "Engineer", ""),
            fact("self", "team", "Data Platform", ""),
            fact("self", "works_at", "Data Platform", ""),
            fact("self", "lives_in", "Porto", "", "2024-04"),
            fact("self", "lives_in", "Portugal", ""),
            fact("self", "studied_at", "Lisbon", ""),
            fact("self", "has_skill", "Spanish", ""),
            fact("self", "has_skill", "Python", ""),
        ],
    }
    out = build(fresh_vault, extraction)
    facts = {(f.p, f.o): f for f in facts_of(out.changeset.ops)}
    assert facts[("works_at", "[[orgs/northwind-labs]]")].valid_from == "2024-04"
    assert ("role", "Engineer") not in facts
    assert ("works_at", "[[orgs/data-platform]]") not in facts
    assert not any(p == "lives_in" and "portugal" in o for p, o in facts)
    assert not any(p == "studied_at" for p, _ in facts)
    assert ("speaks", "Spanish") in facts and ("has_skill", "Python") in facts
    creates = {op.id for op in out.changeset.ops if isinstance(op, CreateEntity)}
    assert "orgs/data-platform" not in creates and "places/porto" in creates


def test_a_duty_filed_under_its_only_employer_keeps_the_employer(fresh_vault: Vault) -> None:
    extraction = {
        "reasoning": "A past internship, written as what the work was.",
        "owner_names": ["Alex Rivera"],
        "entities": [entity("Fabrikam", "org")],
        "facts": [
            fact("self", "works_at", "Fabrikam", "Did demand forecasting", "2021-02", "2021-08")
        ],
    }
    out = build(fresh_vault, extraction)
    facts = {(f.p, f.o): f for f in facts_of(out.changeset.ops)}
    works = facts[("works_at", "[[orgs/fabrikam]]")]
    assert (works.valid_from, works.valid_to) == ("2021-02", "2021-08")
    assert ("responsibility", "Did demand forecasting") in facts
    assert "orgs/fabrikam" in {op.id for op in out.changeset.ops if isinstance(op, CreateEntity)}


def test_starts_dated_to_the_ingest_day_are_dropped(fresh_vault: Vault) -> None:
    from datetime import date

    extraction = {
        "reasoning": "Current job.",
        "owner_names": [],
        "entities": [entity("Northwind Labs", "org")],
        "facts": [
            fact("self", "role", "Data engineer", "", "2026-10-25"),
            fact("self", "role", "Data engineer", "", "2024-04"),
            fact("self", "works_at", "Northwind Labs", "", "2026-10"),
        ],
    }
    out = build_changeset(
        [Extraction.model_validate(extraction)], fresh_vault, src=SRC, today=date(2026, 10, 25)
    )
    facts = {(f.p, f.o): f for f in facts_of(out.changeset.ops)}
    assert facts[("role", "Data engineer")].valid_from == "2024-04"
    assert facts[("works_at", "[[orgs/northwind-labs]]")].valid_from == "2026-10"  # a month is kept


def test_entity_kinds_from_checks_never_touch_the_fact_kind(fresh_vault: Vault) -> None:
    """Regression: a loop over the checks' kinds once shadowed ``kind`` (stated/observed), and
    facts were built with kind='org'."""
    extraction = Extraction.model_validate(
        {
            "reasoning": "r",
            "owner_names": [],
            "entities": [
                entity("Data Platform", "org"),
                entity("Lakehouse Cloud", "project"),
                entity("Harbor Guild", "project"),
            ],
            "facts": [
                fact("self", "works_at", "Northwind Labs", ""),
                fact("self", "member_of", "Data Platform", ""),
                fact("self", "worked_on", "Lakehouse Cloud", ""),
                fact("self", "member_of", "Harbor Guild", ""),
            ],
        }
    )
    out = build_changeset(
        [extraction],
        fresh_vault,
        src=SRC,
        kind="observed",
        entity_kinds={
            "data platform": "team",
            "lakehouse cloud": "tool",
            "harbor guild": "org",
            "northwind labs": "org",
        },
    )
    facts = facts_of(out.changeset.ops)
    assert facts and {f.kind for f in facts} == {"observed"}
    creates = {op.id: op.type for op in out.changeset.ops if isinstance(op, CreateEntity)}
    assert creates == {"orgs/northwind-labs": "org", "orgs/harbor-guild": "org"}
    assert ("team", "Data Platform") in {(f.p, f.o) for f in facts}
    assert any("'Lakehouse Cloud' is a tool" in n for n in out.notes)
