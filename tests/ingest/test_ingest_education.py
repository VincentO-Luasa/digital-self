"""A degree is linked to the school its line names (``at::``), never paired by dates alone."""

from __future__ import annotations

from conftest import entity, fact  # type: ignore[import-not-found]

from brain.ingest.education import School, degree_school
from brain.ingest.extract import Extraction
from brain.ingest.normalise import build_changeset
from brain.vault.ops import AddFact
from brain.vault.store import Vault

SRC = "raw/docs/2026-10-25-alex-rivera.html"
OSTERVALE = School("orgs/ostervale", ("Ostervale Institute of Technology", "Ostervale"))
WESTBRIDGE = School("orgs/westbridge", ("Westbridge University", "Westbridge"))
CALDER = School("orgs/calder", ("Calder College",))
SCHOOLS = [OSTERVALE, WESTBRIDGE, CALDER]

TIMELINE = """## Education

- 2021 — 2022 · Westbridge University · MSc in Data Science, double degree with Ostervale Institute of Technology.
- 2019 — 2022 · Ostervale Institute of Technology · General engineering degree, specialisation in databases.
- 2017 — 2019 · Calder College · Preparatory classes in mathematics and physics.
"""


def link(value: str, text: str, dates=(None, None), schools=SCHOOLS) -> str | None:
    return degree_school(value, dates, schools, text)[0]


def test_timeline_rows_link_each_degree_to_its_own_row() -> None:
    assert link("MSc in Data Science", TIMELINE) == "orgs/westbridge"  # not the double degree
    assert link("General engineering degree", TIMELINE) == "orgs/ostervale"  # not the prep year
    assert link("Preparatory classes in mathematics and physics", TIMELINE) == "orgs/calder"


def test_cv_line_prose_and_line_above() -> None:
    cv = "MSc Data Science, Westbridge University, Porto, 2018 – 2020\nThesis: streaming joins"
    assert link("MSc Data Science", cv) == "orgs/westbridge"
    prose = (
        "I did preparatory classes at Calder College, then a master's in data science at "
        "Westbridge University (2018–2020)."
    )
    assert link("Master's in data science", prose) == "orgs/westbridge"  # prose: the one after
    above = "Westbridge University, Porto\nMSc Data Science, 2018 – 2020"
    assert link("MSc Data Science", above) == "orgs/westbridge"


def test_same_dates_break_a_tie() -> None:
    text = "Westbridge University and Ostervale Institute of Technology: joint MSc in Data Science"
    schools = [School("orgs/westbridge", WESTBRIDGE.names, ("2018", "2020")), OSTERVALE]
    assert link("MSc in Data Science", text, ("2018", "2020"), schools) == "orgs/westbridge"


def test_no_link_without_evidence() -> None:
    assert link("PhD in Astronomy", TIMELINE) is None  # the degree is not in the text
    assert link("MSc Data Science", "Holds an MSc in Data Science.") is None  # no school named
    assert link("MSc Data Science", TIMELINE, schools=[]) is None


def education_extraction() -> dict:
    return {
        "reasoning": "education",
        "owner_names": ["Alex Rivera"],
        "entities": [
            entity("Westbridge University", "org"),
            entity("Ostervale Institute of Technology", "org"),
            entity("Calder College", "org"),
        ],
        "facts": [
            fact("self", "studied_at", "Westbridge University", "", "2021", "2022"),
            fact("self", "degree", "MSc in Data Science", "", "2021", "2022"),
            fact("self", "studied_at", "Ostervale Institute of Technology", ""),  # no dates
            fact("self", "degree", "General engineering degree", "", "2019", "2022"),
            fact("self", "studied_at", "Calder College", "", "2017", "2019"),
        ],
    }


def test_normalise_links_degrees_and_shares_dates(fresh_vault: Vault) -> None:
    out = build_changeset(
        [Extraction.model_validate(education_extraction())],
        fresh_vault,
        src=SRC,
        texts=[TIMELINE],
    )
    facts = {f.o: f for f in out.changeset.ops if isinstance(f, AddFact)}
    msc = facts["MSc in Data Science"]
    assert msc.extra == {"at": "[[orgs/westbridge-university]]"}
    assert msc.text == "Holds the degree MSc in Data Science from Westbridge University"
    engineering = facts["General engineering degree"]
    assert engineering.extra == {"at": "[[orgs/ostervale-institute-of-technology]]"}
    school = facts["[[orgs/ostervale-institute-of-technology]]"]
    assert (school.valid_from, school.valid_to) == ("2019", "2022")  # from its degree's row
    notes = "\n".join(out.notes)
    assert "degree 'MSc in Data Science' at Westbridge University (its line" in notes
    assert "studied_at Ostervale Institute of Technology: dates of its degree" in notes


def test_normalise_without_texts_leaves_degrees_alone(fresh_vault: Vault) -> None:
    out = build_changeset([Extraction.model_validate(education_extraction())], fresh_vault, src=SRC)
    assert all(not f.extra for f in out.changeset.ops if isinstance(f, AddFact))
