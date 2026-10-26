"""The document brief: the call, grounding of its dates, the prompt block every part shares, and
what normalisation takes from it (one entity per name, structural facts, homes, stints). Fictional
persona only (Alex Rivera)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from conftest import (  # type: ignore[import-not-found]
    DOCS,
    brief,
    entity,
    fact,
    homepage_brief,
    homepage_extractions,
    homepage_plan,
    name,
)

from brain.ingest import ingest
from brain.ingest.brief import (
    Brief,
    brief_source,
    build_messages,
    clean_brief,
    ground,
    header,
    homes,
    lookup,
    make_brief,
    name_index,
    render,
    units,
)
from brain.ingest.chunking import Chunk
from brain.ingest.dates import years_in
from brain.ingest.extract import Extraction
from brain.ingest.extract import build_messages as extract_messages
from brain.ingest.normalise import build_changeset
from brain.ingest.pipeline import BRIEF_USES
from brain.ingest.readers import load_document
from brain.llm import StructuredOutputError
from brain.llm.fake import FakeLLM
from brain.vault.ops import AddFact, CreateEntity
from brain.vault.store import Vault

TODAY = date(2026, 10, 25)
HOMEPAGE = DOCS / "alex-rivera.html"


def homepage_text() -> str:
    return load_document(HOMEPAGE).text


def grounded_homepage_brief() -> tuple[Brief, list[str]]:
    notes: list[str] = []
    raw = Brief.model_validate(homepage_brief())
    return ground(clean_brief(raw), homepage_text(), notes), notes


# --- the call and its input ---------------------------------------------------------------------


def test_brief_reads_the_whole_document_or_an_outline(monkeypatch) -> None:
    text = homepage_text()
    chunks = [Chunk(0, ("About",), "Hi, I'm Alex. " * 50), Chunk(1, ("Career",), "Northwind " * 80)]
    assert brief_source(text, chunks) == (text.strip(), True)
    monkeypatch.setattr("brain.ingest.brief.FULL_TEXT_TOKENS", 10)
    source, full = brief_source(text, chunks)
    assert not full
    assert source.startswith("## Part 1: About\nHi, I'm Alex.") and "## Part 2: Career" in source
    assert len(source) < len(chunks[0].text) + len(chunks[1].text)
    user = build_messages(source, doc_title="Alex", full_text=False)[1]["content"]
    assert "DOCUMENT OUTLINE (each part's first lines)" in user


def test_make_brief_cleans_and_grounds_and_a_failure_is_not_fatal() -> None:
    llm = FakeLLM(structured_responses=[homepage_brief()])
    report = make_brief(llm, homepage_text(), [], doc_title="Alex Rivera")
    [call] = llm.calls
    assert call[1]["schema"] is Brief
    assert call[1]["messages"][0]["content"].startswith("You read a whole document once")
    assert report.brief is not None and report.full_text and report.seconds >= 0
    assert report.notes == [
        "brief: home Lisbon: start 2019 not stated with it; dropped",
    ]
    failed = make_brief(
        FakeLLM(structured_responses=[StructuredOutputError("bad", "{")]), "x", [], doc_title=""
    )
    assert failed.brief is None and failed.notes == ["brief: none (bad)"]


def test_clean_brief_keeps_names_not_descriptions() -> None:
    raw = Brief.model_validate(
        brief(
            subject="Alex Rivera",
            names=[
                name("Alex Rivera", "person"),  # the subject is not a glossary name
                name(
                    "Atlas Migration",
                    "project",
                    aliases=["Atlas", "Migration project", "Atlas 2", "my team", "atlas migration"],
                ),
                name("Atlas Migration", "project"),  # repeated
                name(" ", "org"),
            ],
        )
    )
    cleaned = clean_brief(raw)
    assert [(n.name, n.aliases) for n in cleaned.names] == [
        ("Atlas Migration", ["Atlas", "Atlas 2"])
    ]


def test_ground_blanks_dates_the_document_does_not_state() -> None:
    text = (
        "I grew up in Valencia. I've lived in Porto since 2024, after five years in Lisbon.\n\n"
        "### Fabrikam Retail — Lisbon\n\n**Data Analyst**, September 2019 – January 2023.\n\n"
        "Our Madrid office opened in 2015."
    )
    raw = Brief.model_validate(
        brief(
            homes=[
                {"city": "Madrid", "start": "2015", "end": "2019"},  # an office, not a home
                {"city": "Lisbon", "start": "2019", "end": "2024"},  # 2019: a job, not a move
                {"city": "Porto", "start": "2024", "end": "now"},
            ],
            periods=[
                {
                    "kind": "job",
                    "title": "Data Analyst",
                    "org": "Fabrikam Retail",
                    "start": "2019-09",
                    "end": "2022",
                },
            ],
        )
    )
    notes: list[str] = []
    out = ground(raw, text, notes)
    assert [(h.city, h.start, h.end) for h in out.homes] == [
        ("Lisbon", "", "2024"),
        ("Porto", "2024", "now"),
    ]
    assert [(p.start, p.end) for p in out.periods] == [("2019-09", "")]
    assert "brief: dropped home 'Madrid': no sentence says the subject lives there" in notes
    # wrapped prose is one sentence; short lines (a CV's "Org, City" / "Title, dates") are not
    assert units(
        "I moved to Lisbon in the spring of 2019 to join a small data team at\nFabrikam."
    ) == ["I moved to Lisbon in the spring of 2019 to join a small data team at Fabrikam."]
    assert units("Fabrikam Retail, Lisbon\nData Analyst, Sep 2019") == [
        "Fabrikam Retail, Lisbon",
        "Data Analyst, Sep 2019",
    ]


@pytest.mark.parametrize(
    ("line", "years"),
    [
        ("Data Engineer  Northwind Labs  Lisbon  Sep 21 – July 22", {2021, 2022}),
        ("Analyst, Fabrikam Retail, April 24 – Present", {2024}),
        ("Nov 20 – Feb 21 · Fabrikam Retail", {2020, 2021}),
        ("Sept. 21 to Aug. 23", {2021, 2023}),
        ("Fabrikam Retail, 09/2021 – 03/2023", {2021, 2023}),
        ("Fabrikam Retail, 09/21 – 03/23", {2021, 2023}),
        ("Fabrikam Retail (2021-09 to 2023-03)", {2021, 2023}),
        ("Fabrikam Retail, Sep '21 – ’23", {2021, 2023}),
        ("Joined Fabrikam Retail on May 21, 2024", {2024}),  # a day of the month, not '21
        ("Three of the 21 analysts at Fabrikam Retail", set()),
    ],
)
def test_years_in_resume_date_forms(line: str, years: set[int]) -> None:
    assert years_in(line, TODAY) == years


def test_ground_keeps_resume_style_dates() -> None:
    """A CV writes "Sep 21 – July 22" with the organisation on the same line; the brief's
    "2021-09" is stated there. An end "now" is not checked."""
    text = (
        "EXPERIENCE\n"
        "Data Engineer   Northwind Labs   Lisbon   April 24 – Present\n"
        "Analyst   Fabrikam Retail   Porto   Sep 21 – July 22\n"
        "Intern   Contoso Bank   Porto   Nov 20 – Feb 21\n"
        "EDUCATION\n"
        "MSc Data Science, Porto Business School, Porto, 09/2019 – 06/2021\n"
        "Thesis: streaming joins at scale on a budget for small teams\n"
        "BSc Mathematics, University of Coimbra, Coimbra, 2016 – 2019"
    )
    periods = [
        ("job", "Data Engineer", "Northwind Labs", "2024-04", "now"),
        ("job", "Analyst", "Fabrikam Retail", "2021-09", "2022-07"),
        ("job", "Intern", "Contoso Bank", "2020-11", "2021-02"),
        ("study", "MSc Data Science", "Porto Business School", "2019-09", "2021-06"),
        ("study", "BSc Mathematics", "University of Coimbra", "2017", "2019"),  # 2017: wrong
    ]
    raw = Brief.model_validate(
        brief(
            periods=[
                {"kind": k, "title": t, "org": o, "start": s, "end": e} for k, t, o, s, e in periods
            ]
        )
    )
    notes: list[str] = []
    out = ground(raw, text, notes)
    assert [(p.start, p.end) for p in out.periods] == [
        ("2024-04", "now"),
        ("2021-09", "2022-07"),
        ("2020-11", "2021-02"),
        ("2019-09", "2021-06"),
        ("", "2019"),
    ]
    assert notes == [
        "brief: BSc Mathematics at University of Coimbra: start 2017 not stated; dropped"
    ]
    # each CV line is its own unit: the MSc line is not joined with the thesis and the BSc line
    assert "BSc Mathematics, University of Coimbra, Coimbra, 2016 – 2019" in units(text)
    # a timeline row starting with a range is not joined to the line above it
    assert units(
        "Lead Product Designer at Northwind Labs in Lisbon, where I build\n"
        "2019 – 2021 · Fabrikam Retail · Analyst"
    ) == [
        "Lead Product Designer at Northwind Labs in Lisbon, where I build",
        "2019 – 2021 · Fabrikam Retail · Analyst",
    ]
    # a record line is one unit: its organisation and date stay together
    assert units('Fabrikam Retail,Analyst,"Built reports. Ran the BI team.",Jan 2024,') == [
        'Fabrikam Retail,Analyst,"Built reports. Ran the BI team.",Jan 2024,'
    ]
    # wrapped prose that goes on in lower case after a year is still one sentence
    assert units(
        "I joined Fabrikam Retail in Porto as an analyst in September 2019\nand left in 2021."
    ) == ["I joined Fabrikam Retail in Porto as an analyst in September 2019 and left in 2021."]


def test_homes_chain_ends_and_the_last_is_current() -> None:
    b, _ = grounded_homepage_brief()
    assert [(h.city, h.start, h.end, h.current) for h in homes(b)] == [
        ("Valencia", None, None, False),
        ("Seville", None, None, False),  # Lisbon's start was not stated: no end for Seville
        ("Lisbon", None, "2024", False),
        ("Porto", "2024", None, True),
    ]


def test_only_the_last_home_is_current() -> None:
    b = Brief.model_validate(
        brief(
            homes=[
                {"city": "Lisbon", "start": "2019", "end": "now"},
                {"city": "Porto", "start": "2024", "end": "now"},
            ]
        )
    )
    assert [(h.city, h.end, h.current) for h in homes(b)] == [
        ("Lisbon", "2024", False),
        ("Porto", None, True),
    ]


def test_lookup_names_aliases_and_variants() -> None:
    b, _ = grounded_homepage_brief()
    index = name_index(b)
    assert lookup(index, "atlas").name == "Atlas Migration"
    assert lookup(index, "Atlas Migration project").name == "Atlas Migration"
    assert lookup(index, "Atlas Migration 2.0").name == "Atlas Migration"
    assert lookup(index, "Atlas Migration Team") is None  # a capitalised word is another name
    assert lookup(index, "Airflow") is not None and lookup(index, "Airflow pipelines") is None


# --- the prompt block -------------------------------------------------------------------------


def test_every_part_shares_the_brief_prefix() -> None:
    b, _ = grounded_homepage_brief()
    block = render(b)
    assert block.startswith(
        "About: Alex Rivera (homepage, mixed voice)\nTimeline (oldest first):\n"
    )
    # no dates: the extractor takes them from the excerpt, not from the brief's rounded years
    assert "- job: Senior Data Engineer at Northwind Labs\n" in block and "2025" not in block
    assert "- home: Lisbon\n" in block
    dated = render(b, dates=True)
    assert "- job 2025-03–now: Senior Data Engineer at Northwind Labs" in dated
    assert "- home ?–2024: Lisbon" in dated  # the unstated start is not shown
    assert "- Atlas Migration (project of Northwind Labs; also Atlas): project" in block
    assert "- Northwind Labs (org in Porto; also Northwind): employer" in block
    assert "- Airflow (tool)\n" in block  # role "other" is left out
    chunks = [Chunk(0, ("About",), "Hi, I'm Alex."), Chunk(1, ("Career",), "Northwind Labs")]
    users = [
        extract_messages(c, doc_title="Alex", known=[], owner_names=[], today=TODAY, brief=block)[
            1
        ]["content"]
        for c in chunks
    ]
    prefix = (
        f"Today is 2026-10-25.\nDocument: Alex\n\nDOCUMENT BRIEF (the whole document):\n{block}\n\n"
    )
    assert all(u.startswith(prefix) for u in users)
    assert users[0][len(prefix) :].startswith("Section: About\n")
    # without a brief the request is unchanged
    plain = extract_messages(chunks[0], doc_title="Alex", known=[], owner_names=[], today=TODAY)
    assert plain[1]["content"].startswith("Today is 2026-10-25.\nDocument: Alex\nSection: About")
    assert header(b) == "About: Alex Rivera (homepage, mixed voice)"


# --- normalisation ------------------------------------------------------------------------------


def extraction(entities: list[dict], facts: list[dict]) -> Extraction:
    return Extraction.model_validate(
        {"reasoning": "r", "owner_names": ["Alex Rivera"], "entities": entities, "facts": facts}
    )


def changeset(fresh_vault: Vault, extractions: list[Extraction], b: Brief | None):
    out = build_changeset(extractions, fresh_vault, src="raw/docs/x.md", today=TODAY, brief=b)
    created = {op.id: op for op in out.changeset.ops if isinstance(op, CreateEntity)}
    facts = [op for op in out.changeset.ops if isinstance(op, AddFact)]
    return created, facts, out.notes


def triples(facts: list[AddFact]) -> set[tuple[str, str, str]]:
    return {(f.subject, f.p, f.o) for f in facts}


def test_variant_names_become_one_entity(fresh_vault: Vault) -> None:
    b, _ = grounded_homepage_brief()
    parts = [
        extraction(
            [entity("Atlas", "project")],
            [fact("self", "worked_on", "Atlas", "Leads Atlas", "2026")],
        ),
        extraction(
            [entity("Atlas Migration project", "project"), entity("Northwind Labs", "org")],
            [
                fact("Atlas Migration project", "uses", "Airflow", ""),
                fact("self", "works_at", "Northwind Labs", "", "2023-02"),
            ],
        ),
    ]
    created, facts, notes = changeset(fresh_vault, parts, b)
    assert "projects/atlas-migration" in created
    assert not any("atlas" in i and i != "projects/atlas-migration" for i in created)
    assert {"Atlas", "Atlas Migration project"} <= set(created["projects/atlas-migration"].aliases)
    assert ("projects/atlas-migration", "uses", "Airflow") in triples(facts)
    assert "brief: 'Atlas' is 'Atlas Migration'" in notes
    # without the brief they stay two pages
    created, _, _ = changeset(fresh_vault, [p.model_copy(deep=True) for p in parts], None)
    assert {"projects/atlas", "projects/atlas-migration-project"} <= set(created)


def test_a_variant_of_another_type_is_not_merged(fresh_vault: Vault) -> None:
    b = Brief.model_validate(brief(names=[name("Atlas", "place", "other")]))
    parts = [
        extraction(
            [entity("Atlas Migration", "project")],
            [fact("self", "worked_on", "Atlas Migration", "")],
        )
    ]
    created, _, _ = changeset(fresh_vault, parts, b)
    assert "projects/atlas-migration" in created


def test_structure_from_parents_and_roles(fresh_vault: Vault) -> None:
    b, _ = grounded_homepage_brief()
    b.names.append(
        Brief.model_validate(brief(names=[name("Harbor Guild", "org", "member")])).names[0]
    )
    b.names.append(
        Brief.model_validate(brief(names=[name("Bel Ortiz", "person", "family")])).names[0]
    )
    parts = [
        extraction(
            [
                entity("Atlas Migration", "project"),
                entity("Tidepool", "project"),
                entity("Northwind Labs", "org"),
                entity("Sam Okafor", "person"),
                entity("Bel Ortiz", "person"),
            ],
            [
                fact("self", "works_at", "Northwind Labs", "", "2023-02"),
                fact("self", "worked_on", "Atlas Migration", "", "2026"),
                fact("self", "worked_on", "Tidepool", "", "2022"),
                fact("Sam Okafor", "knows", "self", ""),
                fact("Bel Ortiz", "friend_of", "self", ""),
                fact("Sam Okafor", "colleague_of", "self", ""),  # stated: stays
                fact("self", "responsibility", "Mentors juniors at Harbor Guild", "", "2021"),
                fact("Porto", "located_in", "Northwind Labs", ""),  # the wrong way round
            ],
        )
    ]
    created, facts, notes = changeset(fresh_vault, parts, b)
    got = triples(facts)
    assert ("projects/atlas-migration", "part_of", "[[orgs/northwind-labs]]") in got
    assert not any(s == "projects/tidepool" and p == "part_of" for s, p, _ in got)  # side project
    assert ("orgs/northwind-labs", "located_in", "[[places/porto]]") in got  # turned around
    assert not any(s == "places/porto" for s, _, _ in got), [t for t in got if "porto" in t[0]]
    assert "places/porto" in created and created["places/porto"].type == "place"
    member = next(f for f in facts if f.p == "member_of")
    assert (member.subject, member.o, member.valid_from) == (
        "self",
        "[[orgs/harbor-guild]]",
        "2021",
    )
    assert ("people/sam-okafor", "friend_of", "[[self]]") in got  # vague knows → the brief's
    assert ("people/bel-ortiz", "family_of", "[[self]]") in got  # a sister read as a friend
    assert ("people/sam-okafor", "friend_of", "[[self]]") in got
    assert "brief: turned located_in around: Northwind Labs is in Porto" in notes
    # Fabrikam Retail and Lisbon are named by no fact: no pages, no notes about dropping them
    assert "orgs/fabrikam-retail" not in created
    assert not any("fabrikam" in n for n in notes)


def test_parents_the_brief_does_not_list_are_typed_by_the_child(fresh_vault: Vault) -> None:
    b = Brief.model_validate(
        brief(
            names=[
                name("Calder College", "org", "school", "Lowmoor"),
                name("Tidepool", "project", "project", "Harbor Labs"),
            ]
        )
    )
    parts = [
        extraction(
            [entity("Calder College", "org"), entity("Tidepool", "project")],
            [
                fact("self", "studied_at", "Calder College", ""),
                fact("self", "worked_on", "Tidepool", ""),
            ],
        )
    ]
    created, facts, _ = changeset(fresh_vault, parts, b)
    assert created["places/lowmoor"].type == "place" and created["orgs/harbor-labs"].type == "org"
    assert ("orgs/calder-college", "located_in", "[[places/lowmoor]]") in triples(facts)


def test_homes_date_past_cities_and_add_missing_ones(fresh_vault: Vault) -> None:
    b, _ = grounded_homepage_brief()
    parts = [
        extraction(
            [entity("Porto", "place"), entity("Lisbon", "place"), entity("Valencia", "place")],
            [
                fact("self", "lives_in", "Porto", "Lives in Porto", "2024"),
                fact("self", "hometown", "Valencia", ""),
            ],
        )
    ]
    _, facts, notes = changeset(fresh_vault, parts, b)
    lives = {(f.o, f.valid_from, f.valid_to) for f in facts if f.p == "lives_in"}
    assert lives == {("[[places/porto]]", "2024", None), ("[[places/lisbon]]", None, "2024")}
    assert "brief: lives_in Lisbon (?–2024)" in notes
    # a past city the extraction states without an end gets the brief's (it is not the last home)
    parts = [
        extraction(
            [entity("Lisbon", "place")],
            [
                fact("self", "lives_in", "Lisbon", "Lived in Lisbon"),
                fact("self", "lives_in", "Porto", "Lives in Porto", "2024"),
            ],
        )
    ]
    _, facts, notes = changeset(fresh_vault, parts, b)
    lisbon = next(f for f in facts if f.p == "lives_in" and "lisbon" in f.o)
    assert lisbon.valid_to == "2024" and "brief: lives_in Lisbon: end 2024 (moved on)" in notes
    assert not any(f.p == "note" for f in facts)  # no longer a past city without an end


def test_the_last_home_never_gets_an_end(fresh_vault: Vault) -> None:
    b = Brief.model_validate(brief(homes=[{"city": "Porto", "start": "", "end": "2025"}]))
    parts = [extraction([], [fact("self", "lives_in", "Porto", "", "2024")])]
    _, facts, _ = changeset(fresh_vault, parts, b)
    [porto] = [f for f in facts if f.p == "lives_in"]
    assert porto.valid_to is None


def test_separate_stints_stay_separate_only_when_the_brief_shows_them(fresh_vault: Vault) -> None:
    stints = [
        fact("self", "works_at", "Northwind Labs", "", "2014-06", "2014-08"),  # an internship
        fact("self", "works_at", "Northwind Labs", "", "2017-08", "2021-01"),
        fact("self", "works_at", "Northwind Labs", "", "2021-02"),  # touching: the same stint
    ]
    b = Brief.model_validate(
        brief(
            periods=[
                {
                    "kind": "job",
                    "title": "Intern",
                    "org": "Northwind Labs",
                    "start": "2014-06",
                    "end": "2014-08",
                },
                {
                    "kind": "job",
                    "title": "Engineer",
                    "org": "Northwind Labs",
                    "start": "2017-08",
                    "end": "now",
                },
            ]
        )
    )
    parts = [extraction([entity("Northwind Labs", "org")], stints)]
    _, facts, _ = changeset(fresh_vault, parts, b)
    spans = sorted((f.valid_from, f.valid_to) for f in facts if f.p == "works_at")
    assert spans == [("2014-06", "2014-08"), ("2017-08", None)]
    parts = [extraction([entity("Northwind Labs", "org")], [dict(f) for f in stints])]
    _, facts, _ = changeset(fresh_vault, parts, None)
    assert [(f.valid_from, f.valid_to) for f in facts if f.p == "works_at"] == [("2014-06", None)]


# --- the pipeline -----------------------------------------------------------------------------


def test_ingest_briefs_once_for_normalisation(fresh_vault: Vault) -> None:
    llm = FakeLLM(structured_responses=[homepage_plan(), homepage_brief(), *homepage_extractions()])
    report = ingest(HOMEPAGE, fresh_vault, llm, today=TODAY)
    structured = [c[1] for c in llm.calls if c[0] == "structured"]
    assert [c["schema"].__name__ for c in structured] == [
        "SplitPlan",
        "Brief",
        "Extraction",
        "Extraction",
    ]
    assert report.brief is not None and report.brief.brief is not None
    # by default the brief is for normalisation only: the prompts and decisions are unchanged
    assert frozenset({"normalise"}) == BRIEF_USES
    assert not any("DOCUMENT BRIEF" in c["messages"][1]["content"] for c in structured[2:])
    assert not any(c[1]["text"].startswith("About:") for c in llm.calls if c[0] == "decide")
    notes = report.normalised.notes
    assert "brief: home Lisbon: start 2019 not stated with it; dropped" in notes
    atlas = fresh_vault.page("projects/atlas-migration")
    assert atlas is not None and any(
        f.p == "part_of" and f.o == "[[orgs/northwind-labs]]" for f in atlas.facts
    )
    lisbon = [f for p in fresh_vault.pages() if p.subject == "self" for f in p.history + p.facts]
    assert any(f.p == "lives_in" and "lisbon" in f.o and f.valid_to == "2024" for f in lisbon)


def test_ingest_can_share_the_brief_with_every_part(fresh_vault: Vault, monkeypatch) -> None:
    """The measured-and-off uses: the block first in every extraction prompt, the header as
    context of the checks (never of the gate)."""
    monkeypatch.setattr("brain.ingest.pipeline.BRIEF_USES", frozenset({"extract", "checks"}))
    llm = FakeLLM(structured_responses=[homepage_plan(), homepage_brief(), *homepage_extractions()])
    report = ingest(HOMEPAGE, fresh_vault, llm, today=TODAY)
    structured = [c[1] for c in llm.calls if c[0] == "structured"]
    block = render(report.brief.brief)
    assert all(
        f"DOCUMENT BRIEF (the whole document):\n{block}\n\nSection: " in c["messages"][1]["content"]
        for c in structured[2:]
    )
    decided = [c[1]["text"] for c in llm.calls if c[0] == "decide"]
    assert any(t.startswith("About: Alex Rivera (homepage") for t in decided)
    gates = [
        c[1]["text"]
        for c in llm.calls
        if c[0] == "decide" and c[1]["questions"][0].name == "about_a_person"
    ]
    assert gates and not any(t.startswith("About:") for t in gates)
    assert not any(n.startswith("brief: Atlas") for n in report.normalised.notes)  # not used


def test_ingest_goes_on_without_a_brief(fresh_vault: Vault) -> None:
    llm = FakeLLM(
        structured_responses=[
            homepage_plan(),
            StructuredOutputError("bad json", "{"),
            *homepage_extractions(),
        ]
    )
    report = ingest(HOMEPAGE, fresh_vault, llm, today=TODAY)
    assert report.brief is not None and report.brief.brief is None
    assert "brief: none (bad json)" in report.normalised.notes
    assert len(report.chunks) == 2 and fresh_vault.page("orgs/northwind-labs") is not None
    structured = [c[1] for c in llm.calls if c[0] == "structured"]
    assert "DOCUMENT BRIEF" not in structured[2]["messages"][1]["content"]


def test_brief_prompt_is_versioned() -> None:
    raw = (Path(__file__).resolve().parents[2] / "src/brain/prompts/ingest_brief.md").read_text()
    assert raw.startswith("<!-- ingest_brief v")
