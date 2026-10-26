"""Generated index.md and core.md: deterministic, privacy-aware, capped, byte-stable."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from brain.tokens import estimate_tokens
from brain.vault import ChangeSet, apply, gitops
from brain.vault.generated import (
    GENERATED_NOTE,
    build_core,
    build_index,
    first_sentence,
    refresh_generated,
)
from brain.vault.init import init_vault
from brain.vault.ops import AddFact
from brain.vault.store import Vault

TODAY = date(2026, 10, 25)
SECTIONS = [
    "Who they are",
    "Now",
    "Life story",
    "People who matter",
    "Interests and skills",
    "Open loops",
]
EP = "(src:: [[ep-20261025-1412-career]])"
OTHER_SRC = "(src:: [[ep-20261026-0915-atlas]])"


# --- helpers -----------------------------------------------------------------------------------


def headings(core: str) -> list[str]:
    return [line[3:] for line in core.splitlines() if line.startswith("## ")]


def section(core: str, heading: str) -> list[str]:
    body = core.split(f"\n## {heading}\n\n", 1)[1]
    return body.split("\n\n", 1)[0].splitlines()


def identity(core: str) -> str:
    return next(line for line in core.splitlines() if line.startswith("Name: "))


def fact(
    text: str,
    p: str,
    o: str,
    *fields: str,
    n: int,
    rec: str = "2026-10-25",
    src: str = EP,
    kind: str = "stated",
) -> str:
    """A fact line (``fields`` such as ``from:: 2019``) with a deterministic valid block id."""
    b32 = "abcdefghijklmnopqrstuvwxyz234567"
    block = "".join(b32[(n >> (5 * i)) % 32] for i in range(6))
    extra = "".join(f" ({f})" for f in fields)
    return f"- {text} (p:: {p}) (o:: {o}){extra} (rec:: {rec}) {src} (kind:: {kind}) ^f-{block}"


def write_page(
    vault: Vault,
    page_id: str,
    title: str,
    facts: list[str],
    *,
    type_: str = "self",
    regions: tuple[str, ...] = (),
    privacy: str = "normal",
    history: tuple[str, ...] = (),
) -> None:
    subject = "self" if type_ == "self" else page_id
    past = "\n## History\n\n" + "\n".join(history) + "\n" if history else ""
    path = vault.page_path(page_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nid: {page_id}\ntype: {type_}\nsubject: {subject}\naliases: []\n"
        f"regions: [{', '.join(regions)}]\nprivacy: {privacy}\ncreated: 2026-10-20\n"
        f"updated: 2026-10-25\n---\n# {title}\n\n## Facts\n\n" + "\n".join(facts) + "\n" + past,
        encoding="utf-8",
    )


@pytest.fixture
def rich_vault(tmp_vault: Vault) -> Vault:
    """The fixture vault plus traits, values, interests, a partner, a manager and questions."""
    write_page(
        tmp_vault,
        "self/traits",
        "Traits",
        [
            fact("Explains things clearly", "has_strength", "Explaining things clearly", n=1),
            fact("Overcommits to side projects", "has_flaw", "Overcommits", n=2),
            fact("Gets impatient in meetings", "has_flaw", "Impatient", n=3, kind="inferred"),
        ],
        regions=("self.traits",),
        privacy="private",
    )
    write_page(
        tmp_vault,
        "self/values",
        "Values and goals",
        [
            fact("Values honesty", "values", "Honesty", n=4),
            fact("Values craft", "values", "Craft", n=5),
            fact("Values time outdoors", "values", "Time outdoors", n=6),
            fact(
                "Wants to run a half marathon", "goal", "Run a half marathon", "from:: 2026-09", n=7
            ),
        ],
        regions=("self.values_goals",),
    )
    write_page(
        tmp_vault,
        "self/interests",
        "Interests",
        [
            fact("Goes bouldering", "interest", "Bouldering", n=8),
            fact("Climbs at the bouldering gym", "interest", "bouldering", n=9, rec="2026-10-20"),
            fact("Enjoys archery", "likes", "Archery", n=10),
            fact("Likes data modelling", "likes", "Data-modelling", n=11),
            fact("Enjoys data modelling", "likes", "data modelling", n=12, src=OTHER_SRC),
            fact("Dislikes small talk", "dislikes", "Small talk", n=13),
        ],
        regions=("self.interests",),
    )
    write_page(
        tmp_vault,
        "people/jordan-blake",
        "Jordan Blake",
        [fact("Partner of Alex", "partner_of", "[[self]]", "from:: 2021", n=14)],
        type_="person",
        regions=("relationships.family",),
        privacy="private",
    )
    write_page(
        tmp_vault,
        "people/morgan-lee",
        "Morgan Lee",
        [fact("Works with Alex on Atlas", "colleague_of", "[[self]]", n=15)],
        type_="person",
        regions=("relationships.work",),
    )
    profile = tmp_vault.page_path("self/profile")
    text = profile.read_text(encoding="utf-8")
    manager = fact("Reports to Morgan Lee", "manager", "[[people/morgan-lee]]", n=16)
    text = text.replace("\n\n## History", f"\n{manager}\n\n## History", 1)
    profile.write_text(text, encoding="utf-8")
    tmp_vault.questions_path.write_text(
        "# Questions\n\n"
        "- [ ] Where did Alex grow up? (region:: identity) (priority:: high) ^q-a2b3c4\n"
        "- [x] Which languages does Alex speak? (region:: identity) ^q-d5e6f7\n"
        "- [ ] What does a normal weekday look like? (region:: self.routine) ^q-g2h3i4\n"
        "- [ ] How does Alex unwind after a hard week? ^q-j5k6l7\n"
        "- [ ] When is Jordan's birthday? ^q-m2n3o4\n",
        encoding="utf-8",
    )
    return Vault(tmp_vault.root)


# --- index.md ----------------------------------------------------------------------------------


def test_index_groups_pages_by_region(tmp_vault: Vault) -> None:
    text = build_index(tmp_vault)
    assert text.startswith("# Index\n\n" + GENERATED_NOTE)
    sections = text.split("\n## ")[1:]
    found = [s.split("\n", 1)[0] for s in sections]
    # life-map order: identity, career.current_job, ..., projects, relationships.friends
    assert found == ["Identity & basics", "Current job", "Projects", "Friends"]
    assert (
        "- [[projects/atlas-migration]] — Atlas Migration moves Northwind Labs' nightly batch "
        "pipelines from a legacy cluster to a managed warehouse." in text
    )
    assert text.count("- [[") == len(tmp_vault.pages())


def test_index_can_leave_out_private_pages(tmp_vault: Vault) -> None:
    assert "people/sam-okafor" in build_index(tmp_vault)
    assert "people/sam-okafor" not in build_index(tmp_vault, include_private=False)


# --- core.md -----------------------------------------------------------------------------------


def test_core_card_layout(tmp_vault: Vault) -> None:
    core = build_core(tmp_vault, today=TODAY)
    assert core.startswith("# Core\n\n" + GENERATED_NOTE + "\n\nName: Alex Rivera · ")
    assert headings(core) == SECTIONS
    assert identity(core) == (
        "Name: Alex Rivera · Senior data engineer at Northwind Labs (Data platform team)"
        " · lives in Porto · speaks English, Spanish"
    )
    assert section(core, "Now") == [
        "- Leads Atlas Migration (goal: move nightly batch pipelines to a managed warehouse;"
        " status: testing)",
        "- Responsibilities: Nightly batch pipelines",
        "- On their mind: Alex is anxious about the Atlas go-live deadline (2026-10-25)",
    ]
    assert section(core, "People who matter") == ["- Sam Okafor (close friend)"]
    # open loops: the upcoming deadline (so the project line leaves it out) and the intention
    assert section(core, "Open loops") == [
        "- Atlas Migration due 2026-11-15",
        "- Ask how the Atlas go-live went (after 2026-11-15)",
    ]
    assert "Is called Alex Rivera" not in core and "Lives in Porto" not in core


def test_core_groups_multi_valued_facts_into_lists(tmp_vault: Vault) -> None:
    core = build_core(tmp_vault, today=TODAY)
    # one line, salience order: the expert level first, then alphabetical on ties
    assert section(core, "Interests and skills") == [
        "- Skills: Python (expert), Airflow, Spark, SQL"
    ]
    assert "speaks English, Spanish" in identity(core)


def test_core_life_story_periods(tmp_vault: Vault) -> None:
    core = build_core(tmp_vault, today=TODAY)
    assert section(core, "Life story") == [
        "- 2023–now · Data engineer → Senior data engineer, Northwind Labs (Lisbon → Porto)",
        "- 2019–2023 · Analytics engineer, Fabrikam Analytics (Lisbon)",
        "- 2014–2018 · BSc Computer Science, University of Bristol",
    ]


def test_core_life_story_merges_duplicates_and_groups_undated(tmp_vault: Vault) -> None:
    write_page(
        tmp_vault,
        "self/career",
        "Career",
        [
            fact(
                "Worked at Fabrikam",
                "works_at",
                "Fabrikam Analytics",
                "from:: 2019",
                "to:: 2022",
                n=1,
            ),
            fact("Volunteers at a code club", "member_of", "Code club", n=3),
            fact("Built a budgeting app", "worked_on", "Budgeting app", n=4),
        ],
        history=(fact("Worked at a startup", "works_at", "Contoso Startup", n=2),),
        regions=("career.history",),
    )
    story = section(build_core(Vault(tmp_vault.root), today=TODAY), "Life story")
    assert sum("Fabrikam Analytics" in line for line in story) == 1  # duplicate span merged
    assert "- 2019–2023 · Analytics engineer, Fabrikam Analytics (Lisbon)" in story
    assert story[-2:] == ["- Undated: Contoso Startup", "- Other projects: Budgeting app"]


def _education(vault: Vault, *degrees: str) -> list[str]:
    write_page(
        vault,
        "self/education",
        "Education",
        [
            fact(
                "Studied at Calder",
                "studied_at",
                "[[orgs/calder]]",
                "from:: 2017",
                "to:: 2019",
                n=1,
            ),
            fact("Studied at Ostervale", "studied_at", "[[orgs/ostervale]]", n=2),
            fact(
                "Studied at Westbridge",
                "studied_at",
                "[[orgs/westbridge]]",
                "from:: 2021",
                "to:: 2022",
                n=3,
            ),
            *degrees,
        ],
        regions=("education",),
    )
    for org, title in (
        ("calder", "Calder College"),
        ("ostervale", "Ostervale Institute of Technology"),
        ("westbridge", "Westbridge University"),
    ):
        write_page(vault, f"orgs/{org}", title, [], type_="org", regions=("education",))
    return section(build_core(Vault(vault.root), today=TODAY), "Life story")


@pytest.fixture
def blank_vault(tmp_path: Path, git_env: None) -> Vault:
    return Vault(init_vault(tmp_path / "blank").root)


def test_core_pairs_a_degree_with_its_linked_school(blank_vault: Vault) -> None:
    """``at::`` decides; overlapping dates do not (the prep school shares 2019 with the degree,
    the double degree's school overlaps the master's)."""
    story = _education(
        blank_vault,
        fact(
            "Holds the degree General engineering degree",
            "degree",
            "General engineering degree",
            "at:: [[orgs/ostervale]]",
            "from:: 2019",
            "to:: 2022",
            n=4,
        ),
        fact(
            "Holds the degree MSc Data Science",
            "degree",
            "MSc Data Science",
            "at:: [[orgs/westbridge]]",
            "from:: 2021",
            "to:: 2022",
            n=5,
        ),
    )
    assert "- 2021–2022 · MSc Data Science, Westbridge University" in story
    assert "- 2017–2019 · Studied at Calder College" in story
    # the linked school has no dates: the degree keeps its own line, with the school's name
    assert "- 2019–2022 · General engineering degree, Ostervale Institute of Technology" in story
    assert not any("Calder" in line and "degree" in line for line in story)


def test_core_unlinked_degree_needs_more_than_a_boundary_year(blank_vault: Vault) -> None:
    story = _education(
        blank_vault,
        fact(
            "Holds a degree",
            "degree",
            "General engineering degree",
            "from:: 2019",
            "to:: 2022",
            n=4,
        ),
        fact("Holds an MSc", "degree", "MSc Data Science", "from:: 2021", "to:: 2022", n=5),
    )
    assert "- 2017–2019 · Studied at Calder College" in story  # 2019 alone is no evidence
    assert "- 2019–2022 · General engineering degree" in story
    assert "- 2021–2022 · MSc Data Science, Westbridge University" in story  # same period


def test_core_who_they_are_dedupe_and_people(rich_vault: Vault) -> None:
    core = build_core(rich_vault, today=TODAY)
    assert headings(core) == SECTIONS
    assert section(core, "Who they are") == [
        "- Values: Craft, Honesty, Time outdoors",
        "- Strengths: Explaining things clearly",
        "- Flaws: Overcommits, Impatient",  # stated before inferred
        "- Not known yet: habits",
    ]
    interests = section(core, "Interests and skills")
    # near-identical items collapse into one; the merged one has two sources, so it ranks above
    # "Archery" although both are likes and A sorts first
    assert interests[0] == "- Interests and likes: Bouldering, data modelling, Archery"
    assert "- Dislikes: Small talk" in interests
    assert section(core, "People who matter") == [
        "- Jordan Blake (partner)",
        "- Sam Okafor (close friend)",
        "- Morgan Lee (manager)",
    ]
    now = section(core, "Now")
    assert "- Goals: Run a half marathon" in now
    assert "- New lately: Wants to run a half marathon (since 2026-09)" in now
    loops = section(core, "Open loops")
    assert loops[-2:] == [
        "- To ask: Where did Alex grow up?",
        "- To ask: What does a normal weekday look like?",
    ]


def test_core_gaps_line_lists_missing_aspects(tmp_vault: Vault) -> None:
    core = build_core(tmp_vault, today=TODAY)
    assert section(core, "Who they are") == ["- Not known yet: values, strengths, flaws, habits"]
    # the big-model variant does not hint at private aspects
    public = build_core(tmp_vault, today=TODAY, include_private=False)
    assert section(public, "Who they are") == ["- Not known yet: values, habits"]


def test_core_private_filtering(rich_vault: Vault) -> None:
    core = build_core(rich_vault, today=TODAY, include_private=False)
    for secret in ("Sam Okafor", "Jordan Blake", "Overcommits", "Explaining", "anxious"):
        assert secret not in core
    assert section(core, "People who matter") == ["- Morgan Lee (manager)"]
    assert "- Values: Craft, Honesty, Time outdoors" in core
    # a question naming a private person in plain text is left out too
    (rich_vault.root / "questions.md").write_text(
        "# Questions\n\n- [ ] When is Jordan's birthday?\n"
    )
    assert "Open loops" in core and "birthday" not in build_core(
        rich_vault, today=TODAY, include_private=False
    )
    assert "birthday" in build_core(rich_vault, today=TODAY)


def test_core_validity(tmp_vault: Vault) -> None:
    # before Alex moved to Porto and before the senior role
    early = build_core(tmp_vault, today=date(2023, 6, 1))
    assert "lives in Porto" not in identity(early)
    assert "Senior data engineer" not in identity(early)
    assert "Northwind Labs" in identity(early)
    assert "On their mind" not in early  # the journal line is in the future


def test_core_hard_cap_keeps_identity(rich_vault: Vault) -> None:
    small = build_core(rich_vault, today=TODAY, max_tokens=120)
    assert estimate_tokens(small) <= 120
    assert "Senior data engineer at Northwind Labs" in small
    mid = build_core(rich_vault, today=TODAY, max_tokens=300)
    assert estimate_tokens(mid) <= 300
    assert "## Now" in mid  # the current chapter gets spare budget first


def test_core_section_budgets_are_shared(tmp_vault: Vault) -> None:
    """A huge skill list shrinks to fit its share; the other sections keep their lines."""
    skills = [fact(f"Knows tool {i}", "has_skill", f"Tool{i:03d}", n=100 + i) for i in range(200)]
    write_page(tmp_vault, "self/skills", "Skills", skills, regions=("skills",))
    core = build_core(Vault(tmp_vault.root), today=TODAY)
    skills_line = section(core, "Interests and skills")[0]
    assert skills_line.startswith("- Skills: Python (expert), ") and "more)" in skills_line
    assert headings(core) == SECTIONS
    assert len(section(core, "Open loops")) == 2 and len(section(core, "Life story")) == 3


def test_core_stays_small_on_a_large_vault(tmp_vault: Vault) -> None:
    """~60 self facts (a CV-sized profile) still give a card under ~600 tokens."""
    facts, n = [], 200
    for i in range(20):
        facts.append(fact(f"Is skilled in tool {i}", "has_skill", f"Tool number {i}", n=n + i))
    for i in range(10):
        facts.append(fact(f"Is into hobby {i}", "interest", f"Hobby number {i}", n=n + 20 + i))
    for i, p in enumerate(["values"] * 5 + ["has_strength"] * 4 + ["has_flaw"] * 3):
        facts.append(fact(f"Trait {i}", p, f"Trait number {i}", n=n + 40 + i))
    for i in range(6):
        start, end = 2000 + 3 * i, 2002 + 3 * i
        span = (f"from:: {start}", f"to:: {end}")
        facts.append(fact(f"Worked at Org {i}", "works_at", f"Org {i}", *span, n=n + 60 + i))
        facts.append(fact(f"Was role {i}", "role", f"Role {i}", *span, n=n + 70 + i))
    for i in range(6):
        facts.append(fact(f"Side project {i}", "worked_on", f"Side project {i}", n=n + 80 + i))
    write_page(tmp_vault, "self/everything", "Everything", facts, regions=("skills",))
    vault = Vault(tmp_vault.root)
    self_facts = sum(len(p.facts) for p in vault.pages() if p.subject == "self")
    assert self_facts >= 60
    core = build_core(vault, today=TODAY)
    assert estimate_tokens(core) <= 600
    assert headings(core) == SECTIONS
    assert core.count("\n- ") <= 30  # grouped, not one line per fact


def test_core_is_byte_stable(rich_vault: Vault) -> None:
    first = build_core(rich_vault, today=TODAY)
    assert first == build_core(Vault(rich_vault.root), today=TODAY)
    assert "20:" not in first and "T1" not in first  # no timestamps


def test_refresh_writes_and_commits_only_on_change(tmp_vault: Vault) -> None:
    changed = refresh_generated(tmp_vault, today=TODAY)
    assert {p.name for p in changed} == {"core.md", "index.md", "denylist.txt"}
    head = gitops.head(tmp_vault.root)
    assert refresh_generated(tmp_vault, today=TODAY) == []
    assert gitops.head(tmp_vault.root) == head

    cs = ChangeSet(
        ops=[
            AddFact(
                subject="self", p="interest", o="bouldering", text="Goes bouldering", kind="stated"
            )
        ],
        origin="user",
        src=["raw/conversations/2026/10/conv-20261025-1400.md"],
    )
    apply(cs, tmp_vault, review=False)
    changed = refresh_generated(tmp_vault, today=TODAY)
    assert {p.name for p in changed} == {"core.md", "index.md"}  # new page self/interests
    assert "- Interests and likes: bouldering" in tmp_vault.core_path.read_text()
    assert gitops.head(tmp_vault.root) != head


def test_first_sentence() -> None:
    assert first_sentence("Lives in Porto. Speaks Spanish.") == "Lives in Porto."
    assert first_sentence("e.g. this one") == "e.g. this one"
    long = first_sentence("word " * 100, limit=30)
    assert len(long) <= 30 and long.endswith("…")


def test_denylist_lists_private_names_and_keeps_manual_lines(tmp_vault: Vault) -> None:
    from brain.vault.generated import DENYLIST_MARKER, build_denylist

    manual = "# my terms\nSecret Street\n"
    text = build_denylist(tmp_vault, manual + "\n" + DENYLIST_MARKER + "\nStale Name\n")
    head, generated = text.split(DENYLIST_MARKER)
    assert head.strip() == manual.strip()
    names = generated.split()
    assert "Sam" in names and "Stale" not in names  # Sam Okafor's page is private
    assert "Northwind" not in generated  # normal pages are not listed
    assert build_denylist(tmp_vault, text) == text  # stable
