"""The lint checks on the eval's fictional vaults with a scripted model: thresholds, guards,
questions, and what each check proposes. Fictional persona only (Alex Rivera)."""

from __future__ import annotations

from datetime import datetime

import pytest
from linttools import Script  # type: ignore[import-not-found]

from brain.eval import lint_eval as ev
from brain.lint import LintOptions, LintReport, run_lint
from brain.lint.consistency import TEMPLATE_AGENTS, agents_problems
from brain.lint.graph import Graph, units
from brain.lint.names import compare, different_people
from brain.llm import FakeLLM
from brain.vault.models import Fact
from brain.vault.ops import (
    AddEpisode,
    AddFact,
    ChangeSet,
    CreateEntity,
    DemoteEntity,
    MergeEntities,
    Operation,
    RefileFact,
)
from brain.vault.store import Vault
from brain.vault.writer import apply


def lint(vault: Vault, script: Script | None = None, *only: str) -> LintReport:
    llm = FakeLLM(decider=script) if script is not None else FakeLLM()
    options = LintOptions(only=only, full=True, today=ev.EVAL_TODAY, embeddings=False, rerank=False)
    return run_lint(vault, llm, options)


def ops(report: LintReport) -> list[Operation]:
    return [op for f in report.findings for op in f.ops]


def keys(report: LintReport, action: str | None = None) -> list[str]:
    return [f.key for f in report.findings if action is None or f.action == action]


def add(vault: Vault, *op: Operation) -> None:
    cs = ChangeSet(ops=list(op), origin="user", src=["ep-20261018-2100-goals"])
    result = apply(cs, vault, review=False, now=datetime(2026, 10, 26, 9, 0))
    assert not result.skipped, result.skipped


def choose(value_index: int, p: float):
    """Answer a choice question with its ``value_index``-th option at ``p``."""

    def answer(_text: str, q) -> dict[str, float]:
        values = [c.value for c in q.choices]
        rest = (1 - p) / (len(values) - 1)
        index = value_index % len(values)
        return {v: (p if n == index else rest) for n, v in enumerate(values)}

    return answer


def pick(value: str, p: float):
    def answer(_text: str, q) -> dict[str, float]:
        values = [c.value for c in q.choices]
        rest = (1 - p) / (len(values) - 1)
        return {v: (p if v == value else rest) for v in values}

    return answer


# --- names -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "kind", "how"),
    [
        ("University of Westbridge", "Westbridge University", "org", "core"),
        ("École Varenne", "Varenne", "org", "core"),
        ("KDC", "Kestrel Data Cooperative", "org", "acronym"),
        ("K.D.C.", "Kestrel Data Cooperative", "org", "acronym"),
        ("Nortwind Labs", "Northwind Labs", "org", "typo"),
        ("Northwnd Labs", "Northwind Labs", "org", "typo"),
        ("Théo Marchetti", "Theo Marchetti", "person", "core"),
        ("Marchetti, Theo", "Theo Marchetti", "person", "core"),
        ("Project Atlas", "Atlas Migration", "project", "contains"),
        ("Sam", "Sam Brennan", "person", "contains"),
        ("Sam Okafor", "Sam Brennan", "person", "first-name"),
        ("Atlas Migration", "Atlas Reporting", "project", "shared-word"),
        ("Northwind Labs", "Fabrikam Analytics", "org", None),
        ("Lisbon", "Lisboa", "place", "typo"),
        ("AI", "Atlas Ingest", "project", "acronym"),
    ],
)
def test_name_signals(a: str, b: str, kind: str, how: str | None) -> None:
    match = compare(a, b, kind)
    assert (match.how if match else None) == how


@pytest.mark.parametrize(
    ("a", "b", "different"),
    [
        (["Sam Okafor", "Samuel Okafor"], ["Sam Brennan"], True),
        (["Theo Marchetti"], ["Marchetti, Theo"], False),
        (["Maya Lindqvist"], ["Maya Lindkvist"], False),
        (["Sam"], ["Sam Brennan"], False),  # no surname on one side: ask
    ],
)
def test_different_people(a: list[str], b: list[str], different: bool) -> None:
    assert different_people(a, b) is different


def test_units_keep_the_line_that_introduces_a_list_item() -> None:
    text = (ev.OVERLAY / "raw" / "docs" / "cv.md").read_text()
    duty = next(u for u in units(text) if "demand forecasting" in u)
    assert duty.startswith("Fabrikam Analytics, Lisbon") and "›" in duty


def test_a_literal_employer_naming_a_page_is_a_job_with_that_page(
    dirty: tuple[Vault, ev.Gold],
) -> None:
    vault, _ = dirty
    jobs = Graph(vault, ev.EVAL_TODAY).jobs
    assert [(j.org, j.current) for j in jobs] == [
        ("orgs/northwind-labs", True),
        ("orgs/fabrikam-analytics", False),
    ]


# --- duplicates --------------------------------------------------------------------------------


def test_duplicates_merge_at_090_and_settle_several_look_alikes(
    dirty: tuple[Vault, ev.Gold],
) -> None:
    vault, gold = dirty
    titles = [{vault.page(i).title for i in pair} for pair in gold.merges.values()]

    def same(text: str, _q) -> dict[str, float]:
        named = {
            line.removeprefix("Name: ") for line in text.splitlines() if line.startswith("Name: ")
        }
        hit = any(t == named for t in titles) or "Name: Sam\n" in text
        return {"same": 0.95, "different": 0.05} if hit else {"same": 0.05, "different": 0.95}

    script = Script(same=same, which=pick("none of them", 0.97))
    report = lint(vault, script, "duplicates")
    merges = {frozenset({op.keep, op.merge}) for op in ops(report) if isinstance(op, MergeEntities)}
    assert merges == set(gold.merges.values())
    # "Sam" looks like both Sams: one exclusive question said "none", so nothing about it
    assert "which" in script.names()
    assert not any("people/sam " in k or k.endswith("people/sam") for k in keys(report))
    # two people with different surnames are never even compared
    assert not any("Sam Okafor" in t and "Sam Brennan" in t for n, t in script.asked if n == "same")
    # the kept page is the older one
    kept = {op.keep for op in ops(report) if isinstance(op, MergeEntities)}
    assert "orgs/northwind-labs" in kept and "orgs/westbridge-university" in kept


def test_duplicates_below_090_become_questions(dirty: tuple[Vault, ev.Gold]) -> None:
    vault, _ = dirty
    script = Script(same={"same": 0.7, "different": 0.3}, which=choose(1, 0.6))
    report = lint(vault, script, "duplicates")
    assert ops(report) == []
    asks = [f.ask.text for f in report.findings if f.ask]
    assert "Are Nortwind Labs and Northwind Labs the same organisation?" in asks
    sam = [f for f in report.findings if f.key == "merge people/sam people/sam-okafor"]
    assert sam and sam[0].ask and sam[0].ask.private  # Sam Okafor's page is private


def test_which_pick_merges_only_that_pair(clean_vault: Vault) -> None:
    add(
        clean_vault,
        CreateEntity(id="projects/the-atlas", type="project", title="The Atlas"),
        AddFact(
            subject="projects/the-atlas", p="status", o="testing", text="Testing", kind="stated"
        ),
    )
    script = Script(
        same=lambda t, q: (
            {"same": 0.9, "different": 0.1} if "The Atlas" in t else {"same": 0.1, "different": 0.9}
        ),
        which=pick("Atlas Migration", 0.95),
    )
    report = lint(clean_vault, script, "duplicates")
    assert [(op.keep, op.merge) for op in ops(report)] == [
        ("projects/atlas-migration", "projects/the-atlas")
    ]


def test_pages_that_disagree_on_where_they_are_are_not_merged(clean_vault: Vault) -> None:
    script = Script(same={"same": 0.97, "different": 0.03})
    report = lint(clean_vault, script, "duplicates")
    [finding] = [f for f in report.findings if "northwind-traders" in f.key]
    assert finding.ops == [] and "disagree on located_in" in finding.summary and finding.ask


def test_linked_pages_are_never_duplicates(clean_vault: Vault) -> None:
    add(
        clean_vault,
        CreateEntity(id="projects/atlas-cutover", type="project", title="Atlas Cutover"),
        AddFact(
            subject="projects/atlas-cutover",
            p="part_of",
            o="[[projects/atlas-migration]]",
            text="Part of Atlas Migration",
            kind="stated",
        ),
    )
    script = Script(same={"same": 0.97, "different": 0.03})
    lint(clean_vault, script, "duplicates")
    pairs = [t for n, t in script.asked if n == "same"]
    assert pairs and not any(
        "Name: Atlas Cutover" in t and "Name: Atlas Migration" in t for t in pairs
    )


def test_fallback_answers_never_act(dirty: tuple[Vault, ev.Gold]) -> None:
    vault, _ = dirty
    script = Script(same={"same": 1.0, "different": 0.0}, which=choose(1, 1.0))
    script.fallback = True
    report = lint(vault, script, "duplicates")
    assert ops(report) == [] and any(f.ask for f in report.findings)


# --- types -------------------------------------------------------------------------------------


def test_a_topic_page_is_demoted_on_the_value_kinds_mass(dirty: tuple[Vault, ev.Gold]) -> None:
    vault, gold = dirty
    topic = vault.page(gold.demotes["topic-page"])
    types = {p.title: p.type for p in vault.pages()}

    def kind(text: str, _q) -> dict[str, float]:
        name = text.split("Name: ", 1)[1].split("\n", 1)[0]
        if name == topic.title:
            return {"topic": 0.6, "tool": 0.32, "project": 0.08}
        return {types[name]: 0.9, "topic": 0.1}

    script = Script(kind=kind)
    report = lint(vault, script, "types")
    assert ops(report) == [DemoteEntity(id=topic.id)]
    # only doubtful pages are asked about (no facts of their own, or used as a value)
    asked = {t.split("Name: ", 1)[1].split("\n", 1)[0] for _, t in script.asked}
    assert topic.title in asked and "Northwind Labs" not in asked and "Kiln" not in asked


def test_a_doubtful_page_with_facts_is_asked_about(clean_vault: Vault) -> None:
    add(
        clean_vault,
        CreateEntity(id="projects/kubernetes", type="project", title="Kubernetes"),
        AddFact(
            subject="projects/kubernetes",
            p="note",
            o="Runs the clusters",
            text="Runs the clusters",
            kind="stated",
        ),
        AddFact(
            subject="self",
            p="has_skill",
            o="[[projects/kubernetes]]",
            text="Knows Kubernetes",
            kind="stated",
        ),
    )
    report = lint(clean_vault, Script(kind={"tool": 0.97, "project": 0.03}), "types")
    [finding] = report.findings
    assert finding.ops == [] and finding.ask
    assert finding.ask.text == "Is Kubernetes really a project, or a tool?"


def test_demotion_refiles_facts_that_used_the_page_as_a_page(clean_vault: Vault) -> None:
    add(
        clean_vault,
        CreateEntity(id="projects/dbt", type="project", title="dbt"),
        AddFact(
            subject="self", p="worked_on", o="[[projects/dbt]]", text="Worked on dbt", kind="stated"
        ),
        AddFact(
            subject="self",
            p="has_skill",
            o="[[projects/dbt]]",
            text="Skilled in dbt",
            kind="stated",
        ),
    )
    report = lint(clean_vault, Script(kind={"tool": 0.95, "project": 0.05}), "types")
    [refile, demote] = ops(report)
    assert isinstance(refile, RefileFact) and refile.p == "has_skill"
    assert demote == DemoteEntity(id="projects/dbt")


# --- subjects ----------------------------------------------------------------------------------


def test_a_projects_goal_filed_on_the_owner_is_refiled(dirty: tuple[Vault, ev.Gold]) -> None:
    vault, gold = dirty
    script = Script(whose=pick("Tidepool", 0.92))
    report = lint(vault, script, "subjects")
    assert ops(report) == [RefileFact(fact_id="f-nb2aag", subject="projects/tidepool")]
    # personal goals name no project: never asked about
    assert not any("half marathon" in t for _, t in script.asked)


def test_a_project_that_has_a_goal_gets_a_question(dirty: tuple[Vault, ev.Gold]) -> None:
    vault, _ = dirty
    add(
        vault,
        AddFact(
            subject="projects/tidepool",
            p="goal",
            o="Catch anomalies early",
            text="Catch anomalies early",
            kind="stated",
        ),
    )
    report = lint(vault, Script(whose=pick("Tidepool", 0.92)), "subjects")
    [finding] = report.findings
    assert finding.ops == [] and finding.ask and "or Tidepool's?" in finding.ask.text


# --- edges -------------------------------------------------------------------------------------


def test_part_of_located_in_and_job(dirty: tuple[Vault, ev.Gold]) -> None:
    vault, gold = dirty
    script = Script(
        part_of=lambda t, q: (
            pick("Northwind Labs", 0.9)(t, q)
            if "Tidepool" in t
            else pick("a side project", 0.95)(t, q)
        ),
        located=pick("Lisbon", 0.97),
        job=pick("Fabrikam Analytics", 0.95),
    )
    report = lint(vault, script, "edges")
    hits = {ev.classify(op, gold, report) for op in ops(report)}
    assert hits == {"missing-part-of", "org-location", "duty-job"}
    for op in ops(report):
        if isinstance(op, AddFact):
            assert op.kind == "inferred" and op.evidence
    assert any("kiln is a side project" in n for n in report.notes)


def test_unclear_edges_become_questions(dirty: tuple[Vault, ev.Gold]) -> None:
    vault, _ = dirty
    script = Script(
        part_of=pick("unclear", 0.7), located=pick("Lisbon", 0.6), job=pick("unclear", 0.9)
    )
    report = lint(vault, script, "edges")
    assert ops(report) == []
    asks = {f.ask.text for f in report.findings if f.ask}
    assert "Is Tidepool part of your work at Northwind Labs, or a side project?" in asks
    assert "Is Fabrikam Analytics based in Lisbon?" in asks


def test_a_location_is_only_asked_when_a_source_names_one(clean_vault: Vault) -> None:
    script = Script(located=pick("Lisbon", 0.97))
    lint(clean_vault, script, "edges")
    assert "located" not in script.names()  # Westbridge University: no source names a place


# --- consistency (code only) -------------------------------------------------------------------


def test_consistency_without_a_model(dirty: tuple[Vault, ev.Gold]) -> None:
    vault, gold = dirty
    report = run_lint(vault, None, LintOptions(full=True, today=ev.EVAL_TODAY))
    assert {f.check for f in report.findings} == {"consistency"}
    refiles = {(op.fact_id, op.o) for op in ops(report) if isinstance(op, RefileFact)}
    assert refiles == {
        ("f-f6b7a2", "[[orgs/fabrikam-analytics]]"),
        ("f-t5w6f7", "[[orgs/fabrikam-analytics]]"),
    }
    [two] = [f for f in report.findings if f.key == "two lives_in self"]
    assert two.ask and two.ask.text == "Where do you live now: Porto or Braga?"


def test_end_before_start_and_orphans(clean_vault: Vault) -> None:
    vault = clean_vault
    page = vault.page("projects/kiln")
    assert page
    path = vault.page_path("projects/kiln")
    bad = Fact(
        id="f-zz2zz3",
        text="Kiln ran a beta",
        p="note",
        o="Beta",
        rec="2026-10-20",
        src=["ep-20261012-1000-kiln"],
        kind="stated",
        valid_from="2026",
        valid_to="2025",
    )
    from brain.vault.render import render_fact

    path.write_text(
        path.read_text().replace("## Facts\n\n", "## Facts\n\n" + render_fact(bad) + "\n")
    )
    add(vault, CreateEntity(id="orgs/lonely-org", type="org", title="Lonely Org"))
    report = lint(vault, None, "consistency")
    assert "order f-zz2zz3" in keys(report)
    [orphan] = [f for f in report.findings if f.key == "orphan orgs/lonely-org"]
    assert orphan.ask and orphan.ask.text == "What is Lonely Org to you?"


def test_a_page_an_episode_lists_is_not_an_orphan(clean_vault: Vault) -> None:
    vault = clean_vault
    add(vault, CreateEntity(id="orgs/lonely-org", type="org", title="Lonely Org"))
    add(
        vault,
        AddEpisode(
            id="ep-20261021-1800-lonely",
            session="conv-20261021-1800",
            turns=(1, 2),
            channel="chat",
            entities=["orgs/lonely-org"],
            importance=3,
            gist="Alex mentions Lonely Org.",
            started=datetime(2026, 10, 21, 18, 0),
        ),
    )
    report = lint(vault, None, "consistency")
    assert "orphan orgs/lonely-org" not in keys(report)


def test_agents_predicates_are_in_sync(tmp_path) -> None:
    assert agents_problems(TEMPLATE_AGENTS) == []
    broken = tmp_path / "AGENTS.md"
    text = TEMPLATE_AGENTS.read_text().replace("`hometown` ", "").replace("(v0.5)", "(v0.4)")
    broken.write_text(text.replace("`note`*", "`note`* `vibe`"))
    problems = " | ".join(agents_problems(broken))
    assert "v0.4" in problems and "misses predicates: hometown" in problems
    assert "unknown predicates: vibe" in problems
