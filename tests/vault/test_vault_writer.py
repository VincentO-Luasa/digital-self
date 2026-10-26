"""The single writer: every op type, policy, review queue, provenance, git and atomicity.

All data is the fictional persona (Alex Rivera, Northwind Labs, Sam Okafor, Atlas Migration).
"""

from __future__ import annotations

import os
import subprocess
from datetime import date, datetime
from pathlib import Path

import pytest

from brain.vault import gitops, writer
from brain.vault.models import Fact, Page
from brain.vault.ops import (
    AddAlias,
    AddFact,
    AddIntention,
    AddObservation,
    ChangeSet,
    CreateEntity,
    FactSpec,
    MergeEntities,
    Operation,
    Origin,
    SupersedeFact,
    UpdateSummary,
)
from brain.vault.parse import FACT_ID, parse_page, split_frontmatter
from brain.vault.render import render_page
from brain.vault.store import Vault
from brain.vault.writer import (
    ApplyResult,
    ReviewNotFound,
    apply,
    approve,
    needs_review,
    pending_reviews,
    reject,
)

NOW = datetime(2026, 11, 2, 9, 30)
TODAY = NOW.date()
EP = "ep-20261102-0915-checkin"
SRC = [EP]


# --- helpers -----------------------------------------------------------------------------------


def cs(*ops: Operation, origin: Origin = "extract", src: list[str] | None = None) -> ChangeSet:
    return ChangeSet(
        ops=list(ops),
        origin=origin,
        src=SRC if src is None else src,
        rationale="Weekly check-in",
        created=NOW,
    )


def fact(subject: str, p: str, o: str, text: str | None = None, **kw: object) -> AddFact:
    return AddFact(subject=subject, p=p, o=o, text=text or f"{p} {o}", kind="stated", **kw)


def run(vault: Vault, *ops: Operation, **kw: object) -> ApplyResult:
    origin = kw.pop("origin", "extract")
    src = kw.pop("src", None)
    return apply(cs(*ops, origin=origin, src=src), vault, now=NOW, **kw)


def fresh(vault: Vault) -> Vault:
    return Vault(vault.root)


def page(vault: Vault, page_id: str) -> Page:
    found = fresh(vault).page(page_id)
    assert found is not None, page_id
    return found


def current(vault: Vault, page_id: str, fact_id: str) -> Fact:
    return next(f for f in page(vault, page_id).facts if f.id == fact_id)


def git(vault: Vault, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(vault.root), *args], capture_output=True, text=True, check=True
    )
    return out.stdout


def commits(vault: Vault) -> int:
    return int(git(vault, "rev-list", "--count", "HEAD"))


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and ".git" not in p.relative_to(root).parts
    }


def assert_clean(vault: Vault) -> None:
    """Working tree committed, everything parses without warnings, pages render byte-stable."""
    assert git(vault, "status", "--porcelain") == ""
    v = fresh(vault)
    assert v.warnings() == []
    for p in v.pages():
        path = v.page_path(p.id)
        assert render_page(p) == path.read_text(encoding="utf-8"), p.id
    assert not list(vault.root.rglob(".*.tmp"))


# --- create_entity -----------------------------------------------------------------------------


def test_create_entity_writes_page(tmp_vault: Vault) -> None:
    op = CreateEntity(
        id="topics/dbt",
        type="topic",
        title="dbt",
        aliases=["DBT", "data build tool", "data  build tool"],
        regions=["skills"],
        summary="  dbt builds the warehouse models for [[projects/atlas-migration]].\n",
    )
    result = run(tmp_vault, op)
    assert result.applied == [op] and result.created == ["topics/dbt"]
    p = page(tmp_vault, "topics/dbt")
    assert (p.type, p.subject, p.title) == ("topic", "topics/dbt", "dbt")
    assert p.aliases == ["data build tool"]  # title and repeated aliases dropped
    assert p.regions == ["skills"] and p.privacy == "normal"
    assert p.created == p.updated == TODAY
    assert p.summary == "dbt builds the warehouse models for [[projects/atlas-migration]]."
    assert p.facts == []
    assert_clean(tmp_vault)


def test_create_entity_privacy_from_regions_or_explicit(tmp_vault: Vault) -> None:
    run(
        tmp_vault,
        CreateEntity(
            id="people/samuel-okafor",
            type="person",
            title="Samuel Okafor",
            regions=["relationships.friends"],
        ),
        CreateEntity(id="topics/airflow", type="topic", title="Airflow", regions=["skills"]),
        CreateEntity(id="topics/burnout", type="topic", title="Burnout", privacy="private"),
        CreateEntity(id="places/porto", type="place", title="Porto"),
    )
    assert page(tmp_vault, "people/samuel-okafor").privacy == "private"
    assert page(tmp_vault, "topics/airflow").privacy == "normal"
    assert page(tmp_vault, "topics/burnout").privacy == "private"
    assert page(tmp_vault, "places/porto").privacy == "normal"


@pytest.mark.parametrize(
    ("op", "reason"),
    [
        (CreateEntity(id="Topics/DBT", type="topic", title="dbt"), "bad entity id"),
        (CreateEntity(id="topics", type="topic", title="dbt"), "bad entity id"),
        (CreateEntity(id="topics/../../x", type="topic", title="x"), "bad entity id"),
        (CreateEntity(id="topics/dbt", type="person", title="dbt"), "person pages live under"),
        (CreateEntity(id="people/sam-okafor", type="person", title="Sam"), "already exists"),
    ],
)
def test_create_entity_validation(tmp_vault: Vault, op: CreateEntity, reason: str) -> None:
    before = snapshot(tmp_vault.root)
    result = run(tmp_vault, op)
    assert result.applied == [] and len(result.skipped) == 1
    assert reason in result.skipped[0][1]
    assert result.commit is None and snapshot(tmp_vault.root) == before


def test_create_entity_twice_in_one_change_set(tmp_vault: Vault) -> None:
    op = CreateEntity(id="topics/dbt", type="topic", title="dbt")
    result = run(tmp_vault, op, op)
    assert result.applied == [op]
    assert [reason for _, reason in result.skipped] == ["topics/dbt already exists"]


# --- add_fact ----------------------------------------------------------------------------------


def test_add_fact_on_entity(tmp_vault: Vault) -> None:
    op = fact("projects/atlas-migration", "uses", "Kafka", "Atlas streams change events via Kafka")
    result = run(tmp_vault, op)
    assert result.applied == [op] and result.facts_added == 1
    (fact_id,) = result.new_ids
    p = page(tmp_vault, "projects/atlas-migration")
    new = p.facts[-1]
    assert new.id == fact_id and FACT_ID.match(fact_id)
    assert (new.text, new.p, new.o, new.kind) == (
        "Atlas streams change events via Kafka",
        "uses",
        "Kafka",
        "stated",
    )
    assert new.rec == TODAY and new.src == SRC
    assert p.updated == TODAY
    assert_clean(tmp_vault)


def test_add_fact_links_known_entities_and_accepts_wikilink_subject(tmp_vault: Vault) -> None:
    result = run(
        tmp_vault,
        fact("[[projects/atlas-migration]]", "stakeholder", "orgs/northwind-labs-ltd"),
        fact("[[projects/atlas-migration|Atlas]]", "related_to", "self"),
        fact("orgs/northwind-labs", "related_to", "projects/atlas-migration"),
    )
    assert len(result.applied) == 3
    atlas = page(tmp_vault, "projects/atlas-migration").facts
    assert atlas[-2].o == "orgs/northwind-labs-ltd"  # not a page: stays a literal
    assert atlas[-1].o == "[[self]]"
    assert page(tmp_vault, "orgs/northwind-labs").facts[-1].o == "[[projects/atlas-migration]]"


def test_add_fact_about_self_lands_on_profile(tmp_vault: Vault) -> None:
    result = run(tmp_vault, fact("self", "hometown", "Braga"), fact("self/profile", "born", "1994"))
    assert len(result.applied) == 2 and result.created == []
    facts = page(tmp_vault, "self/profile").facts
    assert [(f.p, f.o) for f in facts[-2:]] == [("hometown", "Braga"), ("born", "1994")]


def test_add_fact_about_self_creates_region_page_with_derived_privacy(tmp_vault: Vault) -> None:
    result = run(
        tmp_vault,
        fact("self", "habit", "Runs before work", "Runs most weekday mornings"),
        fact("self", "has_strength", "Calm under pressure", "Stays calm during incidents"),
        fact("self", "values", "Craftsmanship"),
        fact("self", "habit", "Reads before bed"),
    )
    assert len(result.applied) == 4
    assert sorted(result.created) == ["self/goals", "self/routine", "self/traits"]
    routine = page(tmp_vault, "self/routine")
    assert (routine.type, routine.subject, routine.privacy) == ("self", "self", "normal")
    assert routine.regions == ["self.routine"] and routine.title == "Daily life & habits"
    assert [f.o for f in routine.facts] == ["Runs before work", "Reads before bed"]
    traits = page(tmp_vault, "self/traits")
    assert traits.privacy == "private" and traits.regions == ["self.traits"]
    assert page(tmp_vault, "self/goals").facts[0].p == "values"
    assert "- created self/routine" in tmp_vault.log_path.read_text()
    assert_clean(tmp_vault)


@pytest.mark.parametrize(
    ("op", "src", "reason"),
    [
        (fact("people/nobody", "role", "Analyst"), SRC, "unknown entity people/nobody"),
        (fact("self", "favourite_colour", "Green"), SRC, "unknown predicate"),
        (fact("self", "hometown", "Braga"), [], "no source, no fact"),
        (fact("self", "hometown", "Braga (north"), SRC, "cannot be written as a fact line"),
        (fact("self", "lives_in", "Lisbon", **{"from": "2026", "to": "2025"}), SRC, "ends before"),
        (fact("self", "lives_in", "Lisbon", **{"from": "2025-13"}), SRC, "not a valid date"),
        (fact("self", "lives_in", "Lisbon", **{"to": "2025-02-30"}), SRC, "not a valid date"),
    ],
)
def test_add_fact_validation(tmp_vault: Vault, op: AddFact, src: list[str], reason: str) -> None:
    before = snapshot(tmp_vault.root)
    head = gitops.head(tmp_vault.root)
    result = run(tmp_vault, op, src=src)
    assert result.applied == [] and result.queued == []
    assert len(result.skipped) == 1 and reason in result.skipped[0][1]
    assert result.commit is None and gitops.head(tmp_vault.root) == head
    assert snapshot(tmp_vault.root) == before


def test_skipped_op_does_not_block_the_rest(tmp_vault: Vault) -> None:
    good = fact("self", "hometown", "Braga")
    result = run(tmp_vault, fact("people/nobody", "role", "Analyst"), good)
    assert result.applied == [good] and len(result.skipped) == 1
    log = tmp_vault.log_path.read_text()
    assert "+1 facts, 0 superseded, 0 queued, 1 skipped" in log
    assert "- skipped add_fact: unknown entity people/nobody" in log
    assert_clean(tmp_vault)


# --- provenance --------------------------------------------------------------------------------


def test_provenance_is_stamped_on_everything(tmp_vault: Vault) -> None:
    src = [EP, "raw/docs/cv.md"]
    result = run(
        tmp_vault,
        fact("self", "hometown", "Braga"),
        fact("projects/atlas-migration", "uses", "Kafka"),
        AddObservation(text="Alex sounded relieved about Atlas", importance=5),
        AddIntention(text="Ask about the cut-over", when=["projects/atlas-migration"]),
        src=src,
    )
    assert len(result.new_ids) == 4 and len(set(result.new_ids)) == 4
    v = fresh(tmp_vault)
    added = [v.find_fact(i) for i in result.new_ids[:2]]
    for found in added:
        assert found is not None
        assert found[1].src == src and found[1].rec == TODAY
    obs = next(o for o in v.observations() if o.id == result.new_ids[2])
    assert obs.src == src and obs.day == TODAY
    intention = next(i for i in v.intentions() if i.id == result.new_ids[3])
    assert intention.src == src


def test_new_ids_never_reuse_existing_ones(
    tmp_vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    proposals = iter(["f-k3x7q2", "f-s4t5u6", "f-aaaaaa", "f-aaaaaa", "f-bbbbbb"])
    monkeypatch.setattr(writer, "new_id", lambda prefix: next(proposals))
    result = run(tmp_vault, fact("self", "hometown", "Braga"), fact("self", "born", "1994"))
    assert result.new_ids == ["f-aaaaaa", "f-bbbbbb"]  # existing and already-used ids skipped
    assert_clean(tmp_vault)


# --- dedup and conflicts -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("op", "dup_of"),
    [
        (fact("self", "speaks", "English"), "f-g2h3i4"),
        (fact("self", "speaks", "english"), "f-g2h3i4"),  # case-insensitive
        (fact("self", "lives_in", "Porto", **{"from": "2024"}), "f-d5e6f7"),
        (fact("self", "lives_in", "Porto"), "f-d5e6f7"),  # undated restatement adds nothing
        (fact("self", "works_at", "[[orgs/northwind-labs]]", **{"from": "2023"}), "f-k3x7q2"),
        (fact("self", "works_at", "orgs/northwind-labs"), "f-k3x7q2"),  # link vs bare id
    ],
)
def test_duplicates_are_skipped(tmp_vault: Vault, op: AddFact, dup_of: str) -> None:
    result = run(tmp_vault, op)
    assert result.skipped == [(op, f"duplicate of ^{dup_of}")]
    assert result.commit is None


def test_duplicate_within_one_change_set(tmp_vault: Vault) -> None:
    op = fact("self", "hometown", "Braga")
    result = run(tmp_vault, op, op)
    assert result.applied == [op] and len(result.skipped) == 1
    assert result.skipped[0][1].startswith("duplicate of ^")


def test_conflict_on_single_valued_predicate_is_queued(tmp_vault: Vault) -> None:
    op = fact("self", "lives_in", "Lisbon", "Lives in Lisbon")
    expected = "contradicts current fact ^f-d5e6f7 (lives_in:: Porto)"
    assert needs_review(op, tmp_vault) == expected
    result = run(tmp_vault, op)
    assert result.queued == [(op, expected)] and result.applied == []
    assert current(tmp_vault, "self/profile", "f-d5e6f7").o == "Porto"


def test_new_information_about_same_value_is_a_conflict_not_a_duplicate(tmp_vault: Vault) -> None:
    ended = fact("self", "works_at", "[[orgs/northwind-labs]]", **{"from": "2023-02", "to": "2026"})
    earlier = fact("self", "lives_in", "Porto", **{"from": "2022"})
    result = run(tmp_vault, ended, earlier)
    assert [reason for _, reason in result.queued] == [
        "contradicts current fact ^f-k3x7q2 (works_at:: [[orgs/northwind-labs]])",
        "contradicts current fact ^f-d5e6f7 (lives_in:: Porto)",
    ]


def test_non_overlapping_values_do_not_conflict(tmp_vault: Vault) -> None:
    before = fact("self", "lives_in", "Lisbon", **{"from": "2019", "to": "2023"})
    handover = fact("self", "lives_in", "Braga", **{"from": "2023", "to": "2024"})
    result = run(tmp_vault, before, handover)
    assert result.applied == [before, handover] and result.queued == []


def test_multi_valued_predicates_only_conflict_on_same_object(tmp_vault: Vault) -> None:
    ops = [
        fact("self", "speaks", "French"),
        fact("self", "goal", "Run a half marathon"),
        fact("self", "goal", "Learn Rust"),
    ]
    assert run(tmp_vault, *ops).applied == ops
    project_goal = fact("projects/atlas-migration", "goal", "Retire the legacy cluster")
    result = run(tmp_vault, project_goal)  # goal is single-valued for projects
    assert result.queued[0][1].startswith("contradicts current fact ^f-k5l6m7")


def test_conflict_within_one_change_set(tmp_vault: Vault) -> None:
    first = fact("self", "hometown", "Braga")
    second = fact("self", "hometown", "Coimbra")
    result = run(tmp_vault, first, second)
    assert result.applied == [first]
    assert result.queued[0][0] == second and "contradicts current fact" in result.queued[0][1]


def test_applying_a_contradiction_supersedes_the_old_fact(tmp_vault: Vault) -> None:
    op = fact("self", "lives_in", "Lisbon", "Lives in Lisbon", **{"from": "2026-10"})
    result = run(tmp_vault, op, review=False)
    assert result.applied == [op] and result.superseded == 1
    (new_id,) = result.new_ids
    profile = page(tmp_vault, "self/profile")
    assert [f.o for f in profile.facts if f.p == "lives_in"] == ["Lisbon"]
    old = next(f for f in profile.history if f.id == "f-d5e6f7")
    assert (old.valid_from, old.valid_to, old.by) == ("2024", "2026-10", new_id)
    assert_clean(tmp_vault)


def test_applying_an_undated_contradiction_ends_old_fact_today(tmp_vault: Vault) -> None:
    result = run(tmp_vault, fact("self", "lives_in", "Lisbon"), review=False)
    old = next(f for f in page(tmp_vault, "self/profile").history if f.id == "f-d5e6f7")
    assert old.valid_to == TODAY.isoformat() and old.by == result.new_ids[0]


@pytest.mark.parametrize(
    ("op", "origin", "reason"),
    [
        (
            AddFact(
                subject="self",
                p="has_flaw",
                o="Underestimates timelines",
                text="Tends to underestimate migration timelines",
                kind="inferred",
                evidence=["ep-20261025-1412-career", "f-c5d6e7"],
            ),
            "extract",
            "inferred fact",
        ),
        (fact("people/sam-okafor", "knows", "[[orgs/northwind-labs]]"), "extract", "relationship"),
        (
            fact("people/sam-okafor", "lives_in", "Bristol"),
            "external",
            "external change to private",
        ),
        (fact("self", "health", "Sleeps badly"), "external", "private page self/health"),
        (fact("self", "hometown", "Braga"), "external", None),
        (fact("self", "hometown", "Braga"), "user", None),
        (SupersedeFact(fact_id="f-n2o3p4", to="2026-11"), "user", "supersession"),
        (MergeEntities(keep="people/sam-okafor", merge="orgs/northwind-labs"), "user", "merge"),
        (CreateEntity(id="topics/dbt", type="topic", title="dbt"), "external", None),
        (
            CreateEntity(
                id="people/samuel-okafor",
                type="person",
                title="Samuel Okafor",
                regions=["relationships.friends"],
            ),
            "external",
            "external change to private page people/samuel-okafor",
        ),
        (UpdateSummary(id="people/sam-okafor", summary="x"), "external", "private"),
        (AddAlias(id="people/sam-okafor", alias="Sammy"), "external", "private"),
        (AddAlias(id="people/sam-okafor", alias="Sammy"), "extract", None),
        (AddIntention(text="Ask Sam", when=["people/sam-okafor"]), "external", "private"),
        (AddObservation(text="Alex was tired", importance=3), "external", None),
    ],
)
def test_policy(tmp_vault: Vault, op: Operation, origin: Origin, reason: str | None) -> None:
    got = needs_review(op, tmp_vault, origin=origin)
    if reason is None:
        assert got is None
    else:
        assert got is not None and reason in got


# --- supersede_fact ----------------------------------------------------------------------------


def test_supersede_fact_is_queued_by_default(tmp_vault: Vault) -> None:
    op = SupersedeFact(fact_id="f-n2o3p4", to="2026-11")
    result = run(tmp_vault, op)
    assert result.queued == [(op, "supersession")] and result.review_id
    assert current(tmp_vault, "projects/atlas-migration", "f-n2o3p4").o == "testing"


def test_supersede_fact_moves_old_to_history(tmp_vault: Vault) -> None:
    new = FactSpec(p="status", o="cut-over", text="Atlas is in cut-over", kind="stated")
    new.valid_from = "2026-11"
    op = SupersedeFact(fact_id="f-n2o3p4", to="2026-10-31", new=new)
    result = run(tmp_vault, op, review=False)
    assert result.applied == [op] and result.superseded == 1 and result.facts_added == 1
    (new_id,) = result.new_ids
    assert new_id != "f-n2o3p4" and FACT_ID.match(new_id)
    atlas = page(tmp_vault, "projects/atlas-migration")
    assert "f-n2o3p4" not in [f.id for f in atlas.facts]
    old = next(f for f in atlas.history if f.id == "f-n2o3p4")
    assert (old.o, old.valid_from, old.valid_to, old.by) == (
        "testing",
        "2026-10",
        "2026-10-31",
        new_id,
    )
    assert old.src == ["ep-20261025-1412-career"] and old.rec == date(2026, 10, 25)  # untouched
    replacement = atlas.facts[-1]
    assert (replacement.id, replacement.o, replacement.valid_from) == (
        new_id,
        "cut-over",
        "2026-11",
    )
    assert replacement.src == SRC and replacement.rec == TODAY
    assert_clean(tmp_vault)


def test_supersede_end_defaults(tmp_vault: Vault) -> None:
    new = FactSpec.model_validate(
        {
            "p": "deadline",
            "o": "2026-12-01",
            "text": "Go-live moved",
            "kind": "stated",
            "from": "2026-10",
        }
    )
    run(tmp_vault, SupersedeFact(fact_id="f-q5r6s7", new=new), review=False)
    run(tmp_vault, SupersedeFact(fact_id="f-w5x6y7"), review=False)
    history = {f.id: f for f in page(tmp_vault, "projects/atlas-migration").history}
    assert history["f-q5r6s7"].valid_to == "2026-10"  # the new fact's from::
    assert history["f-w5x6y7"].valid_to == TODAY.isoformat()  # no hint: today


def test_supersede_without_replacement_ends_the_fact(tmp_vault: Vault) -> None:
    result = run(tmp_vault, SupersedeFact(fact_id="f-w5x6y7", to="2026-10"), review=False)
    assert result.superseded == 1 and result.facts_added == 0 and result.new_ids == []
    atlas = page(tmp_vault, "projects/atlas-migration")
    ended = next(f for f in atlas.history if f.id == "f-w5x6y7")
    assert ended.valid_to == "2026-10" and ended.by is None
    assert_clean(tmp_vault)


@pytest.mark.parametrize(
    ("op", "src", "reason"),
    [
        (SupersedeFact(fact_id="f-zzzzzz"), SRC, "unknown fact ^f-zzzzzz"),
        (SupersedeFact(fact_id="f-s4t5u6"), SRC, "already superseded"),
        (
            SupersedeFact(
                fact_id="f-n2o3p4", new=FactSpec(p="phase", o="x", text="x", kind="stated")
            ),
            SRC,
            "unknown predicate",
        ),
        (
            SupersedeFact(
                fact_id="f-n2o3p4", new=FactSpec(p="status", o="x", text="x", kind="stated")
            ),
            [],
            "no source, no fact",
        ),
        (SupersedeFact(fact_id="f-n2o3p4", to="2026-09"), SRC, "ends before"),  # from:: 2026-10
        (SupersedeFact(fact_id="f-n2o3p4", to="2026-02-31"), SRC, "not a valid date"),
    ],
)
def test_supersede_validation(
    tmp_vault: Vault, op: SupersedeFact, src: list[str], reason: str
) -> None:
    before = snapshot(tmp_vault.root)
    result = run(tmp_vault, op, src=src, review=False)
    assert len(result.skipped) == 1 and reason in result.skipped[0][1]
    assert snapshot(tmp_vault.root) == before


def test_supersede_twice_in_one_change_set(tmp_vault: Vault) -> None:
    op = SupersedeFact(fact_id="f-n2o3p4", to="2026-11")
    result = run(tmp_vault, op, op, review=False)
    assert result.applied == [op] and "already superseded" in result.skipped[0][1]


# --- merge_entities ----------------------------------------------------------------------------


@pytest.fixture
def with_duplicate(tmp_vault: Vault) -> Vault:
    """A second page for Sam, referenced from facts, intentions, the journal and core.md."""
    dup = "people/samuel-okafor"
    run(
        tmp_vault,
        CreateEntity(
            id=dup,
            type="person",
            title="Samuel Okafor",
            aliases=["Sammy", "Sam"],
            regions=["relationships.friends", "relationships.work"],
            summary="Samuel reviews the [[projects/atlas-migration]] runbooks.",
        ),
        fact(dup, "role", "Platform engineer"),
        fact("projects/atlas-migration", "stakeholder", f"[[{dup}]]"),
        AddIntention(text="Ask Samuel about the runbooks", when=[dup]),
        AddObservation(text=f"Alex met [[{dup}|Samuel]] for lunch", importance=4),
    )
    core = tmp_vault.core_path
    core.write_text(core.read_text() + f"\nClose friends: [[{dup}|Samuel]], [[{dup}-jr]].\n")
    gitops.commit_all(tmp_vault.root, "hand edit")
    return tmp_vault


def test_merge_is_queued_by_default(with_duplicate: Vault) -> None:
    op = MergeEntities(keep="people/sam-okafor", merge="people/samuel-okafor")
    result = run(with_duplicate, op)
    assert result.queued == [(op, "entity merge")]
    assert with_duplicate.page_path("people/samuel-okafor").exists()


def test_merge_entities(with_duplicate: Vault) -> None:
    vault = with_duplicate
    op = MergeEntities(keep="[[people/sam-okafor]]", merge="people/samuel-okafor")
    result = run(vault, op, review=False)
    assert result.applied == [op]
    assert not vault.page_path("people/samuel-okafor").exists()
    assert vault.page_path("people/samuel-okafor") in result.changed
    sam = page(vault, "people/sam-okafor")
    assert sam.aliases == ["Sam", "Samuel Okafor", "Sammy"]  # names of both, no repeats
    assert sam.regions == ["relationships.friends", "relationships.work"]
    assert sam.privacy == "private" and sam.updated == TODAY
    assert [f.p for f in sam.facts] == ["friend_of", "met_at", "last_seen", "role"]
    assert sam.summary.startswith("Sam is one of Alex's closest friends")
    # links repointed in other pages and files; a longer id sharing the prefix is left alone
    atlas = page(vault, "projects/atlas-migration")
    assert atlas.facts[-1].o == "[[people/sam-okafor]]"
    text = "\n".join(
        p.read_text()
        for p in (vault.intentions_path, vault.core_path, vault.journal_path("2026-11"))
    )
    assert "[[people/samuel-okafor]]" not in text and "[[people/samuel-okafor|" not in text
    assert "[[people/sam-okafor|Samuel]]" in text and "[[people/samuel-okafor-jr]]" in text
    intention = fresh(vault).intentions()[-1]
    assert intention.when == ["people/sam-okafor"]
    assert_clean(vault)


@pytest.mark.parametrize(
    ("keep", "merge", "reason"),
    [
        ("people/sam-okafor", "[[people/sam-okafor]]", "into itself"),
        ("people/sam-okafor", "people/nobody", "unknown entity people/nobody"),
        ("self", "people/sam-okafor", "self pages cannot be merged"),
        ("people/sam-okafor", "self/profile", "self pages cannot be merged"),
    ],
)
def test_merge_validation(tmp_vault: Vault, keep: str, merge: str, reason: str) -> None:
    result = run(tmp_vault, MergeEntities(keep=keep, merge=merge), review=False)
    assert len(result.skipped) == 1 and reason in result.skipped[0][1]


# --- update_summary / add_alias ------------------------------------------------------------------


def test_update_summary(tmp_vault: Vault) -> None:
    summary = "Atlas moves nightly batch pipelines to a managed warehouse.\n\nGo-live is 15 Nov."
    result = run(tmp_vault, UpdateSummary(id="[[projects/atlas-migration]]", summary=summary))
    assert len(result.applied) == 1
    atlas = page(tmp_vault, "projects/atlas-migration")
    assert atlas.summary == summary and len(atlas.facts) == 8
    assert_clean(tmp_vault)


@pytest.mark.parametrize("summary", ["Intro\n## Facts\n- injected", "## Notes"])
def test_summary_may_not_open_a_section(tmp_vault: Vault, summary: str) -> None:
    result = run(tmp_vault, UpdateSummary(id="projects/atlas-migration", summary=summary))
    assert len(result.skipped) == 1 and "## section headings" in result.skipped[0][1]
    created = run(
        tmp_vault, CreateEntity(id="topics/dbt", type="topic", title="dbt", summary=summary)
    )
    assert len(created.skipped) == 1 and "## section headings" in created.skipped[0][1]


def test_summary_with_other_headings_round_trips(tmp_vault: Vault) -> None:
    summary = "# Overview\n\n### Phases\n\n#atlas dbt first, then Airflow"
    assert run(tmp_vault, UpdateSummary(id="projects/atlas-migration", summary=summary)).applied
    assert page(tmp_vault, "projects/atlas-migration").summary == summary
    assert_clean(tmp_vault)


def test_add_alias(tmp_vault: Vault) -> None:
    result = run(tmp_vault, AddAlias(id="self", alias="  Lex  "))
    assert len(result.applied) == 1
    assert page(tmp_vault, "self/profile").aliases == ["Alex Rivera", "Alex", "Lex"]
    for alias, reason in [
        ("atlas", "already a name"),
        ("Atlas Migration", "already a name"),
        ("   ", "empty alias"),
    ]:
        skipped = run(tmp_vault, AddAlias(id="projects/atlas-migration", alias=alias)).skipped
        assert len(skipped) == 1 and reason in skipped[0][1]
    assert "unknown entity" in run(tmp_vault, AddAlias(id="people/x", alias="X")).skipped[0][1]


# --- intentions / observations -------------------------------------------------------------------


def test_add_intention(tmp_vault: Vault) -> None:
    op = AddIntention(
        text="Ask how the  dbt migration went",
        when="[[projects/atlas-migration]]",
        after="2026-12",
    )
    result = run(tmp_vault, op)
    (intention_id,) = result.new_ids
    intentions = fresh(tmp_vault).intentions()
    assert [i.id for i in intentions] == ["i-p2m7wd", "i-q3n4xe", intention_id]
    new = intentions[-1]
    assert (new.text, new.when, new.after, new.src, new.done) == (
        "Ask how the dbt migration went",
        ["projects/atlas-migration"],
        "2026-12",
        SRC,
        False,
    )
    assert "empty intention" in run(tmp_vault, AddIntention(text="  ")).skipped[0][1]
    assert_clean(tmp_vault)


def test_add_intention_creates_file(tmp_vault: Vault) -> None:
    tmp_vault.intentions_path.unlink()
    gitops.commit_all(tmp_vault.root, "drop intentions")
    run(tmp_vault, AddIntention(text="Ask about Atlas"))
    assert tmp_vault.intentions_path.read_text().startswith("# Intentions\n\n- [ ] Ask about Atlas")
    assert_clean(tmp_vault)


def test_add_observation_new_and_existing_month(tmp_vault: Vault) -> None:
    first = run(tmp_vault, AddObservation(text="Alex  sounded relieved", importance=6))
    second = run(tmp_vault, AddObservation(text="Alex booked a holiday", importance=3))
    path = tmp_vault.journal_path("2026-11")
    assert path.read_text() == (
        "# Journal 2026-11\n\n"
        f"- 2026-11-02 · imp 6 · Alex sounded relieved · [[{EP}]] ^{first.new_ids[0]}\n"
        f"- 2026-11-02 · imp 3 · Alex booked a holiday · [[{EP}]] ^{second.new_ids[0]}\n"
    )
    # the day comes from the change set (when it was observed), not from `now`
    october = ChangeSet(
        ops=[AddObservation(text="Alex was tired", importance=2)],
        origin="extract",
        src=SRC,
        created=datetime(2026, 10, 30, 18, 0),
    )
    result = apply(october, tmp_vault, now=NOW)
    obs = [o for o in fresh(tmp_vault).observations() if o.id == result.new_ids[0]]
    assert obs[0].day == date(2026, 10, 30)
    assert tmp_vault.journal_path("2026-10").read_text().count("\n- ") == 3
    assert (
        "no source" in run(tmp_vault, AddObservation(text="x", importance=1), src=[]).skipped[0][1]
    )
    assert_clean(tmp_vault)


# --- review queue --------------------------------------------------------------------------------


def _inferred() -> AddFact:
    return AddFact(
        subject="self",
        p="has_flaw",
        o="Underestimates timelines",
        text="Tends to underestimate migration timelines",
        kind="inferred",
        evidence=["ep-20261025-1412-career", "we slipped again"],
    )


def test_queued_ops_land_in_a_pending_review(tmp_vault: Vault) -> None:
    safe = fact("self", "hometown", "Braga")
    result = run(tmp_vault, safe, _inferred(), SupersedeFact(fact_id="f-n2o3p4", to="2026-11"))
    assert result.applied == [safe] and len(result.queued) == 2
    rid = result.review_id
    assert rid and rid.startswith("r-")
    path = tmp_vault.reviews_pending_dir / f"{rid}.md"
    assert path in result.changed and path.exists()
    text = path.read_text()
    assert "Why: inferred fact" in text and "Why: supersession" in text
    (review,) = pending_reviews(fresh(tmp_vault))
    assert (review.id, review.origin, review.src, review.status) == (rid, "extract", SRC, "pending")
    assert review.rationale == "Weekly check-in" and review.created == NOW
    assert review.ops == [op for op, _ in result.queued] and review.path == path
    assert f"- queued for review: {rid}" in tmp_vault.log_path.read_text()
    assert_clean(tmp_vault)


def test_approve_applies_and_moves_review_to_done(tmp_vault: Vault) -> None:
    queued = run(tmp_vault, _inferred())
    rid = queued.review_id
    assert rid
    later = datetime(2026, 12, 1, 8, 0)
    head = gitops.head(tmp_vault.root)
    result = approve(rid, tmp_vault, now=later)
    assert len(result.applied) == 1 and result.queued == [] and result.review_id is None
    assert result.commit and result.commit != head
    traits = page(tmp_vault, "self/traits")
    flaw = traits.facts[0]
    assert flaw.kind == "inferred" and flaw.src == [*SRC, rid]  # the review is provenance too
    assert flaw.rec == later.date() and flaw.evidence == [
        "ep-20261025-1412-career",
        "we slipped again",
    ]
    assert pending_reviews(fresh(tmp_vault)) == []
    done = tmp_vault.reviews_done_dir / "2026" / "12" / f"{rid}.md"
    assert done.exists() and not (tmp_vault.reviews_pending_dir / f"{rid}.md").exists()
    loaded = writer.load_review(done)
    assert loaded.status == "approved" and loaded.ops == [queued.queued[0][0]]
    assert "resolved: 2026-12-01T08:00" in done.read_text()
    with pytest.raises(ReviewNotFound):
        approve(rid, tmp_vault, now=later)
    assert_clean(tmp_vault)


def test_reject_records_reason(tmp_vault: Vault) -> None:
    rid = run(tmp_vault, _inferred()).review_id
    assert rid
    before = page(tmp_vault, "self/profile")
    done = reject(rid, "Not true: Alex plans buffers", tmp_vault, now=NOW)
    assert done == tmp_vault.reviews_done_dir / "2026" / "11" / f"{rid}.md"
    loaded = writer.load_review(done)
    assert loaded.status == "rejected"
    fm, _, _ = split_frontmatter(done.read_text())
    assert fm and fm["reason"] == "Not true: Alex plans buffers"
    assert str(fm["resolved"]).startswith("2026-11-02")
    assert pending_reviews(fresh(tmp_vault)) == []
    assert page(tmp_vault, "self/profile") == before
    assert fresh(tmp_vault).page("self/traits") is None
    assert f"user | {rid} | rejected: Not true" in tmp_vault.log_path.read_text()
    assert_clean(tmp_vault)


@pytest.mark.parametrize("rid", ["r-zzzzzz", "../../log", "../pending/r-x", "", "."])
def test_unknown_review(tmp_vault: Vault, rid: str) -> None:
    before = snapshot(tmp_vault.root)
    with pytest.raises(ReviewNotFound):
        approve(rid, tmp_vault, now=NOW)
    with pytest.raises(ReviewNotFound):
        reject(rid, "nope", tmp_vault, now=NOW)
    assert snapshot(tmp_vault.root) == before


def test_review_true_queues_every_valid_op(tmp_vault: Vault) -> None:
    ops = [
        CreateEntity(id="topics/dbt", type="topic", title="dbt"),
        fact("topics/dbt", "related_to", "projects/atlas-migration"),
        fact("self", "hometown", "Braga"),
        fact("people/nobody", "role", "x"),
    ]
    result = run(tmp_vault, *ops, review=True)
    assert result.applied == [] and len(result.skipped) == 1
    assert [r for _, r in result.queued] == [
        "review requested",
        "depends on an operation queued for review",
        "review requested",
    ]
    assert fresh(tmp_vault).page("topics/dbt") is None
    approved = approve(result.review_id or "", tmp_vault, now=NOW)
    assert approved.applied == ops[:3] and approved.created == ["topics/dbt"]
    assert page(tmp_vault, "topics/dbt").facts[0].o == "[[projects/atlas-migration]]"


def test_review_false_applies_what_policy_would_queue(tmp_vault: Vault) -> None:
    ops = [_inferred(), fact("people/sam-okafor", "knows", "[[orgs/northwind-labs]]")]
    result = run(tmp_vault, *ops, review=False)
    assert result.applied == ops and result.queued == [] and result.review_id is None
    assert not tmp_vault.reviews_pending_dir.exists()


def test_create_then_add_fact_in_one_change_set(tmp_vault: Vault) -> None:
    ops = [
        CreateEntity(id="topics/dbt", type="topic", title="dbt", regions=["skills"]),
        fact("topics/dbt", "related_to", "[[projects/atlas-migration]]"),
        fact("self", "has_skill", "[[topics/dbt]]", "Is good at dbt"),
        AddAlias(id="topics/dbt", alias="data build tool"),
        UpdateSummary(id="topics/dbt", summary="Warehouse modelling tool."),
    ]
    result = run(tmp_vault, *ops)
    assert result.applied == ops and result.queued == []
    dbt = page(tmp_vault, "topics/dbt")
    assert dbt.aliases == ["data build tool"] and dbt.summary == "Warehouse modelling tool."
    assert page(tmp_vault, "self/profile").facts[-1].o == "[[topics/dbt]]"
    assert result.commit and commits(tmp_vault) == 2
    assert_clean(tmp_vault)


def test_ops_depending_on_a_queued_entity_are_queued_and_approved_in_order(
    tmp_vault: Vault,
) -> None:
    dup = "people/samuel-okafor"
    ops = [
        CreateEntity(
            id=dup, type="person", title="Samuel Okafor", regions=["relationships.friends"]
        ),
        fact(dup, "role", "Platform engineer"),
        AddAlias(id=dup, alias="Sammy"),
        fact("self", "hometown", "Braga"),
    ]
    result = run(tmp_vault, *ops, origin="external")
    assert result.applied == [ops[3]]
    assert [r for _, r in result.queued] == [
        f"external change to private page {dup}",
        "depends on an operation queued for review",
        "depends on an operation queued for review",
    ]
    approved = approve(result.review_id or "", tmp_vault, now=NOW)
    assert approved.applied == ops[:3]
    samuel = page(tmp_vault, dup)
    assert samuel.privacy == "private" and samuel.aliases == ["Sammy"]
    assert samuel.facts[0].src == [*SRC, result.review_id]


# --- log, git, round trip ------------------------------------------------------------------------


def test_log_entry_and_one_commit_per_change_set(tmp_vault: Vault) -> None:
    head = gitops.head(tmp_vault.root)
    log_before = tmp_vault.log_path.read_text()
    result = run(
        tmp_vault,
        CreateEntity(id="topics/dbt", type="topic", title="dbt"),
        fact("self", "hometown", "Braga"),
    )
    assert result.commit and result.commit == gitops.head(tmp_vault.root) != head
    assert commits(tmp_vault) == 2
    assert git(tmp_vault, "log", "-1", "--format=%s").strip() == f"brain: extract {EP}"
    added = tmp_vault.log_path.read_text().removeprefix(log_before)
    assert added == (
        f"\n## [2026-11-02 09:30] extract | {EP} | +1 facts, 0 superseded, 0 queued\n"
        "- created topics/dbt\n"
    )
    assert set(git(tmp_vault, "show", "--name-only", "--format=", "HEAD").split()) == {
        "log.md",
        "wiki/self/profile.md",
        "wiki/topics/dbt.md",
    }
    assert_clean(tmp_vault)


def test_all_skipped_or_empty_change_set_makes_no_commit(tmp_vault: Vault) -> None:
    before = snapshot(tmp_vault.root)
    head = gitops.head(tmp_vault.root)
    skipped = run(tmp_vault, fact("people/nobody", "role", "x"), fact("self", "speaks", "English"))
    assert len(skipped.skipped) == 2 and skipped.commit is None and skipped.changed == []
    empty = run(tmp_vault)
    assert empty.commit is None and empty.changed == []
    assert gitops.head(tmp_vault.root) == head and snapshot(tmp_vault.root) == before


def test_round_trip_after_many_changes(tmp_vault: Vault) -> None:
    result = run(
        tmp_vault,
        CreateEntity(
            id="topics/dbt",
            type="topic",
            title="dbt: models & tests",
            aliases=["Data Build Tool", "dbt-core"],
            summary="Used by Atlas.\n\nSee [[projects/atlas-migration]].",
        ),
        fact("self", "has_skill", "[[topics/dbt]]", "Is good at dbt", extra={"level": "advanced"}),
        fact("self", "interest", "Bouldering: indoor, twice a week"),
        fact("projects/atlas-migration", "note", "Cut-over (phase 2) needs sign-off"),
        AddFact(
            subject="self",
            p="has_flaw",
            o="Overcommits",
            text="Often overcommits",
            kind="inferred",
            evidence=['Alex: "I said yes again"', "ep-20261025-1412-career"],
        ),
        SupersedeFact(fact_id="f-n2o3p4", to="2026-11"),
        AddObservation(text="Alex · tired after the release", importance=4),
        AddIntention(text="Ask about the sign-off", when=["projects/atlas-migration"]),
        review=False,
    )
    assert result.skipped == [] and len(result.applied) == 8
    assert_clean(tmp_vault)
    texts = snapshot(tmp_vault.root)
    head = gitops.head(tmp_vault.root)
    v = fresh(tmp_vault)
    for p in v.pages():  # parse → render is byte-stable for every page the writer produced
        assert render_page(parse_page(render_page(p))[0]) == render_page(p)
    assert run(tmp_vault).commit is None
    assert snapshot(tmp_vault.root) == texts and gitops.head(tmp_vault.root) == head


# --- atomicity -----------------------------------------------------------------------------------


def test_failed_write_leaves_vault_unchanged(
    tmp_vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = snapshot(tmp_vault.root)
    head = gitops.head(tmp_vault.root)
    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src: object, dst: object) -> None:
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(writer.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="disk full"):
        run(
            tmp_vault,
            CreateEntity(id="topics/dbt", type="topic", title="dbt"),
            fact("self", "hometown", "Braga"),
            fact("projects/atlas-migration", "uses", "Kafka"),
            AddObservation(text="Alex was tired", importance=2),
        )
    assert calls["n"] >= 3
    monkeypatch.setattr(writer.os, "replace", real_replace)
    assert snapshot(tmp_vault.root) == before
    assert gitops.head(tmp_vault.root) == head
    assert git(tmp_vault, "status", "--porcelain") == ""


def test_failed_commit_leaves_vault_unchanged(
    tmp_vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid = run(tmp_vault, _inferred()).review_id
    assert rid
    before = snapshot(tmp_vault.root)

    def broken(*_args: object) -> str:
        raise gitops.GitError("index.lock exists")

    monkeypatch.setattr(writer, "_commit", broken)
    with pytest.raises(gitops.GitError):
        run(tmp_vault, fact("self", "hometown", "Braga"))
    with pytest.raises(gitops.GitError):
        approve(rid, tmp_vault, now=NOW)
    with pytest.raises(gitops.GitError):
        reject(rid, "no", tmp_vault, now=NOW)
    assert snapshot(tmp_vault.root) == before
    assert git(tmp_vault, "status", "--porcelain") == ""


def test_merge_repoints_degree_school_links(tmp_vault: Vault) -> None:
    """``at::`` (schema v0.3) is a link like ``o::``: a merge repoints it, round trip intact."""
    run(
        tmp_vault,
        CreateEntity(id="orgs/westbridge", type="org", title="Westbridge University"),
        CreateEntity(id="orgs/westbridge-uni", type="org", title="Westbridge Uni"),
        fact("self", "studied_at", "[[orgs/westbridge]]", valid_from="2018", valid_to="2020"),
        fact(
            "self",
            "degree",
            "MSc Data Science",
            "Holds the degree MSc Data Science from Westbridge Uni",
            extra={"at": "[[orgs/westbridge-uni]]"},
        ),
        review=False,
    )
    op = MergeEntities(keep="orgs/westbridge", merge="orgs/westbridge-uni")
    run(tmp_vault, op, review=False)
    degree = next(f for f in page(tmp_vault, "self/profile").facts if f.o == "MSc Data Science")
    assert degree.extra == {"at": "[[orgs/westbridge]]"}
    assert_clean(tmp_vault)
