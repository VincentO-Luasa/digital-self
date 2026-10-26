"""Writer support for graph cleaning (schema v0.5): demote / retype / refile, evidence on
inferred facts, and what a merge does to the facts it brings together.

Fictional persona only (Alex Rivera, Northwind Labs, Sam Okafor, Atlas Migration, Fabrikam).
"""

from __future__ import annotations

import subprocess
from datetime import date, datetime

import pytest

from brain.vault import gitops
from brain.vault.generated import build_core
from brain.vault.models import Page
from brain.vault.ops import (
    AddFact,
    ChangeSet,
    CreateEntity,
    DemoteEntity,
    FactSpec,
    MergeEntities,
    Operation,
    Origin,
    RefileFact,
    RetypeEntity,
    SupersedeFact,
)
from brain.vault.render import render_page
from brain.vault.store import Vault
from brain.vault.writer import ApplyResult, apply, approve, pending_reviews

NOW = datetime(2026, 11, 2, 9, 30)
SRC = ["ep-20261102-0915-checkin"]


def run(vault: Vault, *ops: Operation, origin: Origin = "lint", **kw: object) -> ApplyResult:
    cs = ChangeSet(ops=list(ops), origin=origin, src=SRC, created=NOW)
    return apply(cs, vault, now=NOW, **kw)  # type: ignore[arg-type]


def fact(subject: str, p: str, o: str, text: str | None = None, **kw: object) -> AddFact:
    return AddFact(subject=subject, p=p, o=o, text=text or f"{p} {o}", kind="stated", **kw)


def page(vault: Vault, page_id: str) -> Page:
    found = Vault(vault.root).page(page_id)
    assert found is not None, page_id
    return found


def assert_clean(vault: Vault) -> None:
    status = subprocess.run(
        ["git", "-C", str(vault.root), "status", "--porcelain"], capture_output=True, text=True
    )
    assert status.stdout == ""
    v = Vault(vault.root)
    assert v.warnings() == []
    for p in v.pages():
        assert render_page(p) == v.page_path(p.id).read_text(encoding="utf-8"), p.id


# --- evidence on inferred facts ----------------------------------------------------------------


def _inferred(evidence: list[str]) -> AddFact:
    return AddFact(
        subject="projects/atlas-migration",
        p="part_of",
        o="[[orgs/northwind-labs]]",
        text="Atlas Migration is part of the work at Northwind Labs",
        kind="inferred",
        evidence=evidence,
    )


@pytest.mark.parametrize(
    ("origin", "evidence", "outcome"),
    [
        ("lint", [], "skipped"),
        ("lint", ["  "], "skipped"),
        ("lint", ["ep-20261025-1412-career"], "queued"),
        ("external", ["ep-20261025-1412-career"], "skipped"),
        ("external", ["ep-20261025-1412-career", "f-t2u3v4"], "queued"),
    ],
)
def test_inferred_facts_need_evidence(
    tmp_vault: Vault, origin: Origin, evidence: list[str], outcome: str
) -> None:
    result = run(tmp_vault, _inferred(evidence), origin=origin)
    if outcome == "skipped":
        assert result.skipped and "without evidence::" in result.skipped[0][1]
    else:
        assert result.queued and result.queued[0][1] == "inferred fact"


def test_inferred_replacement_needs_evidence(tmp_vault: Vault) -> None:
    new = FactSpec(p="status", o="paused", text="Atlas is paused", kind="inferred")
    result = run(tmp_vault, SupersedeFact(fact_id="f-n2o3p4", new=new), review=False)
    assert result.skipped and "without evidence::" in result.skipped[0][1]


# --- merge reconciliation ----------------------------------------------------------------------


@pytest.fixture
def duplicate_sam(tmp_vault: Vault) -> Vault:
    """A second page for Sam that restates one fact, clashes on another, and is named by a
    pending review."""
    dup = "people/samuel-okafor"
    run(
        tmp_vault,
        CreateEntity(
            id=dup, type="person", title="Samuel Okafor", regions=["relationships.friends"]
        ),
        fact(
            dup,
            "friend_of",
            "[[self]]",
            "Close friend of Alex since university",
            extra={"rel": "close friend"},
            **{"from": "2014"},
        ),
        fact(dup, "last_seen", "2026-09-30", "Last caught up with Alex on 2026-09-30"),
        fact(dup, "related_to", "[[people/sam-okafor]]", "Same person as Sam?"),
        fact(
            "projects/atlas-migration", "stakeholder", f"[[{dup}]]", "Samuel reviews Atlas runbooks"
        ),
        review=False,
    )
    run(
        tmp_vault,
        fact(dup, "knows", "[[orgs/northwind-labs]]", "Knows Northwind Labs"),
        origin="extract",
    )
    return tmp_vault


def test_merge_folds_restatements_and_queues_clashes(duplicate_sam: Vault) -> None:
    vault = duplicate_sam
    [pending] = pending_reviews(vault)
    result = run(
        vault, MergeEntities(keep="people/sam-okafor", merge="people/samuel-okafor"), review=False
    )
    assert result.applied
    sam = page(vault, "people/sam-okafor")
    friend = [f for f in sam.facts if f.p == "friend_of"]
    assert len(friend) == 1 and friend[0].id == "f-y5z6a7"  # the restatement was folded in
    assert set(friend[0].src) == {"ep-20261021-2030-friends", *SRC}
    # the self-reference the merge created went to History
    assert not any(f.p == "related_to" for f in sam.facts)
    assert any(f.p == "related_to" and f.valid_to for f in sam.history)
    # two last_seen values: the owner chooses, in a new review
    assert [f.p for f in sam.facts].count("last_seen") == 2
    [(clash, why)] = result.queued
    assert isinstance(clash, SupersedeFact) and "clash after merging" in why
    assert result.review_id
    # the pending review that named the merged page now names the kept one
    text = pending.path.read_text(encoding="utf-8")
    assert "people/samuel-okafor" not in text and '"subject": "people/sam-okafor"' in text
    log = vault.log_path.read_text(encoding="utf-8")
    assert "restates ^f-y5z6a7" in log and "self-reference retired" in log
    assert_clean(vault)


def test_merge_folds_restatements_on_other_pages(tmp_vault: Vault) -> None:
    run(
        tmp_vault,
        CreateEntity(id="orgs/northwind", type="org", title="Northwind"),
        fact("self", "member_of", "[[orgs/northwind-labs]]", "Member of Northwind Labs"),
        fact("self", "member_of", "[[orgs/northwind]]", "Member of Northwind", **{"from": "2023"}),
        review=False,
    )
    run(tmp_vault, MergeEntities(keep="orgs/northwind-labs", merge="orgs/northwind"), review=False)
    members = [f for f in page(tmp_vault, "self/profile").facts if f.p == "member_of"]
    assert len(members) == 1 and members[0].valid_from == "2023"  # the more precise one stays
    assert_clean(tmp_vault)


# --- demote_entity -----------------------------------------------------------------------------


@pytest.fixture
def topic_page(tmp_vault: Vault) -> Vault:
    run(
        tmp_vault,
        CreateEntity(id="projects/stream-processing", type="project", title="Stream Processing"),
        fact(
            "self",
            "interest",
            "[[projects/stream-processing]]",
            "Is interested in stream processing",
        ),
        fact(
            "projects/atlas-migration",
            "related_to",
            "[[projects/stream-processing|streaming]]",
            "Atlas touches streaming",
        ),
        review=False,
    )
    return tmp_vault


def test_demote_is_reviewed_then_links_become_values(topic_page: Vault) -> None:
    op = DemoteEntity(id="projects/stream-processing")
    assert run(topic_page, op).queued == [(op, "entity demotion")]
    [review] = pending_reviews(topic_page)
    result = approve(review.id, topic_page, now=NOW)
    assert result.applied == [op]
    assert not topic_page.page_path("projects/stream-processing").exists()
    interests = [
        f for p in Vault(topic_page.root).pages_about("self") for f in p.facts if f.p == "interest"
    ]
    assert [f.o for f in interests] == ["Stream Processing"]
    atlas = page(topic_page, "projects/atlas-migration")
    assert atlas.facts[-1].o == "streaming"
    assert_clean(topic_page)


def test_demote_refuses_a_page_with_facts(tmp_vault: Vault) -> None:
    result = run(tmp_vault, DemoteEntity(id="projects/atlas-migration"), review=False)
    assert result.skipped and "fact(s) of its own" in result.skipped[0][1]
    result = run(tmp_vault, DemoteEntity(id="self/profile"), review=False)
    assert result.skipped and "self pages" in result.skipped[0][1]


# --- retype_entity -----------------------------------------------------------------------------


def test_retype_moves_the_page_and_repoints_links(tmp_vault: Vault) -> None:
    run(
        tmp_vault,
        CreateEntity(
            id="projects/kestrel-data-cooperative", type="project", title="Kestrel Data Cooperative"
        ),
        fact(
            "self",
            "member_of",
            "[[projects/kestrel-data-cooperative]]",
            "Member of Kestrel Data Cooperative",
        ),
        fact("projects/kestrel-data-cooperative", "located_in", "Lisbon", "Based in Lisbon"),
        review=False,
    )
    op = RetypeEntity(id="projects/kestrel-data-cooperative", type="org")
    assert run(tmp_vault, op).queued == [(op, "entity retype")]
    result = run(tmp_vault, op, review=False)
    assert result.applied == [op]
    org = page(tmp_vault, "orgs/kestrel-data-cooperative")
    assert org.type == "org" and org.subject == "orgs/kestrel-data-cooperative"
    assert [f.p for f in org.facts] == ["located_in"]
    assert not tmp_vault.page_path("projects/kestrel-data-cooperative").exists()
    member = [f for f in page(tmp_vault, "self/profile").facts if f.p == "member_of"]
    assert member[0].o == "[[orgs/kestrel-data-cooperative]]"
    assert_clean(tmp_vault)


@pytest.mark.parametrize(
    ("op", "reason"),
    [
        (RetypeEntity(id="projects/atlas-migration", type="project"), "already a project"),
        (RetypeEntity(id="projects/atlas-migration", type="self"), "cannot become a self page"),
        (RetypeEntity(id="projects/atlas-migration", type="org", new_id="people/x"), "bad id"),
        (
            RetypeEntity(id="projects/atlas-migration", type="org", new_id="orgs/northwind-labs"),
            "already exists",
        ),
    ],
)
def test_retype_validation(tmp_vault: Vault, op: RetypeEntity, reason: str) -> None:
    result = run(tmp_vault, op, review=False)
    assert result.skipped and reason in result.skipped[0][1]


# --- refile_fact -------------------------------------------------------------------------------


@pytest.fixture
def tidepool(tmp_vault: Vault) -> Vault:
    run(
        tmp_vault,
        CreateEntity(id="projects/tidepool", type="project", title="Tidepool"),
        CreateEntity(id="orgs/fabrikam-analytics", type="org", title="Fabrikam Analytics"),
        fact(
            "self",
            "goal",
            "Ship Tidepool's real-time alerting by March",
            "Wants to ship Tidepool's real-time alerting by March",
        ),
        fact(
            "self",
            "responsibility",
            "Demand forecasting dashboards",
            "Built demand forecasting dashboards",
            extra={"at": "[[orgs/northwind-labs]]"},
        ),
        review=False,
    )
    return tmp_vault


def _fact_id(vault: Vault, p: str, words: str) -> str:
    for pg in Vault(vault.root).pages():
        for f in (*pg.facts, *pg.history):
            if f.p == p and words in f.text:
                return f.id
    raise AssertionError(words)


def test_refile_moves_a_goal_to_its_project(tidepool: Vault) -> None:
    fid = _fact_id(tidepool, "goal", "Tidepool")
    op = RefileFact(fact_id=fid, subject="projects/tidepool")
    assert run(tidepool, op).queued == [(op, "fact re-filed")]
    result = run(tidepool, op, review=False)
    assert result.applied == [op]
    moved = page(tidepool, "projects/tidepool").facts
    assert [(f.id, f.p, f.kind) for f in moved] == [(fid, "goal", "stated")]
    assert moved[0].src == SRC  # the change set is provenance too (here the same episode)
    assert not any(f.id == fid for f in page(tidepool, "self/goals").facts)
    assert_clean(tidepool)


def test_refile_links_a_literal_and_sets_at(tidepool: Vault) -> None:
    run(
        tidepool,
        RefileFact(fact_id="f-f6b7a2", o="orgs/fabrikam-analytics"),
        RefileFact(
            fact_id=_fact_id(tidepool, "responsibility", "forecasting"),
            at="[[orgs/fabrikam-analytics]]",
        ),
        review=False,
    )
    profile = page(tidepool, "self/profile")
    job = next(f for f in profile.history if f.id == "f-f6b7a2")
    assert job.o == "[[orgs/fabrikam-analytics]]" and job.valid_to == "2023-01"
    assert profile.history.index(job) == 1  # kept its place
    duty = next(f for f in profile.facts if f.p == "responsibility" and "forecasting" in f.text)
    assert duty.extra["at"] == "[[orgs/fabrikam-analytics]]"
    assert_clean(tidepool)
    # the card no longer lists a duty of a job that has ended as a current one
    core = build_core(Vault(tidepool.root), today=date(2026, 11, 2))
    assert "Demand forecasting" not in core


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({}, "nothing to change"),
        ({"subject": "self"}, "nothing to change"),
        ({"subject": "projects/nowhere"}, "unknown entity"),
        ({"at": "Fabrikam"}, "must link to a page"),
        ({"p": "lives_in", "at": "[[orgs/fabrikam-analytics]]"}, "does not qualify"),
        ({"subject": "projects/atlas-migration"}, "would contradict"),
    ],
)
def test_refile_validation(tidepool: Vault, kwargs: dict, reason: str) -> None:
    fid = _fact_id(tidepool, "goal", "Tidepool")
    result = run(tidepool, RefileFact(fact_id=fid, **kwargs), review=False)
    assert result.skipped and reason in result.skipped[0][1]


def test_refile_onto_a_restatement_folds_it(tidepool: Vault) -> None:
    run(
        tidepool,
        fact("projects/tidepool", "goal", "Ship Tidepool's real-time alerting by March"),
        review=False,
    )
    fid = _fact_id(tidepool, "goal", "Wants to ship")
    run(tidepool, RefileFact(fact_id=fid, subject="projects/tidepool"), review=False)
    goals = page(tidepool, "projects/tidepool").facts
    assert len(goals) == 1 and goals[0].id != fid
    assert_clean(tidepool)


def test_refile_of_a_history_fact_stays_in_history(tidepool: Vault) -> None:
    gitops.commit_all(tidepool.root, "noop")
    result = run(tidepool, RefileFact(fact_id="f-l6s2b3", p="hometown"), review=False)
    assert result.applied
    assert any(
        f.id == "f-l6s2b3" and f.p == "hometown" for f in page(tidepool, "self/profile").history
    )
