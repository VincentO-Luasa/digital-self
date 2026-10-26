"""Region choice (privacy, boosts, names) and the order of the questions in a region."""

from __future__ import annotations

import pytest
from interview_helpers import LATER, TODAY  # type: ignore[import-not-found]

from brain.interview.planner import (
    RegionError,
    build_plan,
    choose_region,
    context_lines,
    region_scores,
    resolve_region,
)
from brain.interview.questions import COVERAGE, GapItem, gap_items
from brain.vault.lifemap import load_lifemap

PRIVATE_FIRST = """\
version: 0.1
regions:
  inner.secrets:
    label: Secrets
    weight: 5.0
    privacy: private
    slots:
      - {id: fear, p: fears, subject: self, ask: "What scares you?"}
  inner.dreams:
    label: Dreams
    weight: 1.0
    privacy: private
    slots:
      - {id: dream, p: goal, subject: self, ask: "What do you dream of?"}
  hobbies.music:
    label: Music
    weight: 0.5
    slots:
      - {id: instrument, p: interest, subject: self, ask: "Do you play anything?"}
  hobbies.sport:
    label: Sport
    weight: 0.4
    privacy: private
    slots:
      - {id: sport, p: habit, subject: self, ask: "Which sport do you do?"}
"""


def test_default_choice_is_the_largest_weighted_gap(env) -> None:
    report = env.report(TODAY)
    choice = choose_region(report, [])
    assert choice is not None and choice.id == "self.values_goals"  # 0.9 × (1 − 0)
    scores = {s.id: s for s in region_scores(report, [])}
    assert scores["self.traits"].score == pytest.approx(0.8)  # private: never picked by default
    assert scores["self.traits"].private


def test_private_regions_only_when_named(env_factory) -> None:
    e = env_factory(TODAY)
    (e.vault.root / "life-map.yaml").write_text(PRIVATE_FIRST, encoding="utf-8")
    lifemap = load_lifemap(e.vault.root)
    from brain.interview.coverage import coverage_report

    report = coverage_report(e.db, lifemap, TODAY)
    assert choose_region(report, []).id == "hobbies.music"  # secrets would score 5.0
    assert choose_region(report, [], "inner.secrets").id == "inner.secrets"
    assert choose_region(report, [], "secrets").id == "inner.secrets"  # unique last part
    assert choose_region(report, [], "Dreams").id == "inner.dreams"  # by label
    assert choose_region(report, [], "sport").id == "hobbies.sport"  # unique last part
    assert choose_region(report, [], "hobbies").id == "hobbies.music"  # parent: non-private
    assert resolve_region(lifemap, "hobbies") == ["hobbies.music", "hobbies.sport"]
    with pytest.raises(RegionError, match="unknown region"):
        choose_region(report, [], "garden")


def test_a_parent_names_only_its_non_private_children(env_factory) -> None:
    e = env_factory(TODAY)
    (e.vault.root / "life-map.yaml").write_text(PRIVATE_FIRST, encoding="utf-8")
    from brain.interview.coverage import coverage_report

    report = coverage_report(e.db, load_lifemap(e.vault.root), TODAY)
    with pytest.raises(RegionError, match="only private"):
        choose_region(report, [], "inner")


def test_relationships_parent_on_the_default_map(env) -> None:
    report = env.report(TODAY)
    assert choose_region(report, [], "relationships").id == "relationships.work"
    assert choose_region(report, [], "friends").id == "relationships.friends"


def test_added_questions_boost_their_region(env) -> None:
    report = env.report(TODAY)
    added = [
        GapItem(text=f"q{n}", region="career.history", priority=5, origin="lint") for n in range(5)
    ]
    scores = {s.id: s for s in region_scores(report, added)}
    assert scores["career.history"].boost == pytest.approx(0.3)  # capped
    assert scores["career.history"].score == pytest.approx(0.6)  # 0.6 × (1 − 0.5) + 0.3
    assert scores["career.history"].score > scores["self.interests"].score  # 0.5 without items
    assert choose_region(report, added).id == "self.values_goals"  # a boost, not an override
    coverage_items = gap_items(report)  # coverage items never boost (already in coverage)
    assert all(s.boost == 0 for s in region_scores(report, coverage_items))


def test_plan_order_stale_then_empty_then_partial(env) -> None:
    report = env.report(LATER)
    plan = build_plan(report, "projects")
    assert [(p.mode, p.slot) for p in plan] == [
        ("confirm", "status"),
        ("confirm", "deadline"),
        ("discover", "more"),
    ]
    assert plan[0].cite == "Atlas is in the testing phase"
    assert plan[0].known[0].startswith("Atlas is in the testing phase (since 2026-10; noted")
    assert plan[0].topic == "Projects: Atlas Migration"
    assert plan[-1].known == ("Already known: Atlas Migration",)
    context = context_lines(report, plan[0])
    assert any("managed warehouse" in line for line in context)
    assert not any("go-live" in line for line in context)  # stale: asked in its own step

    job = build_plan(env.report(TODAY), "career.current_job")
    assert [(p.mode, p.slot) for p in job] == [
        ("ask", "manager"),
        ("more", "responsibilities"),
        ("discover", "more"),
    ]
    assert job[1].known == (job[1].known[0],) and "nightly batch" in job[1].known[0]
    assert context_lines(env.report(TODAY), job[0])  # starts from what is known


def test_plan_puts_added_items_first_and_skips_dismissed_gaps(env) -> None:
    report = env.report(TODAY)
    hometown = next(i for i in gap_items(report) if i.slot == "hometown")
    items = [
        GapItem(text="Is Porto home now?", region="identity", priority=2, origin="lint"),
        GapItem(text="Any nickname?", region="identity", priority=4, origin="owner"),
        GapItem(text="done", region="identity", origin="lint", done=True),
        GapItem(text=hometown.text, region="identity", slot="hometown", origin=COVERAGE, done=True),
    ]
    plan = build_plan(report, "identity", items)
    assert [p.hint for p in plan[:2]] == ["Any nickname?", "Is Porto home now?"]
    assert plan[0].mode == "item" and plan[0].question_id == items[1].id
    assert all(p.slot != "hometown" for p in plan)


def test_per_entity_plans(env) -> None:
    report = env.report(TODAY)
    family = build_plan(report, "relationships.family")
    assert [(p.mode, p.hint) for p in family] == [
        ("discover", "Who are the people in your family you're closest to?")
    ]
    work = build_plan(report, "relationships.work")
    assert [(p.mode, p.entity) for p in work] == [
        ("ask", "people/maya-lindqvist"),
        ("discover", None),
    ]
    assert work[0].hint == "How do you work with Maya Lindqvist?"
    assert build_plan(report, "relationships.work", max_entities=0)[0].mode == "discover"
