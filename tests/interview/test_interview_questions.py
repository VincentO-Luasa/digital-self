"""questions.md: generated from coverage, deterministic, keeps added items, readable by the card."""

from __future__ import annotations

from interview_helpers import LATER, TODAY  # type: ignore[import-not-found]

from brain.interview.questions import (
    COVERAGE,
    GapItem,
    add_question,
    gap_items,
    parse_questions,
    read_questions,
    refresh_questions,
    resolve_question,
    write_questions,
)
from brain.vault.generated import build_core


def test_gap_items_from_coverage(env) -> None:
    items = {(i.region, i.slot, i.about): i for i in gap_items(env.report(TODAY))}
    hometown = items[("identity", "hometown", None)]
    assert hometown.text == "Where did you grow up?"
    assert (hometown.reason, hometown.priority, hometown.origin) == ("empty", 5, COVERAGE)
    assert items[("skills", "technical", None)].reason == "partial, 4 of 5"
    family = items[("relationships.family", "discover", None)]
    assert family.private and family.reason == "none known yet"
    assert (
        items[(None, "relation", "people/theo-marchetti")].text == "Who is Theo Marchetti to you?"
    )
    assert not items[("relationships.work", "relation", "people/maya-lindqvist")].private
    assert ("projects", "status", "projects/atlas-migration") not in items  # fresh today

    stale = {(i.region, i.slot): i for i in gap_items(env.report(LATER))}
    status = stale[("projects", "status")]
    assert status.text == "Atlas is in the testing phase: still true?"
    assert status.reason == "stale, noted 2026-10-25"
    assert status.about == "projects/atlas-migration"


def test_file_is_deterministic_sorted_and_round_trips(env) -> None:
    assert write_questions(env.vault, env.report(TODAY))
    text = env.vault.questions_path.read_text(encoding="utf-8")
    assert not write_questions(env.vault, env.report(TODAY))  # unchanged → not rewritten
    lines = [line for line in text.splitlines() if line.startswith("- [")]
    priorities = [int(line.split("(priority:: ")[1][0]) for line in lines]
    assert priorities == sorted(priorities, reverse=True)
    assert lines[0].startswith("- [ ] Where did you grow up? (region:: identity)")
    assert "(privacy:: private)" in text
    parsed = parse_questions(text)
    assert [i.render() for i in parsed] == lines
    assert all(i.id.startswith("q-") and i.origin == COVERAGE for i in parsed)


def test_added_and_owner_items_survive_regeneration(env) -> None:
    write_questions(env.vault, env.report(TODAY))
    path = env.vault.questions_path
    path.write_text(path.read_text() + "- Ask me about the bike trip\n", encoding="utf-8")
    item = add_question(
        env.vault,
        "Is Atlas Migration part of your job at Northwind Labs (or a side project)?",
        region="projects",
        reason="graph cleaning: project without an owner org",
        priority=4,
        origin="lint",
        about="projects/atlas-migration",
    )
    assert item.origin == "lint" and not item.private
    assert add_question(env.vault, item.text, region="projects", origin="lint").id == item.id
    private = add_question(env.vault, "Who did you see at the reunion?", region="relationships")
    assert not private.private  # "relationships" is not a region itself
    friends = add_question(env.vault, "How is Sam's new flat?", region="relationships.friends")
    assert friends.private

    assert write_questions(env.vault, env.report(LATER)) is True
    items = {i.text: i for i in read_questions(env.vault)}
    kept = items[item.text]
    assert (kept.region, kept.priority, kept.about) == ("projects", 4, "projects/atlas-migration")
    assert kept.reason == "graph cleaning: project without an owner org"
    assert items["Ask me about the bike trip"].origin == "owner"
    assert sum(1 for i in items.values() if i.text == item.text) == 1


def test_dismissed_gaps_stay_dismissed_and_filled_gaps_disappear(env) -> None:
    write_questions(env.vault, env.report(LATER))
    first = next(i for i in read_questions(env.vault) if i.slot == "status")
    path = env.vault.questions_path
    path.write_text(path.read_text().replace(f"- [ ] {first.text}", f"- [x] {first.text}"))
    write_questions(env.vault, env.report(LATER))
    again = next(i for i in read_questions(env.vault) if i.id == first.id)
    assert again.done
    write_questions(env.vault, env.report(TODAY))  # the status is fresh again: gap gone
    assert all(i.id != first.id for i in read_questions(env.vault))


def test_resolve_and_refresh(env) -> None:
    item = add_question(env.vault, "Which conference did you speak at?", region="career.history")
    assert resolve_question(env.vault, item.id)
    assert not resolve_question(env.vault, item.id)
    assert next(i for i in read_questions(env.vault) if i.id == item.id).done
    assert refresh_questions(env.vault, env.db, today=TODAY) is True
    assert refresh_questions(env.vault, env.db, today=TODAY) is False


def test_core_card_shows_the_top_open_questions(env) -> None:
    write_questions(env.vault, env.report(TODAY))
    core = build_core(env.vault, today=TODAY)
    assert "To ask: Where did you grow up?" in core
    assert "(region::" not in core and "^q-" not in core


def test_parse_tolerates_hand_edits() -> None:
    text = "\n".join(
        [
            "# Questions",
            "* [X] Done thing (priority:: 9) ^q-aaaaaa",
            "- [ ] Odd (priority:: high) (region:: identity)",
            "- [ ]   ",
            "not a list line",
        ]
    )
    done, odd = parse_questions(text)
    assert done.done and done.priority == 5 and done.qid == "q-aaaaaa"
    assert odd.priority == 3 and odd.region == "identity" and odd.origin == "owner"
    assert GapItem(text="x", origin="owner").id == GapItem(text="X ", origin="owner").id
