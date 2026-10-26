"""Coverage of the life map from the facts table: slots, staleness, per-entity regions."""

from __future__ import annotations

from datetime import date

import pytest
from interview_helpers import LATER, TODAY  # type: ignore[import-not-found]

from brain.interview.coverage import (
    KnownFact,
    _Facts,
    confirmed_date,
    score_slot,
)
from brain.vault.lifemap import Slot


def fact(o: str, *, p: str = "speaks", subject: str = "self", **kw) -> KnownFact:
    rec = kw.pop("rec", date(2026, 1, 1))
    return KnownFact(
        id=kw.pop("id", f"f-{o.casefold()[:6]}"),
        subject=subject,
        p=p,
        o=o,
        text=kw.pop("text", f"{p} {o}"),
        valid_from=kw.pop("valid_from", None),
        valid_to=kw.pop("valid_to", None),
        rec=rec,
        confirmed=kw.pop("confirmed", rec),
        current=kw.pop("current", True),
    )


def test_self_regions_on_the_fixture_vault(env) -> None:
    report = env.report(TODAY)
    identity = report.regions["identity"]
    status = {s.slot.id: s.status for s in identity.slots}
    assert status == {
        "name": "filled",
        "city": "filled",
        "hometown": "empty",
        "languages": "filled",
    }
    assert identity.coverage == pytest.approx(0.75)

    skills = report.regions["skills"]  # 4 skills of the 5 wanted: partial credit
    assert skills.slots[0].status == "partial"
    assert skills.coverage == pytest.approx(0.8)

    # timeless slots count history: the degree and the school that ended in 2018
    assert report.regions["education"].coverage == 1.0
    history = {s.slot.id: s.status for s in report.regions["career.history"].slots}
    assert history == {"past_roles": "filled", "achievements": "empty"}

    assert report.regions["self.values_goals"].coverage == 0.0
    assert 0 < report.overall() < 1


def test_present_slots_need_current_facts_valid_today(env) -> None:
    city = next(s for s in env.report(TODAY).regions["identity"].slots if s.slot.id == "city")
    # Porto is current; Lisbon (history, ended 2024) is shown but does not count
    assert [f.o for f in city.facts] == ["Porto"]


def test_slots_go_stale_after_stale_days(env) -> None:
    projects = env.report(LATER).regions["projects"]
    status = {s.slot.id: s.status for s in projects.slots}
    assert status == {
        "goal": "filled",
        "status": "stale",  # noted 2026-10-25, stale after 30 days
        "stakeholders": "filled",
        "deadline": "stale",  # stale after 60 days
    }
    assert projects.coverage == pytest.approx(0.5)
    assert env.report(TODAY).regions["projects"].coverage == 1.0


def test_confirmation_in_sources_refreshes_a_fact() -> None:
    today = date(2026, 10, 26)
    sources = ["raw/docs/cv.md", "ep-20261020-0900-career", "[[r-abcdef]]"]
    assert confirmed_date(date(2026, 1, 1), sources, today) == date(2026, 10, 20)
    assert confirmed_date(date(2026, 1, 1), ["raw/docs/2026-03-02-cv.html"], today) == date(
        2026, 3, 2
    )
    assert confirmed_date(date(2026, 1, 1), ["ep-20300101-0000-x"], today) == date(2026, 1, 1)

    slot = Slot(id="city", p="lives_in", subject="self", stale_days=30)
    old = fact("Porto", p="lives_in", rec=date(2026, 1, 1))
    assert score_slot("identity", slot, [old], today).status == "stale"
    confirmed = fact("Porto", p="lives_in", rec=date(2026, 1, 1), confirmed=date(2026, 10, 20))
    assert score_slot("identity", slot, [confirmed], today).status == "filled"


def test_multi_slots_count_distinct_fresh_values() -> None:
    slot = Slot(id="languages", p="speaks", subject="self", multi=True, min=3)
    facts = [fact("English"), fact("english", id="f-dup"), fact("Spanish")]
    scored = score_slot("identity", slot, facts, date(2026, 10, 26))
    assert (scored.status, scored.score) == ("partial", pytest.approx(2 / 3))
    assert score_slot("identity", slot, [], date(2026, 10, 26)).status == "empty"


def test_per_entity_regions(env) -> None:
    report = env.report(TODAY)
    friends = report.regions["relationships.friends"]
    assert [e.id for e in friends.entities] == ["people/sam-okafor"]
    assert friends.private and friends.entities[0].private
    work = report.regions["relationships.work"]
    assert [e.id for e in work.entities] == ["people/maya-lindqvist"]
    assert work.coverage == 0.0 and work.slots[0].ask == "How do you work with Maya Lindqvist?"
    # a region with no entity yet scores 0 (the interview asks who there is)
    assert report.regions["relationships.family"].coverage == 0.0
    assert not report.regions["relationships.family"].slots
    # a finished project is not asked about
    assert [e.id for e in report.regions["projects"].entities] == ["projects/atlas-migration"]
    # a person no region claims, with no relationship facts, is unplaced
    assert [u.id for u in report.unplaced] == ["people/theo-marchetti"]


def test_relationship_stated_on_the_owner_page_counts_for_the_person() -> None:
    facts = _Facts(
        [fact("people/x", p="friend_of", subject="self"), fact("Porto", p="lives_in", subject="y")]
    )
    assert [f.o for f in facts.about("people/x", "friend_of")] == ["people/x"]
    assert facts.about("people/x", "lives_in") == []
