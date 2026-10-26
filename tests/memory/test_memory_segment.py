"""Segmentation: gate, importance, repairs (no model server)."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from memtools import Decider, fake, segmentation  # type: ignore[import-not-found]

from brain.memory import segment as seg
from brain.vault.raw import Turn
from brain.vault.store import Vault

AT = datetime(2026, 10, 27, 21, 0)


def turns(*texts: str) -> list[Turn]:
    roles = ["user", "assistant"]
    return [Turn(n + 1, roles[n % 2], AT, t) for n, t in enumerate(texts)]  # type: ignore[arg-type]


def labels(**kw) -> seg.SegmentedEpisode:
    base = dict(
        first_turn=1,
        last_turn=1,
        topic="t",
        regions=[],
        mentions=[],
        gist="g",
        summary="s",
        emotion="none",
        goal="none",
        future=False,
        remember=False,
        change="none",
    )
    return seg.SegmentedEpisode(**{**base, **kw})


@pytest.mark.parametrize(
    ("kw", "score"),
    [
        ({}, 2),
        ({"emotion": "mild", "goal": "related"}, 4),
        ({"emotion": "strong", "change": "major"}, 8),
        ({"remember": True, "future": True}, 6),
        ({"emotion": "strong", "goal": "central", "change": "major", "remember": True}, 10),
    ],
)
def test_importance_is_computed_in_code(kw: dict, score: int) -> None:
    assert seg.importance(labels(**kw)) == score


def test_schema_enforces_life_map_regions(mvault: Vault) -> None:
    schema = seg.segmentation_schema(tuple(mvault.lifemap.regions))
    regions = schema.model_json_schema()["$defs"]["SegmentedEpisode"]["properties"]["regions"]
    assert "career.current_job" in regions["items"]["enum"]
    with pytest.raises(ValueError):
        schema.model_validate(segmentation({"first_turn": 1, "last_turn": 1, "regions": ["nope"]}))


def test_small_talk_is_skipped_by_the_gate(mvault: Vault) -> None:
    llm = fake([], Decider(personal=0.03))
    report = seg.segment(
        llm,
        turns("hey", "hi!", "tell me a joke"),
        lifemap=mvault.lifemap,
        owner="Alex Rivera",
        channel="chat",
        today=date(2026, 10, 27),
    )
    assert report.skipped and not report.segments
    assert not [c for c in llm.calls if c[0] == "structured"]  # no segmentation call
    assert "p(personal)=0.030" in report.notes[0]


def test_gate_failure_lets_the_conversation_through(mvault: Vault) -> None:
    def broken(text, questions):
        raise RuntimeError("server hiccup")

    llm = fake([segmentation({"first_turn": 1, "last_turn": 1, "regions": ["projects"]})])
    llm.decider = broken
    report = seg.segment(
        llm,
        turns("Atlas is in testing"),
        lifemap=mvault.lifemap,
        owner="Alex Rivera",
        channel="chat",
        today=date(2026, 10, 27),
    )
    assert len(report.segments) == 1 and "no decision" in report.notes[0]


def test_clean_episodes_repairs_ranges_regions_and_gists(mvault: Vault) -> None:
    t = turns("a", "b", "c", "d", "e", "f")
    raw = [
        labels(first_turn=4, last_turn=9, topic="second", regions=["projects", "nope"]),
        labels(first_turn=0, last_turn=4, topic="first", gist=" ".join(["word"] * 40)),
        labels(first_turn=2, last_turn=2, topic="assistant only"),
    ]
    notes: list[str] = []
    out = seg.clean_episodes(raw, t, mvault.lifemap, notes)
    assert [(s.topic, s.first, s.last) for s in out] == [("first", 1, 4), ("second", 5, 6)]
    assert out[1].regions == ["projects"]
    assert len(out[0].gist.split()) == seg.GIST_WORDS
    assert any("unknown regions" in n for n in notes)
    assert any("no owner turn" in n for n in notes)


def test_messages_list_regions_first_and_clip_the_assistant(mvault: Vault) -> None:
    long = " ".join(["blah"] * 100)
    messages = seg.build_messages(
        turns("I moved to Braga", long),
        lifemap=mvault.lifemap,
        owner="Alex Rivera",
        channel="interview",
        today=date(2026, 10, 27),
        regions_hint=["identity"],
    )
    user = messages[1]["content"]
    assert user.startswith("REGIONS (id: life area):\n- identity: Identity & basics\n")
    assert "interviewed about: identity" in user
    assert "[1] OWNER: I moved to Braga" in user
    assert "[2] ASSISTANT: " + " ".join(["blah"] * 60) + " …" in user
