"""The memory evaluation set parses, uses the vocabulary, and the scorer counts what it should
(no model server)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from brain.eval import memory_eval as ev
from brain.vault.store import Vault

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "memory"))
from memtools import Decider, extraction, fake, segmentation  # type: ignore[import-not-found]  # noqa: E402

CASES = ev.load_set()


def test_the_set_covers_the_required_situations() -> None:
    names = {c.name for c in CASES}
    assert len(CASES) >= 8
    assert {"small-talk", "multi-topic", "restated", "implicit-contradiction"} <= names
    assert any(c.supersede for c in CASES) and any(c.queued for c in CASES)
    assert any(c.intentions for c in CASES)
    assert any(c.count == (0, 0) for c in CASES) and any(c.count[1] >= 2 for c in CASES)


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_gold_is_valid(case: ev.GoldCase) -> None:
    assert ev.gold_problems(case) == []


def test_scorer_counts_corrections_and_wrong_auto_applies(tmp_path: Path, git_env: None) -> None:
    case = next(c for c in CASES if c.name == "correction-current")
    vault = ev.seed_vault(tmp_path / "vault")
    transcript = ev.write_transcript(vault, case)
    segments = segmentation(
        {"first_turn": 1, "last_turn": 3, "topic": "moved", "regions": ["identity"]}
    )
    facts = extraction(
        [
            {
                "subject": "self",
                "predicate": "lives_in",
                "object": "Braga",
                "start": "2026-09",
                "label": "contradicts",
                "known": "f-d5e6f7",
            },
            # an extra explicit-looking change of the role: not in the gold → wrongly applied
            {
                "subject": "self",
                "predicate": "role",
                "object": "Commuter",
                "label": "contradicts",
                "known": "f-m4n5o6",
            },
        ]
    )
    from brain.memory import process_session

    report = process_session(
        vault,
        fake([segments, facts], Decider()),
        transcript,
        channel="chat",
        today=ev.EVAL_TODAY,
        refresh=False,
    )
    score = ev.score_case(case, report, vault, seconds=1.0)
    c = score.counts()
    assert (c["count_ok"], c["supersede_ok"], c["supersede"]) == (1, 1, 1)
    assert c["wrong_auto"] == 1 and "f-m4n5o6" in score.wrong_auto[0]
    assert c["applied"] == 2 and c["applied_matched"] == 1  # the role is unverified
    assert ev.metrics(c, 1.0)["wrongly_auto_applied"] == 1
    assert ev.report_json([score], {})["overall"]["supersessions_correct"] == 1.0


def test_small_talk_scores_an_empty_session(tmp_path: Path, git_env: None) -> None:
    case = next(c for c in CASES if c.name == "small-talk")
    vault = ev.seed_vault(tmp_path / "vault")
    transcript = ev.write_transcript(vault, case)
    from brain.memory import process_session

    report = process_session(
        vault,
        fake([], Decider(personal=0.02)),
        transcript,
        channel="chat",
        today=ev.EVAL_TODAY,
        refresh=False,
    )
    score = ev.score_case(case, report, Vault(vault.root), seconds=0.1)
    assert score.count_ok and score.counts()["wrong_auto"] == 0
