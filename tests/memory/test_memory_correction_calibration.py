"""The owner-correction decision on labelled fictional cases
(``tests/fixtures/memory/corrections.yaml``).

Live (``-m llm``): asks the model and, with ``BRAIN_RECORD_FIXTURES=1``, records the scores in
``tests/fixtures/llm/memory/correction-scores.json``. Offline: the recorded scores must put every
explicit correction at or above ``CORRECTION_P`` and every other case below it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from brain.config import load_settings
from brain.llm import LLM
from brain.memory.extract import CORRECTION, CORRECTION_P, correction_input

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CASES = yaml.safe_load((FIXTURES / "memory" / "corrections.yaml").read_text())["cases"]
SCORES = FIXTURES / "llm" / "memory" / "correction-scores.json"


def case_input(case: dict) -> str:
    return correction_input(case["stored"], "\n".join(case["conversation"]))


def test_recorded_scores_separate_at_the_threshold() -> None:
    data = json.loads(SCORES.read_text())
    assert data["question"] == CORRECTION.model_dump(), "re-record: the question changed"
    scores = {s["id"]: s for s in data["scores"]}
    assert set(scores) == {c["id"] for c in CASES}, "re-record: the cases changed"
    for case in CASES:
        s = scores[case["id"]]
        assert s["input"] == case_input(case)
        if case["explicit"]:
            assert s["p"] >= CORRECTION_P, case["id"]
        else:
            assert s["p"] < CORRECTION_P, case["id"]


@pytest.mark.llm
def test_live_scores() -> None:
    client = LLM(load_settings().llm)
    if not client.health():
        pytest.skip("local model server is not running")
    rows = []
    try:
        for case in CASES:
            [d] = client.decide(case_input(case), [CORRECTION])
            p = 0.0 if d.refusal else d.probabilities.get("yes", 0.0)
            rows.append(
                {
                    "id": case["id"],
                    "explicit": case["explicit"],
                    "p": round(p, 4),
                    "input": case_input(case),
                }
            )
    finally:
        client.close()
    for r in rows:
        print(f"{'+' if r['explicit'] else '-'} {r['p']:.3f} {r['id']}")
    if os.environ.get("BRAIN_RECORD_FIXTURES"):
        SCORES.parent.mkdir(parents=True, exist_ok=True)
        payload = {"question": CORRECTION.model_dump(), "scores": rows}
        SCORES.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    wrong = [r["id"] for r in rows if (r["p"] >= CORRECTION_P) != r["explicit"]]
    assert not wrong
