"""The paraphrase thresholds on labelled fictional pairs (``tests/fixtures/dedup/pairs.yaml``).

A wrong merge loses a fact, so the rule must not merge any pair labelled different. The scores
(cosine of the instructed embeddings, the typed decision's p(same) for pairs in the asked band)
are recorded by the live test; re-record after changing the pairs, the instruction or the
question: ``BRAIN_RECORD_FIXTURES=1 uv run pytest -m llm tests/ingest/test_ingest_dedup_calibration.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from brain.config import load_settings
from brain.ingest.dedup import (
    ASK_ABOVE,
    INSTRUCT,
    MERGE_ABOVE,
    SAME_P,
    _Item,
    question,
    words,
)
from brain.llm import LLM

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PAIRS = FIXTURES / "dedup" / "pairs.yaml"
SCORES = FIXTURES / "llm" / "dedup" / "pair-scores.json"
MIN_RECALL = 0.65  # paraphrases merged; precision must be 1.0


def pairs() -> list[dict[str, Any]]:
    return yaml.safe_load(PAIRS.read_text(encoding="utf-8"))["pairs"]


def statements(pair: dict[str, Any]) -> tuple[str, str]:
    a, b = (_Item(pair["p"], s["o"], s["text"], None, None) for s in (pair["a"], pair["b"]))
    return a.statement, b.statement


def merged(a: str, b: str, cos: float, p_same: float | None) -> bool:
    """``dedup._find`` for one pair."""
    if words(a) == words(b) or cos >= MERGE_ABOVE:
        return True
    return cos >= ASK_ABOVE and p_same is not None and p_same >= SAME_P


def outcome(rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    """(paraphrases merged, paraphrases, different pairs merged)"""
    hits = [merged(r["a"], r["b"], r["cos"], r["p_same"]) for r in rows]
    tp = sum(h and r["same"] for h, r in zip(hits, rows, strict=True))
    fp = sum(h and not r["same"] for h, r in zip(hits, rows, strict=True))
    return tp, sum(r["same"] for r in rows), fp


def test_recorded_scores_merge_no_different_pair() -> None:
    rows = json.loads(SCORES.read_text(encoding="utf-8"))
    labelled = [(*statements(p), p["same"]) for p in pairs()]
    assert [(r["a"], r["b"], r["same"]) for r in rows] == labelled, "re-record the scores"
    tp, n, fp = outcome(rows)
    assert fp == 0
    assert tp / n >= MIN_RECALL
    # the margins the thresholds keep (docs/decisions.md)
    different = [r for r in rows if not r["same"]]
    assert max(r["cos"] for r in different) <= MERGE_ABOVE - 0.04
    asked = [r for r in different if ASK_ABOVE <= r["cos"] < MERGE_ABOVE]
    assert all(r["p_same"] <= SAME_P - 0.5 for r in asked)


@pytest.mark.llm
def test_live_scores_merge_no_different_pair() -> None:
    llm = LLM(load_settings().llm)
    if not llm.health():
        llm.close()
        pytest.skip("local model server is not running")
    labelled = pairs()
    sides = [statements(p) for p in labelled]
    va = llm.embed([INSTRUCT + a for a, _ in sides])
    vb = llm.embed([INSTRUCT + b for _, b in sides])
    rows = []
    for pair, (a, b), x, y in zip(labelled, sides, va, vb, strict=True):
        cos = round(float(x @ y), 4)
        p_same = None
        if ASK_ABOVE <= cos < MERGE_ABOVE:
            decision = llm.decide(f"A: {a}\nB: {b}", [question(pair["p"])])[0]
            p_same = None if decision.refusal else round(decision.probabilities["same"], 4)
        rows.append(
            {"p": pair["p"], "same": pair["same"], "a": a, "b": b, "cos": cos, "p_same": p_same}
        )
    tp, n, fp = outcome(rows)
    print(f"paraphrases merged {tp}/{n} ({tp / n:.2f}), different pairs merged {fp}")
    if os.environ.get("BRAIN_RECORD_FIXTURES"):
        SCORES.parent.mkdir(parents=True, exist_ok=True)
        SCORES.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    assert fp == 0
    assert tp / n >= MIN_RECALL
