"""The evaluation set itself: every document has a valid gold file, and the reverse."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.eval.ingest_eval import EVAL_DIR, load_gold, load_gold_set
from brain.eval.ingest_eval import gold_problems as problems_of
from brain.ingest.readers import load_document
from brain.tokens import estimate_tokens

GOLD_FILES = sorted((EVAL_DIR / "gold").glob("*.yaml"))
DOCS = sorted(p for p in (EVAL_DIR / "docs").iterdir() if p.is_file())


def test_every_document_has_one_gold_file() -> None:
    golds = load_gold_set()
    assert sorted(g.path.name for g in golds) == sorted(p.name for p in DOCS)
    assert len(DOCS) >= 6
    assert len({p.suffix for p in DOCS}) >= 4  # html, pdf, md, txt


@pytest.mark.parametrize("path", GOLD_FILES, ids=lambda p: p.stem)
def test_gold_file_is_valid(path: Path) -> None:
    gold = load_gold(path)
    assert problems_of(gold) == []
    assert gold.owner, "the owner's names let the scorer spot a page made for the owner"
    assert gold.level("optional")


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_document_reads_and_has_a_useful_size(path: Path) -> None:
    doc = load_document(path)
    assert 400 <= estimate_tokens(doc.text) <= 2500


def test_only_fictional_persona() -> None:
    """The set uses the Maya Lindqvist persona, distinct from the unit tests' Alex Rivera."""
    for path in [*DOCS, *GOLD_FILES]:
        if path.suffix == ".pdf":
            continue
        assert "Alex Rivera" not in path.read_text(encoding="utf-8")
