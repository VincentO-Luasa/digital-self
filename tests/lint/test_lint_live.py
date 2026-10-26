"""Live lint on the eval's dirty vault (seed 1) with the local model; records the decisions as
replay fixtures when ``BRAIN_RECORD_FIXTURES=1`` (embeddings and rerank off, so the replay with a
fake model asks the same questions). Fictional data only."""

from __future__ import annotations

import os
import shutil

import pytest
from linttools import FIXTURES  # type: ignore[import-not-found]

from brain.config import load_settings
from brain.eval import lint_eval as ev
from brain.lint import LintOptions, run_lint
from brain.llm import LLM
from brain.llm.recorder import Recorder
from brain.vault.store import Vault

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def llm():
    settings = load_settings()
    recorder = None
    if os.environ.get("BRAIN_RECORD_FIXTURES") == "1":
        shutil.rmtree(FIXTURES, ignore_errors=True)
        recorder = Recorder(FIXTURES, prefix="lint-")
    client = LLM(settings.llm, recorder=recorder)
    if not client.health():
        pytest.skip("model server not running")
    yield client
    client.close()


def _run(vault: Vault, llm: LLM):
    options = LintOptions(full=True, today=ev.EVAL_TODAY, embeddings=False, rerank=False)
    return run_lint(vault, llm, options)


def test_live_dirty_vault_is_cleaned(dirty: tuple[Vault, ev.Gold], llm: LLM) -> None:
    vault, gold = dirty
    report = _run(vault, llm)
    for f in report.findings:
        print(f.check, f.key, "|", f.action, "|", f.decision)
    hits = {ev.classify(op, gold, report) for f in report.findings for op in f.ops}
    assert None not in hits, "a wrong change"
    assert len(hits) >= 10


def test_live_clean_vault_changes_nothing(clean_vault: Vault, llm: LLM) -> None:
    report = _run(clean_vault, llm)
    assert [op for f in report.findings for op in f.ops] == []
