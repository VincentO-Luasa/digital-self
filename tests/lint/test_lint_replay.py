"""Replay of a recorded live lint (tests/fixtures/lint/llm) on the eval's dirty vault: the same
questions are asked, every proposed change fixes a known noise item, and once the owner approves
every review the vault is clean. Fictional data only."""

from __future__ import annotations

from linttools import Replay  # type: ignore[import-not-found]

from brain.eval import lint_eval as ev
from brain.lint import LintOptions, apply_lint, run_lint
from brain.llm import FakeLLM
from brain.vault.store import Vault
from brain.vault.writer import pending_reviews

OPTIONS = LintOptions(full=True, today=ev.EVAL_TODAY, embeddings=False, rerank=False)


def test_replay_fixes_the_noise_and_nothing_else(dirty: tuple[Vault, ev.Gold]) -> None:
    vault, gold = dirty
    replay = Replay()
    report = run_lint(vault, FakeLLM(decider=replay), OPTIONS)
    assert replay.missing == [], "a question changed: re-record (see conftest)"
    hits = [ev.classify(op, gold, report) for f in report.findings for op in f.ops]
    assert None not in hits
    assert set(hits) == set(gold.items)
    apply_lint(vault, report, now=ev.EVAL_NOW)
    score = ev.score("replay", vault, report, gold, 0.0)
    assert score.wrong == [] and sorted(score.fixed) == sorted(gold.items)
    assert score.asks_matched == ["two-cities"]
    # every lint review was approved; nothing of lint's is left pending
    assert all(r.origin != "lint" for r in pending_reviews(Vault(vault.root)))


def test_replay_clean_vault_proposes_nothing(clean_vault: Vault) -> None:
    replay = Replay()
    report = run_lint(clean_vault, FakeLLM(decider=replay), OPTIONS)
    assert replay.missing == []
    assert [op for f in report.findings for op in f.ops] == []
