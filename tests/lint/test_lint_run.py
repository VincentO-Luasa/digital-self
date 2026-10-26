"""Running and applying lint: questions (never twice), one review per finding, incremental runs,
hand edits committed apart, the CLI. Fictional persona only (Alex Rivera)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from linttools import Replay  # type: ignore[import-not-found]
from typer.testing import CliRunner

from brain import cli
from brain.config import Settings
from brain.eval import lint_eval as ev
from brain.interview.questions import read_questions
from brain.lint import LintOptions, apply_lint, run_lint
from brain.lint.run import last_rev, state_path
from brain.llm import FakeLLM
from brain.vault import gitops
from brain.vault.store import Vault
from brain.vault.writer import pending_reviews

OPTIONS = dict(today=ev.EVAL_TODAY, embeddings=False, rerank=False)


def git(vault: Vault, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(vault.root), *args], capture_output=True, text=True, check=True
    ).stdout


def replayed(vault: Vault, **kw) -> object:
    return run_lint(vault, FakeLLM(decider=Replay()), LintOptions(full=True, **OPTIONS, **kw))


def test_run_writes_nothing(dirty: tuple[Vault, ev.Gold]) -> None:
    vault, _ = dirty
    head = gitops.head(vault.root)
    replayed(vault)
    assert gitops.head(vault.root) == head and git(vault, "status", "--porcelain") == ""
    assert not state_path(vault).exists()


def test_apply_queues_one_review_per_finding_and_asks_once(dirty: tuple[Vault, ev.Gold]) -> None:
    vault, _ = dirty
    report = replayed(vault)
    apply_lint(vault, report, now=ev.EVAL_NOW)
    with_ops = [f for f in report.findings if f.ops]
    reviews = pending_reviews(Vault(vault.root))
    assert len(reviews) == len(with_ops) and {r.origin for r in reviews} == {"lint"}
    assert sorted(r.rationale for r in reviews) == sorted(f.summary for f in with_ops)
    asked = [q for q in read_questions(vault) if q.origin == "lint"]
    assert [q.text for q in asked] == ["Where do you live now: Porto or Braga?"]
    assert git(vault, "status", "--porcelain") == "" and last_rev(vault) == gitops.head(vault.root)
    # a second full run proposes the same repairs but adds no question twice
    again = replayed(vault)
    apply_lint(vault, again, now=ev.EVAL_NOW)
    assert len([q for q in read_questions(vault) if q.origin == "lint"]) == 1


def test_incremental_run_checks_what_changed(dirty: tuple[Vault, ev.Gold]) -> None:
    vault, _ = dirty
    apply_lint(vault, replayed(vault), now=ev.EVAL_NOW)
    quiet = run_lint(vault, FakeLLM(decider=Replay()), LintOptions(**OPTIONS))
    assert quiet.since == last_rev(vault) and quiet.scope == 0 and quiet.findings == []
    # a hand edit: a new page; the next run looks at it and its neighbours only
    path = vault.page_path("orgs/fabrikam")
    path.write_text(
        "---\nid: orgs/fabrikam\ntype: org\nsubject: orgs/fabrikam\nprivacy: normal\n"
        "created: 2026-10-27\nupdated: 2026-10-27\n---\n# Fabrikam\n\n## Facts\n",
        encoding="utf-8",
    )
    script_asked: list[str] = []

    def decider(text, questions):
        script_asked.append(text)
        return []  # no answer

    report = run_lint(vault, FakeLLM(decider=decider), LintOptions(**OPTIONS))
    assert report.scope == 1
    assert script_asked and all("Fabrikam" in t for t in script_asked)


def test_since_an_unknown_revision_is_an_error(clean_vault: Vault) -> None:
    with pytest.raises(ValueError, match="--since"):
        run_lint(clean_vault, None, LintOptions(since="no-such-rev", **OPTIONS))
    with pytest.raises(ValueError, match="unknown check"):
        run_lint(clean_vault, None, LintOptions(only=["spelling"], **OPTIONS))


def test_hand_edits_are_committed_apart(dirty: tuple[Vault, ev.Gold]) -> None:
    vault, _ = dirty
    notes = vault.page_path("projects/kiln")
    notes.write_text(notes.read_text() + "\n## Notes\n\nTry the new runner.\n", encoding="utf-8")
    apply_lint(vault, replayed(vault), now=ev.EVAL_NOW)
    log = git(vault, "log", "--format=%s")
    assert "brain: hand edits (committed apart by lint)" in log
    first_lint = git(vault, "log", "--format=%s", "-n", "1", "--", "wiki/projects/kiln.md")
    assert first_lint.strip() == "brain: hand edits (committed apart by lint)"


# --- CLI ---------------------------------------------------------------------------------------


@pytest.fixture
def env(dirty: tuple[Vault, ev.Gold], monkeypatch: pytest.MonkeyPatch):
    vault, _ = dirty
    settings = Settings(vault=vault.root)
    llm = FakeLLM(decider=Replay())
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "make_llm", lambda _s: llm)
    monkeypatch.setattr(cli, "_refresh", lambda *a: None)
    (vault.root / "SCHEMA.md").touch()
    return vault


def test_cli_dry_run_and_apply(env: Vault) -> None:
    runner = CliRunner()
    # the fake model embeds and reranks too: replay only covers the recorded questions, so
    # unrecorded ones are "no decision" notes, never changes
    out = runner.invoke(cli.app, ["lint", "--dry-run", "--all", "--only", "consistency"])
    assert out.exit_code == 0, out.output
    assert "dry run: nothing written" in out.output and "names the page" in out.output
    assert not state_path(env).exists()
    out = runner.invoke(cli.app, ["lint", "--all", "--only", "consistency"])
    assert out.exit_code == 0, out.output
    assert "2 review(s) queued" in out.output and "question(s) added" in out.output
    bad = runner.invoke(cli.app, ["lint", "--only", "spelling"])
    assert bad.exit_code == 1 and "unknown check" in bad.output


def test_cli_without_a_model_runs_code_checks(env: Vault, monkeypatch) -> None:
    monkeypatch.setattr(cli, "make_llm", lambda _s: FakeLLM(healthy=False))
    out = CliRunner().invoke(cli.app, ["lint", "--dry-run", "--all"])
    assert out.exit_code == 0, out.output
    assert "only the code checks run" in out.output


def test_eval_gold_vaults_are_well_formed(tmp_path: Path, git_env: None) -> None:
    for seed in (1, 2, 3):
        vault = ev.clean_vault(tmp_path / f"v{seed}")
        gold = ev.inject(vault, seed)
        v = Vault(vault.root)
        assert v.warnings() == []
        ids = [f.id for p in v.pages() for f in (*p.facts, *p.history)]
        assert len(ids) == len(set(ids))
        assert len(gold.items) == 12 and gold.asks
        for page_id in (i for pair in gold.merges.values() for i in pair):
            assert v.page(page_id), page_id
