"""The dirty-graph eval's scorer: gold items, what counts as a fix, a wrong change, a question.
No model: the lint report is built by hand. Fictional data only."""

from __future__ import annotations

from pathlib import Path

from brain.eval import lint_eval as ev
from brain.lint import Ask, Finding, LintReport, apply_lint
from brain.vault.ops import AddFact, DemoteEntity, MergeEntities, RefileFact


def _report(*findings: Finding) -> LintReport:
    return LintReport(findings=list(findings))


def test_every_gold_fix_scores_and_extras_are_wrong(tmp_path: Path, git_env: None) -> None:
    vault = ev.clean_vault(tmp_path / "vault")
    gold = ev.inject(vault, 2)
    findings = []
    for name, pair in gold.merges.items():
        keep, gone = sorted(pair)
        findings.append(
            Finding("duplicates", name, name, ops=[MergeEntities(keep=keep, merge=gone)])
        )
    for name, page in gold.demotes.items():
        findings.append(Finding("types", name, name, ops=[DemoteEntity(id=page)]))
    for name, (fid, what, value) in gold.refiles.items():
        link = value if what == "subject" else f"[[{value}]]"
        findings.append(
            Finding("subjects", name, name, ops=[RefileFact(fact_id=fid, **{what: link})])
        )
    for name, (s, p, o) in gold.edges.items():
        obj = f"[[{o}]]" if "/" in o else o.title()
        op = AddFact(
            subject=s,
            p=p,
            o=obj,
            text=f"{p} {o}",
            kind="inferred",
            evidence=["ep-20261010-0900-history"],
        )
        findings.append(Finding("edges", name, name, ops=[op], src=["ep-20261010-0900-history"]))
    # a change no noise asked for: wrong, even if it applies cleanly
    findings.append(
        Finding(
            "edges",
            "kiln",
            "kiln",
            ops=[
                AddFact(
                    subject="projects/kiln",
                    p="part_of",
                    o="[[orgs/northwind-labs]]",
                    text="x",
                    kind="inferred",
                    evidence=["ep-20261012-1000-kiln"],
                )
            ],
            src=["ep-20261012-1000-kiln"],
        )
    )
    city = gold.variants["two-cities"]
    findings.append(
        Finding(
            "consistency",
            "two",
            "two",
            ask=Ask(
                text=f"Where do you live now: Porto or {city}?", reason="lint two lives_in self"
            ),
        )
    )
    report = _report(*findings)
    apply_lint(vault, report, now=ev.EVAL_NOW)
    score = ev.score("hand", vault, report, gold, 1.0)
    assert sorted(score.fixed) == sorted(gold.items), (score.missed, score.proposed_not_verified)
    assert len(score.wrong) == 1 and "projects/kiln" in score.wrong[0]
    assert score.asks_matched == ["two-cities"]
    data = ev.report_json([score], {"model": "none"})
    assert data["overall"]["fixed"] == 12 and data["overall"]["wrong"] == 1


def test_a_proposed_fix_that_does_not_hold_is_not_counted(tmp_path: Path, git_env: None) -> None:
    vault = ev.clean_vault(tmp_path / "vault")
    gold = ev.inject(vault, 1)
    keep, gone = sorted(gold.merges["dup-org-typo"])
    report = _report(Finding("duplicates", "m", "m", ops=[MergeEntities(keep=keep, merge=gone)]))
    apply_lint(vault, report, now=ev.EVAL_NOW)
    report.applied = []  # the owner never approves it
    score = ev.score("pending", vault, report, gold, 1.0)
    assert score.fixed == [] and score.proposed_not_verified == ["dup-org-typo"]
