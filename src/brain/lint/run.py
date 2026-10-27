"""``brain lint``: run the graph-cleaning checks and hand their proposals to the writer.

``run_lint`` reads the vault and returns a ``LintReport`` (findings with operations, questions or
report lines; nothing is written). ``apply_lint`` then

1. commits uncommitted hand edits on their own ("brain: hand edits"), so they never ride along
   in a brain commit;
2. adds each finding's question to ``questions.md`` (origin ``lint``) unless the same doubt is
   already there, open or answered;
3. applies each finding's operations as its own change set (origin ``lint``) under the normal
   review policy, so the owner approves or rejects every repair on its own (every lint
   operation is reviewed: merges, demotions, retypes, re-filed facts, inferred edges);
4. records the vault revision in ``.brain/lint.json`` (derived): the next run checks only the
   pages changed since, and their neighbours.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from brain.lint.common import Context
from brain.lint.consistency import check_consistency
from brain.lint.duplicates import check_duplicates
from brain.lint.edges import check_edges
from brain.lint.graph import Graph
from brain.lint.kinds import check_kinds
from brain.lint.report import CHECKS, Ask, Counting, Finding, LintReport, Timer
from brain.lint.subjects import check_subjects
from brain.vault import gitops
from brain.vault.models import SELF_ID
from brain.vault.store import Vault
from brain.vault.writer import apply

log = logging.getLogger(__name__)

STATE = "lint.json"
RUNNERS: dict[str, Callable[[Context], list[Finding]]] = {
    "duplicates": check_duplicates,
    "types": check_kinds,
    "subjects": check_subjects,
    "edges": check_edges,
    "consistency": check_consistency,
}


@dataclass
class LintOptions:
    only: Sequence[str] = ()  # checks to run (default: all of CHECKS)
    since: str | None = None  # git revision: check pages changed since (default: the last lint)
    full: bool = False  # the whole vault, whatever the last lint
    embeddings: bool = True
    rerank: bool = True
    today: date | None = None


def state_path(vault: Vault) -> Path:
    return vault.brain_dir / STATE


def last_rev(vault: Vault) -> str | None:
    try:
        return json.loads(state_path(vault).read_text(encoding="utf-8")).get("rev")
    except (OSError, ValueError):
        return None


def save_state(vault: Vault, rev: str | None, now: datetime) -> None:
    path = state_path(vault)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"rev": rev, "at": now.replace(microsecond=0).isoformat()}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def changed_entities(vault: Vault, rev: str) -> set[str]:
    """Entity ids whose page changed since ``rev`` (``self`` for any self page)."""
    out = set()
    for path in gitops.changed_since(vault.root, rev, "wiki"):
        if not path.endswith(".md") or not path.startswith("wiki/"):
            continue
        page_id = path.removeprefix("wiki/").removesuffix(".md")
        out.add(SELF_ID if page_id.startswith("self/") else page_id)
    return out


def run_lint(vault: Vault, llm: Any, options: LintOptions | None = None) -> LintReport:
    """Run the checks; nothing is written. ``llm`` may be None: only the code checks
    (``consistency``) run."""
    options = options or LintOptions()
    unknown = set(options.only) - set(CHECKS)
    if unknown:
        raise ValueError(f"unknown check(s): {', '.join(sorted(unknown))}")
    start = time.perf_counter()
    report = LintReport()
    graph = Graph(vault, options.today or date.today())
    report.pages = len(graph.entities)
    since = options.since if options.since else None if options.full else last_rev(vault)
    scope = None
    if since:
        try:
            changed = changed_entities(vault, since)
        except gitops.GitError as err:
            if options.since:
                raise ValueError(f"--since {since}: {err}") from err
            report.notes.append(f"last lint at {since[:12]} is unknown to git: whole vault")
            since = None
        else:
            scope = set(changed)
            for entity in changed:
                scope |= graph.neighbours(entity)
            report.since = since
    report.scope = len(graph.entities) if scope is None else len(scope - {SELF_ID})
    model = Counting(llm, report.costs) if llm is not None else None
    ctx = Context(graph, model, report, scope, options.embeddings, options.rerank)
    for name in CHECKS:
        if options.only and name not in options.only:
            continue
        if model is None and name != "consistency":
            continue  # the other checks need the model's decisions
        with Timer(report.costs, name):
            report.findings += RUNNERS[name](ctx)
    report.seconds = time.perf_counter() - start
    return report


def _question_key(reason: str) -> str:
    return reason.partition(":")[0].strip()


def known_question(vault: Vault, ask: Ask) -> bool:
    """The same doubt is already in questions.md, open or answered (never asked twice)."""
    from brain.interview.questions import read_questions

    key = _question_key(ask.reason)
    text = " ".join(ask.text.casefold().split())
    for item in read_questions(vault):
        if item.origin == "lint" and _question_key(item.reason) == key:
            return True
        if item.open and " ".join(item.text.casefold().split()) == text:
            return True
    return False


def apply_lint(vault: Vault, report: LintReport, *, now: datetime | None = None) -> LintReport:
    """Write the report: questions, one reviewed change set per finding, the lint state."""
    from brain.interview.questions import add_question

    now = now or datetime.now()
    if gitops.is_dirty(vault.root):
        gitops.commit_all(vault.root, "brain: hand edits (committed apart by lint)")
        report.notes.append("committed uncommitted hand edits apart")
    for finding in report.findings:
        ask = finding.ask
        if ask is None or known_question(vault, ask):
            continue
        item = add_question(
            vault,
            ask.text,
            region=ask.region,
            reason=ask.reason,
            priority=ask.priority,
            origin="lint",
            about=ask.about,
            private=ask.private or None,
            doubt=finding.key,
        )
        report.questions.append(item.id)
    for finding in report.findings:
        if finding.ops:
            changeset = finding.changeset().model_copy(update={"created": now})
            report.applied.append(apply(changeset, vault, now=now))
    if gitops.is_dirty(vault.root):
        gitops.commit_all(vault.root, f"brain: lint questions ({len(report.questions)})")
    save_state(vault, gitops.head(vault.root), now)
    return report
