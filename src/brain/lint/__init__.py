"""Graph cleaning (``brain lint``): duplicate entities, doubtful page types, facts on the wrong
subject, missing structural edges and consistency checks. Proposes reviewed operations and
interview questions; the writer applies them (``brain.lint.run``)."""

from brain.lint.report import CHECKS, Ask, Finding, LintReport
from brain.lint.run import LintOptions, apply_lint, run_lint

__all__ = ["CHECKS", "Ask", "Finding", "LintOptions", "LintReport", "apply_lint", "run_lint"]
