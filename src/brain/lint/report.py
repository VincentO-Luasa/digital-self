"""What a lint run found: findings (each with its operations, its question for the owner, or
just a report line), costs, and the combined change set."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from brain.llm.decide import Decision, Question
from brain.vault.ops import ChangeSet, Operation
from brain.vault.writer import ApplyResult

CHECKS = ("duplicates", "types", "subjects", "edges", "consistency")


@dataclass(frozen=True)
class Ask:
    """A question for a later interview (``questions.md``, origin ``lint``)."""

    text: str
    reason: str  # starts with the finding's key, so the same doubt is never asked twice
    about: str | None = None
    region: str | None = None
    priority: int = 3
    private: bool = False


@dataclass
class Finding:
    check: str  # one of CHECKS
    key: str  # stable: "<kind> <ids>", e.g. "merge orgs/a orgs/b"
    summary: str
    about: list[str] = field(default_factory=list)  # entity or fact ids
    ops: list[Operation] = field(default_factory=list)
    ask: Ask | None = None
    p: float | None = None  # the decision's probability for its choice
    decision: str = ""  # Decision.summary()
    src: list[str] = field(default_factory=list)  # provenance for the change set

    @property
    def action(self) -> str:
        if self.ops:
            return ", ".join(dict.fromkeys(op.op for op in self.ops))
        return "question" if self.ask else "report"

    def changeset(self) -> ChangeSet:
        return ChangeSet(ops=self.ops, origin="lint", src=self.src, rationale=self.summary)


@dataclass
class Costs:
    decisions: int = 0
    embedded: int = 0
    reranked: int = 0
    seconds: dict[str, float] = field(default_factory=dict)


@dataclass
class LintReport:
    pages: int = 0  # entity pages in the vault
    scope: int = 0  # pages checked (incremental runs check fewer)
    since: str | None = None
    findings: list[Finding] = field(default_factory=list)
    costs: Costs = field(default_factory=Costs)
    notes: list[str] = field(default_factory=list)
    applied: list[ApplyResult] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)  # ids of questions added
    seconds: float = 0.0

    def changeset(self) -> ChangeSet:
        """Every proposed operation as one change set."""
        ops = [op for f in self.findings for op in f.ops]
        src = list(dict.fromkeys(s for f in self.findings if f.ops for s in f.src))
        return ChangeSet(ops=ops, origin="lint", src=src, rationale="brain lint")

    def by_check(self, check: str) -> list[Finding]:
        return [f for f in self.findings if f.check == check]


class Counting:
    """Wraps a model: counts decisions, embedded texts and rerank calls; errors pass through."""

    def __init__(self, llm: Any, costs: Costs) -> None:
        self.llm = llm
        self.costs = costs

    def decide(self, text: str, questions: Sequence[Question]) -> list[Decision]:
        self.costs.decisions += len(questions)
        return self.llm.decide(text, questions)

    def embed(self, texts: Sequence[str], *, query: bool = False) -> Any:
        self.costs.embedded += len(texts)
        return self.llm.embed(texts, query=query)

    def rerank(self, query: str, documents: Sequence[str], top_n: int | None = None) -> Any:
        self.costs.reranked += 1
        return self.llm.rerank(query, documents, top_n=top_n)


class Timer:
    def __init__(self, costs: Costs, name: str) -> None:
        self.costs, self.name = costs, name

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        spent = time.perf_counter() - self.start
        self.costs.seconds[self.name] = self.costs.seconds.get(self.name, 0.0) + spent
