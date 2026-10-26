"""Shared plumbing of the lint checks: the run context and one-question typed decisions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import OpenAIError

from brain.lint.graph import Graph
from brain.lint.report import LintReport
from brain.llm.base import StructuredOutputError
from brain.llm.decide import ChoiceQuestion, Decision
from brain.vault.models import SELF_ID

log = logging.getLogger(__name__)

ERRORS = (OpenAIError, httpx.HTTPError, OSError, ValueError, StructuredOutputError, LookupError)
ASK_P = 0.5  # below this the model's best answer is no answer: nothing is done or asked


@dataclass(frozen=True)
class Verdict:
    choice: str
    p: float
    decision: Decision

    @property
    def measured(self) -> bool:
        """False for a structured fallback (a label without a probability): never acted on."""
        return not self.decision.fallback

    def acts(self, threshold: float) -> bool:
        return self.measured and self.p >= threshold

    def p_of(self, *values: str) -> float:
        """The probability mass of some options together."""
        return sum(self.decision.probabilities.get(v, 0.0) for v in values)

    def summary(self) -> str:
        tag = ", fallback" if self.decision.fallback else ""
        return f"{self.decision.name}={self.choice} (p {self.p:.2f}{tag})"


@dataclass
class Context:
    graph: Graph
    llm: Any  # brain.lint.report.Counting around the model, or None (code-only checks)
    report: LintReport
    scope: set[str] | None = None  # entity ids to check; None: the whole vault
    embeddings: bool = True
    rerank: bool = True
    busy: set[str] = field(default_factory=set)  # pages an op of this run removes or renames
    kept: set[str] = field(default_factory=set)  # pages a merge of this run keeps

    def in_scope(self, *entities: str) -> bool:
        return self.scope is None or any(e in self.scope for e in entities)

    def free(self, *entities: str) -> bool:
        return not any(e in self.busy for e in entities)

    def note(self, text: str) -> None:
        self.report.notes.append(text)

    def private(self, *entities: str) -> bool:
        lifemap = self.graph.vault.lifemap
        for entity in entities:
            page = self.graph.pages.get(entity)
            if page and (
                page.privacy == "private" or lifemap.privacy_for(page.regions) == "private"
            ):
                return True
        return False

    def region(self, entity: str, default: str | None = None) -> str | None:
        page = self.graph.pages.get(entity)
        if page and page.regions and entity != SELF_ID:
            return page.regions[0]
        return default

    def decide(self, text: str, question: ChoiceQuestion, what: str) -> Verdict | None:
        """One choice question; None when the model is unavailable or gives no answer."""
        if self.llm is None:
            return None
        try:
            [decision] = self.llm.decide(text, [question])
        except ERRORS as err:
            self.note(f"{what}: no decision ({type(err).__name__}: {err})")
            return None
        if decision.refusal or decision.choice is None:
            self.note(f"{what}: no answer")
            return None
        p = decision.probabilities.get(decision.choice, decision.confidence)
        return Verdict(decision.choice, p, decision)
