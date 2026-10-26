"""Merge facts that say the same thing in other words.

A document restates a duty ("Fine-tunes graph neural networks for solubility prediction" in the
intro, "Responsible for fine-tuning graph neural networks to predict solubility" under the job),
a skill ("deep-learning" / "Deep learning") or an interest, and the vault already holds facts
from earlier documents. ``normalise`` merges facts whose objects are equal; this step merges
paraphrases of multi-valued literal predicates (``PREDICATES``) with the same subject and
predicate, within the change set and against the vault's current facts:

1. code: equal words after folding case, accents and punctuation → the same (no model);
2. embeddings (``llm.embed``, Qwen3-Embedding, an instruction on both sides): cosine ≥
   ``MERGE_ABOVE`` → the same;
3. ``ASK_ABOVE`` ≤ cosine < ``MERGE_ABOVE``: a typed decision ("the same thing in other words,
   or different things?"); p(same) ≥ ``SAME_P`` → the same.

Thresholds were calibrated on labelled fictional pairs (``tests/fixtures/dedup/pairs.yaml``,
docs/decisions.md): a wrong merge loses a fact, so they are set well above the closest
different pair. Two dated facts whose periods leave a gap (the same duty at two jobs) are never
merged.

The more specific statement is kept (more words; then the earlier start), with the dates of
both (``normalise._merge_start``; no end if either is still open; an undated one takes the
other's). A new fact that restates a vault fact takes that fact's object and text, so the
writer's restatement rule applies (skipped, or a review if it brings new dates). Every merge is
a ``dedup:`` note. Without the embedding endpoint only step 1 runs, and a note says so.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import numpy as np
from openai import OpenAIError

from brain.ingest.normalise import _merge_start, _month, template_text
from brain.llm.base import StructuredOutputError
from brain.llm.decide import Choice, ChoiceQuestion, Decision, Question
from brain.vault.models import SELF_ID, is_multi_valued, link_target
from brain.vault.ops import AddFact, ChangeSet
from brain.vault.store import Vault

log = logging.getLogger(__name__)

MERGE_ABOVE = 0.90  # cosine: the same without asking (closest different pair seen: 0.85)
ASK_ABOVE = 0.75  # cosine: ask the model below MERGE_ABOVE, down to this
SAME_P = 0.8  # decision: p(same) to merge (different pairs in the asked band: ≤ 0.11)
ASK_TOP = 2  # at most this many candidates asked about per fact
INSTRUCT = "Instruct: Retrieve a statement about a person that says the same thing\nQuery: "
ERRORS = (OpenAIError, httpx.HTTPError, OSError, ValueError, StructuredOutputError)
LABEL = {
    "responsibility": "duties in a job",
    "has_skill": "skills",
    "interest": "interests",
    "habit": "habits",
    "likes": "things the person likes",
    "dislikes": "things the person dislikes",
    "goal": "goals",
    "values": "values",
    "note": "notes",
    "certification": "certifications",
}
PREDICATES = frozenset(LABEL)  # multi-valued literal predicates whose paraphrases merge
STOP = frozenset(["a", "an", "the", "of", "in", "at", "for", "and", "to", "with", "on", "by"])


class DedupModel(Protocol):
    def embed(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray: ...

    def decide(self, text: str, questions: Sequence[Question]) -> list[Decision]: ...


@dataclass
class DedupReport:
    merged: int = 0
    embedded: int = 0
    asked: int = 0
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class _Item:
    """A fact of one (subject, predicate) group: a vault fact (fixed) or a new one."""

    p: str
    o: str
    text: str
    start: str | None
    end: str | None
    op: AddFact | None = None  # None: already in the vault
    vector: np.ndarray | None = None

    @property
    def statement(self) -> str:
        """What the fact says: its own text, or its value when the text is the template."""
        templates = (template_text(self.p, self.o, past=past) for past in (False, True))
        return self.o if not self.text or self.text in templates else self.text


def words(text: str) -> tuple[str, ...]:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().casefold()
    return tuple(w for w in re.findall(r"[a-z0-9+#]+", folded) if w not in STOP)


def _dated(item: _Item) -> bool:
    return bool(item.start or item.end)


def _apart(a: _Item, b: _Item) -> bool:
    """Both dated, with more than a month between their periods: two stints, not one."""

    def gap(first: _Item, second: _Item) -> bool:
        end, start = _month(first.end, end=True), _month(second.start, end=False)
        return end is not None and start is not None and start - end > 1

    return _dated(a) and _dated(b) and (gap(a, b) or gap(b, a))


def question(p: str) -> ChoiceQuestion:
    return ChoiceQuestion(
        name="same",
        instructions=(
            f"Statements A and B are two {LABEL.get(p, 'facts')} listed for the same person. "
            "How do they relate?"
        ),
        choices=[
            Choice(
                value="same",
                description="the same thing in other words, or one is a more detailed version "
                "of the other",
            ),
            Choice(
                value="different", description="different things, even if related or worded alike"
            ),
        ],
    )


def _merge_into(keep: _Item, other: _Item) -> None:
    """``keep`` (a new fact) absorbs ``other``: the more specific statement, both periods."""
    if len(words(other.statement)) > len(words(keep.statement)) or (
        len(words(other.statement)) == len(words(keep.statement))
        and (other.start or "9999") < (keep.start or "9999")
    ):
        keep.o, keep.text, keep.vector = other.o, other.text, other.vector
    if not _dated(keep):
        keep.start, keep.end = other.start, other.end
    elif _dated(other):
        keep.start = _merge_start(keep.start, other.start)
        keep.end = None if None in (keep.end, other.end) else max(keep.end or "", other.end or "")


def merge_paraphrases(
    changeset: ChangeSet, vault: Vault, llm: Any, *, ask: bool = True
) -> tuple[ChangeSet, DedupReport]:
    """The change set without new facts that restate another one (see the module doc)."""
    started = time.perf_counter()
    report = DedupReport()
    groups: dict[tuple[str, str], list[_Item]] = {}
    for op in changeset.ops:
        if (
            isinstance(op, AddFact)
            and op.p in PREDICATES
            and is_multi_valued(op.p, op.subject)
            and not link_target(op.o)
        ):
            item = _Item(op.p, op.o, op.text, op.valid_from, op.valid_to, op)
            groups.setdefault((op.subject, op.p), []).append(item)
    if not groups:
        return changeset, report
    existing: dict[tuple[str, str], list[_Item]] = {}
    for page in vault.pages():
        for f in page.facts:
            key = (page.subject if page.subject != SELF_ID else SELF_ID, f.p)
            if key in groups and not link_target(f.o):
                old = _Item(f.p, f.o, f.text, f.valid_from, f.valid_to)
                existing.setdefault(key, []).append(old)
    candidates = [k for k, items in groups.items() if len(items) + len(existing.get(k, [])) > 1]

    vectors_ok = _embed(
        llm, [i for k in candidates for i in (*existing.get(k, []), *groups[k])], report
    )
    dropped: set[int] = set()
    for key in candidates:
        kept: list[_Item] = list(existing.get(key, []))
        for item in groups[key]:
            match, how = _find(llm, key[1], item, kept, vectors_ok and ask, vectors_ok, report)
            if match is None:
                kept.append(item)
                continue
            report.merged += 1
            assert item.op is not None
            if match.op is None:  # a vault fact: restate it, and the writer's rule applies
                note = f"{item.statement!r} restates the vault's {match.statement!r} ({how})"
                report.notes.append(f"dedup: {key[1]} {note}")
                item.op.o, item.op.text = match.o, match.text
                continue
            dropped.add(id(item.op))
            before = match.statement
            _merge_into(match, item)
            report.notes.append(
                f"dedup: merged {key[1]} {item.statement!r} into {before!r} ({how})"
                + (f", kept {match.statement!r}" if match.statement != before else "")
            )
        for item in kept:
            if item.op is not None:
                item.op.o, item.op.text = item.o, item.text
                item.op.valid_from, item.op.valid_to = item.start, item.end
    ops = [op for op in changeset.ops if id(op) not in dropped]
    report.seconds = time.perf_counter() - started
    return changeset.model_copy(update={"ops": ops}), report


def _embed(llm: Any, items: list[_Item], report: DedupReport) -> bool:
    if not items:
        return False
    try:
        vectors = llm.embed([INSTRUCT + i.statement for i in items])
    except ERRORS as err:
        log.warning("dedup: embeddings failed: %s", err)
        report.notes.append(f"dedup: no embeddings ({type(err).__name__}): paraphrases not merged")
        return False
    for item, vector in zip(items, vectors, strict=True):
        item.vector = vector
    report.embedded = len(items)
    return True


def _find(
    llm: Any,
    p: str,
    item: _Item,
    kept: list[_Item],
    ask: bool,
    vectors: bool,
    report: DedupReport,
) -> tuple[_Item | None, str]:
    """The kept fact ``item`` restates, and how that was decided."""
    options = [k for k in kept if not _apart(k, item)]
    for k in options:
        if words(k.statement) == words(item.statement):
            return k, "same words"
    if not vectors or item.vector is None:
        return None, ""
    scored = sorted(
        (
            (float(k.vector @ item.vector), n, k)
            for n, k in enumerate(options)
            if k.vector is not None
        ),
        key=lambda t: (-t[0], t[1]),
    )
    if scored and scored[0][0] >= MERGE_ABOVE:
        return scored[0][2], f"cosine {scored[0][0]:.2f}"
    if not ask:
        return None, ""
    for cos, _, k in [s for s in scored if s[0] >= ASK_ABOVE][:ASK_TOP]:
        try:
            decision = llm.decide(f"A: {k.statement}\nB: {item.statement}", [question(p)])[0]
        except ERRORS as err:
            report.notes.append(f"dedup: no decision ({type(err).__name__}): not merged")
            return None, ""
        report.asked += 1
        same = None if decision.refusal else decision.probabilities.get("same")
        if same is not None and same >= SAME_P:
            return k, f"cosine {cos:.2f}, p(same)={same:.2f}"
    return None, ""
