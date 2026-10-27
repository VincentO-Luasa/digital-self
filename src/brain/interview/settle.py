"""Settle lint doubts from the owner's interview answers (docs/architecture.md, Flow 2).

For each plan step that asked lint doubts (``PlanItem.doubts``) and got an answer that is not
private, one typed decision maps the owner's words onto the doubt's option set, the "does not
say" option first (as lint orders its escape options):

- ``type``: unclear / project / topic / tool / activity / organisation / person / place / event;
- ``part_of``: unclear / on their own / each of the owner's employers;
- ``merge``: unclear / the same / two different ones;
- ``job`` (a duty's employer): unclear / each employer.

What happens (``APPLY_P`` = 0.9, ``ASK_P`` = 0.5; docs/decisions.md "Interview settles lint
doubts"):

- the owner's answer at p ≥ ``APPLY_P`` (a measured probability) → the operation is applied: the
  owner said it, like an owner correction. ``part_of`` is a ``stated`` fact sourced to the
  transcript; a value kind demotes the page (after re-filing the facts that used it as a page,
  as lint does), another type retypes it, "the same" merges, a duty's job re-files ``at``.
  The owner confirming the current filing ("a project", "two different ones", "on my own")
  settles the doubt with no change.
- ``ASK_P`` ≤ p < ``APPLY_P``, or a fallback decision (no probability) → the same operation is
  queued as a review (``part_of`` then ``inferred`` with the transcript and the owner's words as
  evidence).
- "does not say", or p < ``ASK_P`` → nothing; the questions.md item stays open.

A ``part_of`` answer is used only when the page stays a project (a type doubt on the same page
that is not answered "project" leaves it open). Nothing is settled from a private answer.
Every change is one change set (origin ``user``, source = the transcript), so ``log.md`` and git
record it and it can be reverted.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

from brain.interview.doubts import Doubt, EntityInfo
from brain.lint.common import ERRORS, Context
from brain.lint.graph import TYPE_WORDS, Graph
from brain.lint.kinds import VALUE_KINDS, refiles
from brain.lint.names import fold
from brain.lint.report import LintReport
from brain.llm.decide import Choice, ChoiceQuestion, Decision
from brain.prompts import load_prompt
from brain.vault.models import Page, link_target
from brain.vault.ops import (
    AddFact,
    ChangeSet,
    DemoteEntity,
    MergeEntities,
    Operation,
    RefileFact,
    RetypeEntity,
)
from brain.vault.store import Vault
from brain.vault.writer import apply, log_event

log = logging.getLogger(__name__)

APPLY_P = 0.9
ASK_P = 0.5
UNCLEAR = "unclear"
OWN = "on their own"
SAME, DIFFERENT = "same", "different"
QUOTE_WORDS = 40

Action = Literal["applied", "queued", "kept", "open", "refused"]


@dataclass
class Settlement:
    doubt: str  # the lint key
    action: Action
    detail: str = ""
    choice: str | None = None
    p: float = 0.0
    ops: list[Operation] = field(default_factory=list)

    @property
    def settled(self) -> bool:
        """The doubt is answered: its questions.md item can be ticked."""
        return self.action in ("applied", "queued", "kept")


# --- questions ---------------------------------------------------------------------------------


# what the owner says a thing is (lint's KIND describes things; here the owner's words decide:
# with lint's descriptions "stream processing", which the owner called a project, read as a topic)
OWNER_KINDS = {
    "project": "something they say was a project of theirs, one piece of work they built, led "
    "or worked on",
    "topic": "a field, subject or technique they work in, not one project",
    "tool": "a tool, product, language or technology they use",
    "activity": "a hobby, sport or pastime",
    "org": "an organisation: company, university, school, club or community",
    "person": "a person",
    "place": "a city, region, country or address",
    "event": "something that happened at a time: a conference, a trip, a wedding",
}


def kind_question(title: str) -> ChoiceQuestion:
    return ChoiceQuestion(
        name="kind",
        instructions=f"In this conversation, what does the owner say {title} is to them?",
        choices=[
            Choice(value=UNCLEAR, description="the owner does not say"),
            *(Choice(value=k, description=d) for k, d in OWNER_KINDS.items()),
        ],
    )


def part_of_question(title: str, jobs: Sequence[str]) -> ChoiceQuestion:
    return ChoiceQuestion(
        name="part_of",
        instructions=f"According to the owner, where did they do {title}?",
        choices=[
            Choice(value=UNCLEAR, description="the owner does not say"),
            Choice(value=OWN, description="on their own, outside any job"),
            *(Choice(value=j, description=f"as part of their work at {j}") for j in jobs),
        ],
    )


def same_question(a: str, b: str, word: str) -> ChoiceQuestion:
    return ChoiceQuestion(
        name="same",
        instructions=f"According to the owner, are {a} and {b} the same {word}?",
        choices=[
            Choice(value=UNCLEAR, description="the owner does not say"),
            Choice(value=SAME, description=f"yes, one {word} with two names"),
            Choice(value=DIFFERENT, description=f"no, two different {word}s"),
        ],
    )


def job_question(duty: str, jobs: Sequence[str]) -> ChoiceQuestion:
    return ChoiceQuestion(
        name="job",
        instructions=f"According to the owner, at which job did they have this duty: {duty}?",
        choices=[
            Choice(value=UNCLEAR, description="the owner does not say"),
            *(Choice(value=j, description=f"at {j}") for j in jobs),
        ],
    )


# --- one plan step -----------------------------------------------------------------------------


@dataclass
class _Verdict:
    choice: str
    p: float
    measured: bool
    probabilities: dict[str, float]

    def mass(self, *values: str) -> float:
        return sum(self.probabilities.get(v, 0.0) for v in values)


def _verdict(d: Decision) -> _Verdict | None:
    if d.refusal or d.choice is None:
        return None
    p = d.probabilities.get(d.choice, d.confidence)
    return _Verdict(d.choice, p, not d.fallback, dict(d.probabilities))


def render_exchanges(exchanges: Sequence[tuple[str, str]]) -> str:
    return "\n".join(f"Interviewer: {q}\nOwner: {a}" for q, a in exchanges)


def _quote(exchanges: Sequence[tuple[str, str]]) -> str:
    words = " ".join(a for _, a in exchanges).replace('"', "'").split()
    text = " ".join(words[:QUOTE_WORDS])
    return text + ("…" if len(words) > QUOTE_WORDS else "")


class Settler:
    """Settles the doubts of one interview (``settle_step`` per plan step, then ``apply``)."""

    def __init__(
        self,
        vault: Vault,
        llm: Any,
        *,
        transcript: str,
        today: date,
        now: datetime,
        infos: Mapping[str, EntityInfo] | None = None,
    ) -> None:
        self.vault = vault
        self.llm = llm
        self.transcript = transcript
        self.today = today
        self.now = now
        self.infos = dict(infos or {})

    def graph(self) -> Graph:
        return Graph(self.vault, self.today)

    def _jobs(self, g: Graph, page: Page | None) -> list[str]:
        info = self.infos.get(page.id) if page else None
        if info and info.jobs:
            return list(info.jobs)
        seen: dict[str, None] = {}
        for job in g.jobs:
            seen.setdefault(job.name, None)
        return list(seen)

    def settle_step(
        self, doubts: Sequence[str], exchanges: Sequence[tuple[str, str]], brief: Sequence[str]
    ) -> list[Settlement]:
        """Map the answers of one step onto its doubts and apply or queue what follows."""
        parsed = [d for d in (Doubt.parse(k) for k in doubts) if d and d.settles]
        if not parsed or not exchanges:
            return []
        g = self.graph()
        questions: list[ChoiceQuestion] = []
        for d in parsed:
            q = self._question(g, d)
            if q is not None and q.name not in {x.name for x in questions}:
                questions.append(q)
        if not questions:
            return [Settlement(d.key, "open", "gone from the vault") for d in parsed]
        text = load_prompt("interview_settle").format(
            brief="\n".join(f"- {b}" for b in brief) or "- (nothing more)",
            exchanges=render_exchanges(exchanges),
        )
        try:
            decisions = self.llm.decide(text, questions)
        except ERRORS as err:
            log.warning("settling %s failed: %s", ", ".join(doubts), err)
            return [
                Settlement(d.key, "open", f"no decision ({type(err).__name__})") for d in parsed
            ]
        verdicts = {q.name: _verdict(d) for q, d in zip(questions, decisions, strict=False)}
        quote = _quote(exchanges)
        outcomes = [self._outcome(g, d, verdicts, quote) for d in parsed]
        return _placed_is_a_project(outcomes)

    def _question(self, g: Graph, d: Doubt) -> ChoiceQuestion | None:
        page = g.pages.get(d.ids[0])
        if d.kind == "job":
            ref = g.find(d.ids[0])
            if ref is None:
                return None
            return job_question(f"“{g.readable(ref.fact.text)}”", self._jobs(g, None))
        if page is None:
            return None
        if d.kind == "type":
            return kind_question(page.title)
        if d.kind == "part_of":
            return part_of_question(page.title, self._jobs(g, page))
        if d.kind == "merge" and len(d.ids) > 1 and d.ids[1] in g.pages:
            word = TYPE_WORDS.get(page.type, page.type)
            return same_question(page.title, g.pages[d.ids[1]].title, word)
        return None

    # -- outcomes ---------------------------------------------------------------------------

    def _outcome(
        self, g: Graph, d: Doubt, verdicts: Mapping[str, _Verdict | None], quote: str
    ) -> Settlement:
        name = {"type": "kind", "part_of": "part_of", "merge": "same", "job": "job"}[d.kind]
        v = verdicts.get(name)
        if v is None:
            return Settlement(d.key, "open", "no answer")
        out = Settlement(d.key, "open", choice=v.choice, p=v.p)
        if v.choice == UNCLEAR:
            out.detail = "the owner does not say"
            return out
        if d.kind == "type":
            return self._type(g, d, v, out)
        if d.kind == "part_of":
            kind = verdicts.get("kind")
            if kind is not None and (kind.choice != "project" or kind.p < ASK_P):
                out.detail = "not a project, or not said: no part_of"
                return out
            return self._part_of(g, d, v, out, quote)
        if d.kind == "merge":
            return self._merge(g, d, v, out)
        return self._job(g, d, v, out)

    def _level(self, v: _Verdict, p: float | None = None) -> Action | None:
        p = v.p if p is None else p
        if p < ASK_P:
            return None
        return "applied" if v.measured and p >= APPLY_P else "queued"

    def _type(self, g: Graph, d: Doubt, v: _Verdict, out: Settlement) -> Settlement:
        page = g.pages.get(d.ids[0])
        if page is None:
            out.detail = "page gone"
            return out
        if v.choice == page.type:
            if v.p >= APPLY_P or not v.measured:
                out.action, out.detail = "kept", f"the owner confirms it is a {page.type}"
            return out
        p = v.mass(*VALUE_KINDS) if v.choice in VALUE_KINDS else v.p
        if (level := self._level(v, p)) is None:
            return out
        if v.choice in VALUE_KINDS:
            if g.facts_of(page.subject, history=True) or page.notes.strip():
                out.detail = f"the owner says {v.choice}, but the page has facts or notes"
                out.action = "kept"
                log_event(
                    self.vault,
                    "user",
                    [self.transcript],
                    f"owner says {page.id} is a {v.choice}; not demoted: it has facts or notes",
                    now=self.now,
                )
                return out
            ctx = Context(graph=g, llm=None, report=LintReport())
            out.ops = [*refiles(ctx, page, v.choice), DemoteEntity(id=page.id)]
        else:
            out.ops = [RetypeEntity(id=page.id, type=v.choice)]  # type: ignore[arg-type]
        return self._apply(out, level, f"the owner says {page.title} is a {v.choice}")

    def _part_of(self, g: Graph, d: Doubt, v: _Verdict, out: Settlement, quote: str) -> Settlement:
        page = g.pages.get(d.ids[0])
        if page is None or page.type != "project":
            out.detail = "not a project page any more"
            return out
        if any(r.fact.p == "part_of" for r in g.facts_of(page.subject)):
            out.action, out.detail = "kept", "already has part_of"
            return out
        level = self._level(v)
        if level is None:
            return out
        if v.choice == OWN:
            if level == "applied":
                out.action, out.detail = "kept", "the owner's own project: no part_of"
            return out
        job = next((j for j in g.jobs if j.name == v.choice), None)
        if job is None or job.org is None:
            if level == "applied":
                out.action, out.detail = "kept", f"at {v.choice}, which has no page"
            return out
        text = f"{page.title} is part of the owner's work at {job.name}"
        if level == "applied":
            fact = AddFact(
                subject=page.id, p="part_of", o=f"[[{job.org}]]", text=text, kind="stated"
            )
        else:
            fact = AddFact(
                subject=page.id,
                p="part_of",
                o=f"[[{job.org}]]",
                text=text,
                kind="inferred",
                evidence=[self.transcript, *([quote] if quote else [])],
            )
        out.ops = [fact]
        return self._apply(out, level, f"the owner says {page.title} was at {job.name}")

    def _merge(self, g: Graph, d: Doubt, v: _Verdict, out: Settlement) -> Settlement:
        a, b = (g.pages.get(i) for i in d.ids[:2])
        if a is None or b is None:
            out.detail = "a page is gone"
            return out
        level = self._level(v)
        if level is None:
            return out
        if v.choice == DIFFERENT:
            if level == "applied":
                out.action, out.detail = "kept", "the owner says they are different"
            return out
        keep, gone = sorted((a, b), key=lambda p: _weight(g, p), reverse=True)
        out.ops = [MergeEntities(keep=keep.id, merge=gone.id)]
        return self._apply(out, level, f"the owner says {a.title} and {b.title} are one")

    def _job(self, g: Graph, d: Doubt, v: _Verdict, out: Settlement) -> Settlement:
        ref = g.find(d.ids[0])
        job = next((j for j in g.jobs if j.name == v.choice and j.org), None)
        if ref is None or job is None:
            out.detail = "fact gone, or a job without a page"
            return out
        if link_target(ref.fact.extra.get("at", "")) == job.org:
            out.action, out.detail = "kept", "already filed there"
            return out
        level = self._level(v)
        if level is None:
            return out
        out.ops = [RefileFact(fact_id=ref.fact.id, at=f"[[{job.org}]]")]
        return self._apply(out, level, f"the owner says the duty was at {job.name}")

    def _apply(self, out: Settlement, level: Action, why: str) -> Settlement:
        cs = ChangeSet(
            ops=out.ops,
            origin="user",
            src=[self.transcript],
            rationale=f"interview: {why} (lint {out.doubt}, p {out.p:.2f})",
            created=self.now,
        )
        result = apply(cs, self.vault, review=level != "applied", now=self.now)
        if result.skipped and not (result.applied or result.queued):
            out.action = "refused"
            out.detail = "; ".join(reason for _, reason in result.skipped)
            return out
        out.action = level
        out.detail = why + (f" (review {result.review_id})" if result.review_id else "")
        return out


def _placed_is_a_project(outcomes: list[Settlement]) -> list[Settlement]:
    """A project the owner placed at one of their jobs (part_of applied or queued) is a project:
    its type doubt is settled when "project" was also the decision's best answer (p ≥ ASK_P)."""
    placed = {
        o.doubt.split(" ", 1)[1]
        for o in outcomes
        if o.doubt.startswith("part_of ") and o.action in ("applied", "queued")
    }
    for o in outcomes:
        page = o.doubt.split(" ", 1)[1]
        typed = o.doubt.startswith("type ") and o.action == "open" and page in placed
        if typed and o.choice == "project" and o.p >= ASK_P:
            o.action, o.detail = "kept", "the owner places it at one of their jobs"
    return outcomes


def _weight(g: Graph, page: Page) -> tuple:
    links = len(g.facts_of(page.subject, history=True)) + len(g.incoming.get(page.subject, []))
    return (links, -page.created.toordinal(), len(page.title), fold(page.id))
