"""Check 3: facts filed on the wrong subject (a project's target filed as the owner's goal).

1. **Candidates** (code): the owner's current facts with a predicate projects also have
   (``goal``, ``deadline``, ``status``, ``uses``) whose statement, or the source sentences that
   share its words, name a project page.
2. **Decision**: one typed choice on the statement and those sentences: whose {goal} is it,
   the owner's own, one of the named projects', or unclear.
3. **Action** at ``ACT_P``: ``refile_fact`` to the project (same statement, dates and sources);
   when the project already holds another current value of that single-valued predicate, the
   owner is asked instead. ``ASK_P`` ≤ p < ``ACT_P``: a question.
"""

from __future__ import annotations

from brain.lint.common import ASK_P, Context
from brain.lint.graph import Ref, names_in, units
from brain.lint.names import fold
from brain.lint.report import Ask, Finding
from brain.llm.decide import Choice, ChoiceQuestion
from brain.prompts import load_prompt
from brain.vault.generated import first_sentence
from brain.vault.models import SELF_ID, Page
from brain.vault.ops import RefileFact

ACT_P = 0.8
LABEL = {"goal": "goal", "deadline": "deadline", "status": "status", "uses": "tool"}
OWNER, UNCLEAR = "the owner", "unclear"
MAX_PROJECTS = 3
STOP = frozenset(
    [
        "the",
        "a",
        "an",
        "of",
        "to",
        "and",
        "or",
        "in",
        "on",
        "at",
        "for",
        "by",
        "with",
        "is",
        "are",
        "was",
        "be",
        "my",
        "our",
        "their",
        "its",
    ]
)


def _content_words(text: str) -> set[str]:
    return {w for w in fold(text).split() if w not in STOP and len(w) > 2}


def related_sentences(ctx: Context, ref: Ref, names: list[str], limit: int = 4) -> list[str]:
    """Sentences (list items with their heading line) of the fact's sources that share two
    words with it, then ones that name one of ``names``."""
    want = _content_words(ref.fact.text) | _content_words(ref.fact.o)
    about: list[str] = []
    naming: list[str] = []
    for src in ref.fact.src:
        for sentence in units(ctx.graph.source_text(src) or ""):
            if sentence in about or sentence in naming:
                continue
            if len(want & _content_words(sentence)) >= 2:
                about.append(sentence)
            elif names and names_in(sentence, names):
                naming.append(sentence)
    return (about + naming)[:limit]


def _projects_named(ctx: Context, ref: Ref) -> list[Page]:
    g = ctx.graph
    projects = [p for p in g.entities.values() if p.type == "project" and ctx.free(p.id)]
    text = " ".join([ref.fact.text, *related_sentences(ctx, ref, [])])
    named = [p for p in projects if names_in(text, g.names(p))]
    return sorted(named, key=lambda p: p.id)[:MAX_PROJECTS]


def question(label: str, projects: list[Page]) -> ChoiceQuestion:
    choices = [  # the answers that are not a project first: a listed name draws the eye
        Choice(value=UNCLEAR, description="the text does not say"),
        Choice(value=OWNER, description=f"a personal {label} of the owner, not one project's"),
        *(
            Choice(value=p.title, description=f"the {label} of the project {p.title}")
            for p in projects
        ),
    ]
    return ChoiceQuestion(name="whose", instructions=f"Whose {label} is this?", choices=choices)


def check_subjects(ctx: Context) -> list[Finding]:
    g = ctx.graph
    findings = []
    for ref in g.facts_of(SELF_ID):
        label = LABEL.get(ref.fact.p)
        if label is None:
            continue
        projects = _projects_named(ctx, ref)
        if not projects or not ctx.in_scope(SELF_ID, *(p.id for p in projects)):
            continue
        names = [n for p in projects for n in g.names(p)]
        sentences = related_sentences(ctx, ref, names)
        listing = "\n".join(
            f"- {p.title}: {first_sentence(g.readable(p.summary), 160) if p.summary else p.type}"
            for p in projects
        )
        text = load_prompt("lint_fact_subject").format(
            statement=g.readable(ref.fact.text),
            sources="\n".join(f"- {s}" for s in sentences) or "(nothing more)",
            projects=listing,
        )
        v = ctx.decide(text, question(label, projects), f"subjects ^{ref.fact.id}")
        if v is None or v.choice in (OWNER, UNCLEAR) or v.p < ASK_P:
            continue
        project = next(p for p in projects if p.title == v.choice)
        finding = Finding(
            check="subjects",
            key=f"subject {ref.fact.id}",
            summary=f"^{ref.fact.id} {ref.fact.text!r} reads like {project.title}'s {label}, "
            "not the owner's",
            about=[ref.fact.id, project.id],
            p=v.p,
            decision=v.summary(),
        )
        held = [
            r
            for r in g.facts_of(project.subject)
            if r.fact.p == ref.fact.p
            and ref.fact.p != "uses"
            and fold(r.fact.o) != fold(ref.fact.o)
        ]
        if v.acts(ACT_P) and not held:
            finding.ops = [RefileFact(fact_id=ref.fact.id, subject=project.id)]
        else:
            if v.acts(ACT_P):
                finding.summary += f"; {project.title} already has a {label}"
            finding.ask = Ask(
                text=f"Is “{g.readable(ref.fact.text).rstrip('.')}” your own {label}, "
                f"or {project.title}'s?",
                reason=f"lint subject {ref.fact.id}: {project.id} p {v.p:.2f}",
                about=project.id,
                region=ctx.region(project.id, "projects"),
                private=ctx.private(project.id),
            )
        findings.append(finding)
    return findings
