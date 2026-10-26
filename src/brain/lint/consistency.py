"""Check 5: consistency, in code only (no model).

- **Two current values** of a single-valued predicate for one subject (two cities lived in
  now), and two current jobs that both claim to be open-ended: the owner is asked which holds.
- **Valid time ends before it starts**: reported and asked.
- **Orphan pages**: no facts and no links: reported (and, but for people, whom coverage asks
  about, a question). A page with one link and no facts is not an orphan: ingest files most
  employers and schools that way (their facts live on the owner's page).
- **A literal that names a page**: a fact whose value is exactly the title or an alias of one
  page of a type the predicate accepts (``works_at`` "Fabrikam Analytics" while
  ``orgs/fabrikam-analytics`` exists) is re-filed with the link (``refile_fact``, reviewed).
- **Format**: parse warnings, inferred facts without ``evidence::``.
- **AGENTS.md**: the predicate list and schema version mirrored in ``vault_template/AGENTS.md``
  (and the vault's copy) match ``brain.vault.models``.
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

from brain.config import REPO_DIR
from brain.ingest.normalise import OBJECT_TYPES
from brain.lint.common import Context
from brain.lint.graph import Ref
from brain.lint.names import fold
from brain.lint.report import Ask, Finding
from brain.vault.models import (
    PREDICATES,
    RELATIONSHIP_PREDICATES,
    SCHEMA_VERSION,
    SELF_ID,
    is_multi_valued,
    link_target,
)
from brain.vault.ops import RefileFact
from brain.vault.writer import overlaps

TEMPLATE_AGENTS = REPO_DIR / "vault_template" / "AGENTS.md"
VALUE_ONLY = frozenset({"name", "born", "role", "team", "deadline", "status", "note", "speaks"})
VALUE_ONLY |= frozenset({"nationality", "degree", "certification", "responsibility", "goal"})
VALUE_ONLY |= frozenset({"values", "habit", "health", "fears", "has_strength", "has_flaw"})
TOPIC_OBJECTS = frozenset({"has_skill", "uses", "interest", "likes", "dislikes"})
QUALIFIERS = frozenset({"at", "level", "rel"})
QUESTION = {
    "lives_in": "Where do you live now: {a} or {b}?",
    "role": "What is your role now: {a} or {b}?",
    "team": "Which team are you in now: {a} or {b}?",
    "manager": "Who is your manager now: {a} or {b}?",
}


def _region(ctx: Context, p: str, subject: str) -> str | None:
    if subject == SELF_ID:
        return ctx.graph.vault.lifemap.region_for(p, "self")
    page = ctx.graph.entities.get(subject)
    kind = page.type if page else "person"
    return ctx.graph.vault.lifemap.region_for(p, kind) or ctx.region(subject)


def _two_current(ctx: Context, findings: list[Finding]) -> None:
    g = ctx.graph
    subjects = {p.subject for p in g.pages.values()}
    for subject in sorted(subjects):
        if not ctx.in_scope(subject):
            continue
        current = g.current(subject)
        by_p: dict[str, list[Ref]] = {}
        for ref in current:
            by_p.setdefault(ref.fact.p, []).append(ref)
        for p, refs in sorted(by_p.items()):
            multi = is_multi_valued(p, subject)
            if multi and p != "works_at":
                continue
            for a, b in combinations(refs, 2):
                fa, fb = a.fact, b.fact
                if fold(g.readable(fa.o)) == fold(g.readable(fb.o)) or not overlaps(fa, fb):
                    continue
                objects = [t for t in (link_target(fa.o), link_target(fb.o)) if t]
                if not ctx.free(*objects):
                    continue  # one of them is being merged or demoted: that settles it
                if multi and (fa.valid_to or fb.valid_to):
                    continue  # works_at: only two open-ended current jobs are doubtful
                va, vb = g.readable(fa.o), g.readable(fb.o)
                who = "you" if subject == SELF_ID else g.title(subject)
                if subject == SELF_ID and p in QUESTION:
                    text = QUESTION[p].format(a=va, b=vb)
                elif p == "works_at":
                    text = f"Do you still work at both {va} and {vb}?"
                else:
                    ta, tb = fa.text.rstrip("."), fb.text.rstrip(".")
                    text = f"For {who}, which is right now: {ta}, or {tb}?"
                key = f"two {p} {subject}"
                findings.append(
                    Finding(
                        check="consistency",
                        key=key,
                        summary=f"two current {p} for {g.title(subject)}: "
                        f"^{fa.id} {va!r} and ^{fb.id} {vb!r}",
                        about=[fa.id, fb.id],
                        ask=Ask(
                            text=text,
                            reason=f"lint {key}: ^{fa.id} ^{fb.id}",
                            about=None if subject == SELF_ID else subject,
                            region=_region(ctx, p, subject),
                            priority=4,
                            private=ctx.private(subject) if subject != SELF_ID else False,
                        ),
                    )
                )


def _order(ctx: Context, findings: list[Finding]) -> None:
    g = ctx.graph
    for ref in g.refs:
        f = ref.fact
        if not ctx.in_scope(ref.subject) or not (f.valid_from and f.valid_to):
            continue
        if f.valid_to < f.valid_from[: len(f.valid_to)]:
            key = f"order {f.id}"
            findings.append(
                Finding(
                    check="consistency",
                    key=key,
                    summary=f"^{f.id} ends before it starts "
                    f"({f.valid_from} → {f.valid_to}): {f.text!r}",
                    about=[f.id],
                    ask=Ask(
                        text=f"“{g.readable(f.text).rstrip('.')}”: when did that start and end?",
                        reason=f"lint {key}",
                        about=None if ref.subject == SELF_ID else ref.subject,
                        private=ctx.private(ref.subject),
                    ),
                )
            )


def _orphans(ctx: Context, findings: list[Finding]) -> None:
    g = ctx.graph
    for page in sorted(g.entities.values(), key=lambda p: p.id):
        if not ctx.in_scope(page.id) or not ctx.free(page.id) or page.id in ctx.kept:
            continue
        # no facts, no links and no episode (a page with one link and no facts of its own is how
        # ingest files most employers and schools: their facts live on the owner's page)
        if g.facts_of(page.subject, history=True) or g.incoming.get(page.subject):
            continue
        if any(page.subject in e.entities for e in g.episodes.values()):
            continue
        key = f"orphan {page.id}"
        ask = None
        if page.type != "person":  # people: coverage asks "who is X to you?"
            ask = Ask(
                text=f"What is {page.title} to you?",
                reason=f"lint {key}",
                about=page.id,
                region=ctx.region(page.id),
                priority=2,
                private=ctx.private(page.id),
            )
        findings.append(
            Finding(
                check="consistency",
                key=key,
                summary=f"orphan page {page.id}: no facts, no links, no episode",
                about=[page.id],
                ask=ask,
            )
        )


def _accepts(p: str, page_type: str) -> bool:
    if p in VALUE_ONLY or p in RELATIONSHIP_PREDICATES:
        return False
    if p in OBJECT_TYPES:
        return page_type in OBJECT_TYPES[p]
    if p in TOPIC_OBJECTS:
        return page_type == "topic"
    return p == "related_to"


def _literals(ctx: Context, findings: list[Finding]) -> None:
    g = ctx.graph
    for ref in g.refs:
        f = ref.fact
        if link_target(f.o) or (ref.history and f.by) or not ctx.in_scope(ref.subject):
            continue
        ids = [i for i in g.name_index.get(fold(f.o), []) if _accepts(f.p, g.entities[i].type)]
        ids = [i for i in ids if i != ref.subject and ctx.free(i)]
        if len(ids) != 1:
            continue
        target = ids[0]
        findings.append(
            Finding(
                check="consistency",
                key=f"link {f.id}",
                summary=f"^{f.id} {f.p}:: {f.o!r} names the page {target}",
                about=[f.id, target],
                ops=[RefileFact(fact_id=f.id, o=f"[[{target}]]")],
            )
        )


def _format(ctx: Context, findings: list[Finding]) -> None:
    g = ctx.graph
    for warning in g.vault.warnings():
        findings.append(
            Finding(check="consistency", key=f"format {warning}", summary=f"format: {warning}")
        )
    for ref in g.refs:
        if ref.fact.kind == "inferred" and not ref.fact.evidence:
            findings.append(
                Finding(
                    check="consistency",
                    key=f"evidence {ref.fact.id}",
                    summary=f"inferred fact ^{ref.fact.id} has no evidence::",
                    about=[ref.fact.id],
                )
            )


PREDICATE_HEADING = re.compile(r"^###\s+Predicates\s+\(v([\d.]+)\)", re.M)


def agents_predicates(text: str) -> tuple[str | None, set[str]]:
    """The schema version and the predicates listed under ``### Predicates (vX)``."""
    m = PREDICATE_HEADING.search(text)
    if not m:
        return None, set()
    section = text[m.end() :]
    end = re.search(r"^#{1,3}\s", section, re.M)
    section = section[: end.start()] if end else section
    found: set[str] = set()
    for line in section.splitlines():
        if not line.lstrip().startswith("- "):
            continue
        body = re.sub(r"\([^)]*\)", "", line.split(":", 1)[-1])  # drop "(optional `at`: …)"
        found |= {w for w in re.findall(r"`([a-z_]+)`", body) if w not in QUALIFIERS}
    return m.group(1), found


def agents_problems(path: Path) -> list[str]:
    if not path.is_file():
        return []
    version, listed = agents_predicates(path.read_text(encoding="utf-8"))
    out = []
    if version != SCHEMA_VERSION:
        out.append(f"{path.name} lists predicates of v{version}, the schema is v{SCHEMA_VERSION}")
    if missing := sorted(PREDICATES - listed):
        out.append(f"{path.name} misses predicates: {', '.join(missing)}")
    if extra := sorted(listed - PREDICATES):
        out.append(f"{path.name} lists unknown predicates: {', '.join(extra)}")
    return out


def _agents(ctx: Context, findings: list[Finding]) -> None:
    for path, hint in (
        (TEMPLATE_AGENTS, " (update vault_template/AGENTS.md)"),
        (
            ctx.graph.vault.root / "AGENTS.md",
            " (`brain init-vault --update-template` refreshes it)",
        ),
    ):
        for problem in agents_problems(path):
            findings.append(
                Finding(check="consistency", key=f"agents {path}", summary=f"{problem}{hint}")
            )


def check_consistency(ctx: Context) -> list[Finding]:
    findings: list[Finding] = []
    _two_current(ctx, findings)
    _order(ctx, findings)
    _orphans(ctx, findings)
    _literals(ctx, findings)
    _format(ctx, findings)
    _agents(ctx, findings)
    return findings
