"""Check 1: duplicate entities (two pages for one person, organisation, project, place…).

1. **Candidates** (code), pairs of pages of the same type: names alike once folded (same words
   in any order without generic words such as "University of" or "École", an acronym, a typo,
   one name inside the other, two people sharing a first name; ``brain.lint.names``), or, failing
   that, profiles (names + summary + facts) whose embeddings have cosine ≥ ``EMBED_MIN``.
   Never candidates: two people whose full names have different surnames, and two pages linked
   by a fact (one is part of, or related to, the other).
2. **Order**: a page's candidates are ordered by the reranker (its profile against theirs) and
   only the best ``MAX_PER_PAGE`` are asked about.
3. **Decision**: one typed choice per pair on both profiles and the source sentences naming
   them: the same {type}, or two different ones. p(same) ≥ ``MERGE_P`` proposes
   ``merge_entities`` (always reviewed), unless the pages disagree on a fact that would tell two
   entities apart (born, hometown, located_in); that, and ``ASK_P`` ≤ p < ``MERGE_P``, becomes a
   question for the owner.
4. **Several look-alikes** (p(same) ≥ ``ASK_P`` with two or more pages: "Sam" next to two
   Sams, "The Atlas migration" next to Atlas Migration and Atlas Reporting): one exclusive
   choice "which of these is the same, or none?". Only a pick at ``MERGE_P`` merges, and settles
   the page's other pairs as different ("none" at ``MERGE_P`` settles them all); otherwise all its
   pairs become questions.

The page kept is the one with more facts and links (then the older page, then the longer name).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from brain.lint.common import ASK_P, ERRORS, Context, Verdict
from brain.lint.graph import TYPE_WORDS
from brain.lint.names import best_match, different_people, fold
from brain.lint.report import Ask, Finding
from brain.llm.decide import Choice, ChoiceQuestion
from brain.prompts import load_prompt
from brain.vault.models import Page
from brain.vault.ops import MergeEntities

MERGE_P = 0.9  # p(same) to propose a merge (a wrong merge corrupts two pages)
# cosine of two profiles (instructed) that makes them candidates without a name match. On the
# eval vaults: duplicates 0.50-0.95 (acronyms lowest), different pages ≤ 0.65 ("Sam" / "Sam Okafor")
EMBED_MIN = 0.75
MAX_PER_PAGE = 3
MAX_PAIRS = 60  # decisions per run; the rest wait for the next run
KINDS = ("person", "org", "project", "place", "event", "topic")
TELLING = ("born", "hometown", "located_in")  # different values: two entities
INSTRUCT = (
    "Instruct: Retrieve the page that describes the same person, organisation or thing\nQuery: "
)
SAME, DIFFERENT = "same", "different"


def question(kind: str) -> ChoiceQuestion:
    word = TYPE_WORDS.get(kind, kind)
    return ChoiceQuestion(
        name="same",
        instructions=f"Are page A and page B about the same {word}?",
        choices=[
            Choice(
                value=SAME,
                description=f"yes, one {word}: one name is another way of writing the other "
                "(a nickname, a misspelling, an abbreviation, a translation, a longer form)",
            ),
            Choice(
                value=DIFFERENT,
                description=f"no, two different {word}s that only look alike or are related",
            ),
        ],
    )


@dataclass
class Pair:
    a: Page
    b: Page
    how: str
    strength: int
    verdict: Verdict | None = None


def _profile(ctx: Context, page: Page) -> str:
    return ctx.graph.describe(page)


def _telling_conflict(ctx: Context, a: Page, b: Page) -> str | None:
    g = ctx.graph
    for p in TELLING:
        va = {fold(g.readable(r.fact.o)) for r in g.facts_of(a.subject) if r.fact.p == p}
        vb = {fold(g.readable(r.fact.o)) for r in g.facts_of(b.subject) if r.fact.p == p}
        if va and vb and not va & vb:
            return p
    return None


def _weight(ctx: Context, page: Page) -> tuple:
    g = ctx.graph
    links = len(g.facts_of(page.subject, history=True)) + len(g.incoming.get(page.subject, []))
    return (links, -page.created.toordinal(), len(page.title), page.id)


def _candidates(ctx: Context) -> list[Pair]:
    g = ctx.graph
    pairs: dict[tuple[str, str], Pair] = {}
    for kind in KINDS:
        pages = sorted((p for p in g.entities.values() if p.type == kind), key=lambda p: p.id)
        for a, b in combinations(pages, 2):
            if not ctx.in_scope(a.id, b.id) or _never(ctx, a, b):
                continue
            if m := best_match(g.names(a), g.names(b), kind):
                pairs[(a.id, b.id)] = Pair(a, b, f"{m.how}: {m.a!r} / {m.b!r}", m.strength)
        if ctx.embeddings and ctx.llm is not None and len(pages) > 1:
            _embedding_pairs(ctx, pages, pairs)
    return list(pairs.values())


def _never(ctx: Context, a: Page, b: Page) -> bool:
    if a.type == "person" and different_people(ctx.graph.names(a), ctx.graph.names(b)):
        return True
    return ctx.graph.linked(a.subject, b.subject) or ctx.graph.linked(b.subject, a.subject)


def _embedding_pairs(ctx: Context, pages: Sequence[Page], pairs: dict) -> None:
    try:
        vectors = ctx.llm.embed([INSTRUCT + _profile(ctx, p) for p in pages])
    except ERRORS as err:
        ctx.note(f"duplicates: no embeddings ({type(err).__name__}): names only")
        ctx.embeddings = False
        return
    sims = np.asarray(vectors) @ np.asarray(vectors).T
    for i, j in combinations(range(len(pages)), 2):
        a, b = pages[i], pages[j]
        if (a.id, b.id) in pairs or sims[i, j] < EMBED_MIN:
            continue
        if ctx.in_scope(a.id, b.id) and not _never(ctx, a, b):
            pairs[(a.id, b.id)] = Pair(a, b, f"embedding {sims[i, j]:.2f}", 1)


def _ordered(ctx: Context, pairs: list[Pair]) -> list[Pair]:
    """Each page keeps its best ``MAX_PER_PAGE`` candidates (reranked when it has several)."""
    by_page: dict[str, list[Pair]] = {}
    for pair in pairs:
        by_page.setdefault(pair.a.id, []).append(pair)
        by_page.setdefault(pair.b.id, []).append(pair)
    keep: set[int] = set()
    for page_id, mine in by_page.items():
        mine.sort(key=lambda p: -p.strength)
        if len(mine) > 1 and ctx.rerank and ctx.llm is not None:
            page = ctx.graph.pages[page_id]
            others = [p.b if p.a.id == page_id else p.a for p in mine]
            try:
                ranked = ctx.llm.rerank(
                    _profile(ctx, page), [_profile(ctx, o) for o in others], top_n=len(others)
                )
                mine = [mine[i] for i, _ in ranked]
            except ERRORS as err:
                ctx.note(f"duplicates: no rerank ({type(err).__name__}): name order kept")
                ctx.rerank = False
        keep |= {id(p) for p in mine[:MAX_PER_PAGE]}
    chosen = [p for p in pairs if id(p) in keep]
    chosen.sort(key=lambda p: (-p.strength, p.a.id, p.b.id))
    return chosen[:MAX_PAIRS]


def _input(ctx: Context, pair: Pair) -> str:
    g = ctx.graph
    sources = [*g.sources_about(pair.a.subject), *g.sources_about(pair.b.subject)]
    names = [*g.names(pair.a), *g.names(pair.b)]
    excerpt, _ = g.excerpt(dict.fromkeys(sources), names)
    return load_prompt("lint_same_entity").format(
        kind=TYPE_WORDS.get(pair.a.type, pair.a.type),
        a=_profile(ctx, pair.a),
        b=_profile(ctx, pair.b),
        sources=excerpt or "(nothing more)",
    )


NONE = "none of them"


def which_question(kind: str, others: list[Page]) -> ChoiceQuestion:
    word = TYPE_WORDS.get(kind, kind)
    titles = [o.title for o in others]
    values = titles if len({fold(t) for t in titles}) == len(titles) else [o.id for o in others]
    return ChoiceQuestion(
        name="which",
        instructions=f"Which of the other pages is about the same {word} as the page?",
        choices=[
            Choice(value=NONE, description=f"none: each is a different {word}"),
            *(
                Choice(value=v, description=f"the same {word} as {o.title}")
                for v, o in zip(values, others, strict=True)
            ),
        ],
    )


def _which(ctx: Context, page: Page, others: list[Page]) -> tuple[str | None, float]:
    """One exclusive choice for a page with several look-alikes: (the matching page id, or
    ``NONE``, or None without an answer; its p)."""
    g = ctx.graph
    sources = [g.sources_about(page.subject)] + [g.sources_about(o.subject) for o in others]
    names = [*g.names(page), *(n for o in others for n in g.names(o))]
    excerpt, _ = g.excerpt(dict.fromkeys(s for group in sources for s in group), names)
    text = load_prompt("lint_which_entity").format(
        kind=TYPE_WORDS.get(page.type, page.type),
        page=_profile(ctx, page),
        candidates="\n\n".join(f"[{o.title}]\n{_profile(ctx, o)}" for o in others),
        sources=excerpt or "(nothing more)",
    )
    q = which_question(page.type, others)
    v = ctx.decide(text, q, f"duplicates which {page.id}")
    if v is None or not v.measured:
        return None, 0.0
    ctx.note(f"duplicates: which page is {page.id}? {v.summary()}")
    if v.choice == NONE:
        return NONE, v.p
    values = [c.value for c in q.choices if c.value != NONE]
    return others[values.index(v.choice)].id, v.p


def check_duplicates(ctx: Context) -> list[Finding]:
    pairs = _ordered(ctx, _candidates(ctx))
    for pair in pairs:
        pair.verdict = ctx.decide(
            _input(ctx, pair), question(pair.a.type), f"duplicates {pair.a.id} / {pair.b.id}"
        )
    likely = [p for p in pairs if (v := p.verdict) and v.choice == SAME and v.p >= ASK_P]
    partners: dict[str, list[Page]] = {}
    for pair in likely:
        partners.setdefault(pair.a.id, []).append(pair.b)
        partners.setdefault(pair.b.id, []).append(pair.a)
    # a page with several look-alikes: one exclusive choice decides which one (if any) it is
    which = {
        page_id: _which(ctx, ctx.graph.pages[page_id], others)
        for page_id, others in sorted(partners.items())
        if len(others) > 1
    }
    findings = []
    for pair in likely:
        v = pair.verdict
        assert v is not None
        a, b = pair.a, pair.b
        blocked = None
        for this, other in ((a, b), (b, a)):
            if this.id not in which:
                continue
            winner, p = which[this.id]
            if winner is not None and p >= MERGE_P and winner != other.id:
                blocked = "skip"  # it is another look-alike, or none of them
            elif winner != other.id or p < MERGE_P:
                blocked = blocked or f"several pages look like {this.title} (which: p {p:.2f})"
        if blocked == "skip":
            continue
        keep, gone = sorted((a, b), key=lambda p: _weight(ctx, p), reverse=True)
        key = f"merge {min(a.id, b.id)} {max(a.id, b.id)}"
        finding = Finding(
            check="duplicates",
            key=key,
            summary=f"{gone.title!r} ({gone.id}) looks like {keep.title!r} ({keep.id}) "
            f"[{pair.how}]",
            about=[keep.id, gone.id],
            p=v.p,
            decision=v.summary(),
        )
        if blocked is None and (clash := _telling_conflict(ctx, a, b)):
            blocked = f"they disagree on {clash}"
        if blocked is None and not ctx.free(a.id, b.id):
            blocked = "already changed by this run"
        if v.acts(MERGE_P) and blocked is None:
            finding.ops = [MergeEntities(keep=keep.id, merge=gone.id)]
            ctx.busy.add(gone.id)
            ctx.kept.add(keep.id)
        else:
            if blocked:
                finding.summary += f"; not merged: {blocked}"
            word = TYPE_WORDS.get(a.type, a.type)
            finding.ask = Ask(
                text=f"Are {gone.title} and {keep.title} the same {word}?",
                reason=f"lint {key}: p(same) {v.p:.2f}",
                about=keep.id,
                region=ctx.region(keep.id) or ctx.region(gone.id),
                private=ctx.private(a.id, b.id),
            )
        findings.append(finding)
    return findings
