"""Check 2: pages of a doubtful type (a topic, tool or hobby made into a project or organisation
page; an organisation filed as a place).

1. **Screen** (code): a page is doubtful when other facts use it as a value (``has_skill``,
   ``interest``, ``uses``, ``likes``…: only topics and tools are skills or interests), or when it
   has no fact of its own and at most one link. Pages whose facts and links fit their type are
   not asked about (cost).
2. **Decision**: one typed choice on the page's profile and the source sentences naming it
   (never its current type: shown a proposal, the model agrees with it): person / organisation /
   project / place / event / topic / tool / activity.
3. **Action** at ``ACT_P`` (it removes or moves a page, like a merge), and only when a fact or
   link supports the new type (used as a value; a predicate only that type takes): a topic, tool
   or activity
   is demoted (``demote_entity``: links become the literal name), after re-filing the facts that
   used it as a page (``worked_on`` X → ``has_skill`` / ``interest``); another page type is a
   ``retype_entity``. A page with facts of its own is not demoted: the owner is asked.
   ``ASK_P`` ≤ p < ``ACT_P``, or no supporting fact: a question, unless nothing but the name
   speaks for it (no supporting fact, no source sentence): then a report line only.
"""

from __future__ import annotations

from brain.ingest.normalise import OBJECT_TYPES
from brain.lint.common import ASK_P, Context
from brain.lint.graph import TYPE_WORDS
from brain.lint.report import Ask, Finding
from brain.llm.decide import Choice, ChoiceQuestion
from brain.prompts import load_prompt
from brain.vault.models import SELF_ID, Page, link_target
from brain.vault.ops import DemoteEntity, Operation, RefileFact, RetypeEntity

ACT_P = 0.9
VALUE_PREDICATES = frozenset({"has_skill", "interest", "uses", "likes", "dislikes", "habit"})
# predicates that fit each page type, as the page's own facts or as links to it
FITS: dict[str, frozenset[str]] = {
    "project": frozenset(
        {"owner", "status", "goal", "deadline", "stakeholder", "part_of", "worked_on"}
    ),
    "org": frozenset(
        {"works_at", "studied_at", "member_of", "stakeholder", "located_in", "part_of"}
    ),
    "person": frozenset(
        {
            "friend_of",
            "family_of",
            "partner_of",
            "colleague_of",
            "mentor_of",
            "knows",
            "manager",
            "owner",
            "stakeholder",
            "works_at",
            "lives_in",
            "met_at",
            "last_seen",
        }
    ),
    "place": frozenset({"lives_in", "hometown", "located_in", "met_at"}),
    "event": frozenset({"met_at", "part_of", "related_to"}),
}
VALUE_KINDS = ("topic", "tool", "activity")
_PERSON = "friend_of family_of partner_of colleague_of mentor_of knows met_at last_seen born"
_PERSON += " hometown lives_in works_at studied_at role team manager"
# what the subject of a predicate can be (as OBJECT_TYPES says what its object can be)
SUBJECT_TYPES: dict[str, frozenset[str]] = {
    **{p: frozenset({"person"}) for p in _PERSON.split()},
    **{p: frozenset({"project"}) for p in ("status", "goal", "deadline", "owner", "stakeholder")},
    "uses": frozenset({"project", "org"}),
    "located_in": frozenset({"org", "place", "event", "project"}),
    "part_of": frozenset({"org", "place", "event", "project"}),
}
KIND = ChoiceQuestion(
    name="kind",
    instructions="What is it?",
    choices=[
        Choice(value="person"),
        Choice(
            value="org",
            description="an organisation: company, university, school, club or community",
        ),
        Choice(
            value="project",
            description="work someone builds, leads or takes part in, "
            "including their own app or library",
        ),
        Choice(value="place", description="a city, region, country or address"),
        Choice(
            value="event",
            description="something that happened at a time: a conference, a trip, a wedding",
        ),
        Choice(value="topic", description="a field, subject or technique"),
        Choice(value="tool", description="a tool, product, language or technology someone uses"),
        Choice(value="activity", description="a hobby, sport or pastime"),
    ],
)
# a fact that linked to a demoted page as a page: the predicate it gets as a value
AS_VALUE = {
    ("self", "tool"): "has_skill",
    ("self", "topic"): "has_skill",
    ("self", "activity"): "interest",
    ("other", "tool"): "uses",
    ("other", "topic"): "related_to",
    ("other", "activity"): "related_to",
}
KEEP_AS_VALUE = VALUE_PREDICATES | {"related_to", "note", "values", "goal", "responsibility"}


def _doubtful(ctx: Context, page: Page) -> str | None:
    g = ctx.graph
    incoming = g.incoming.get(page.subject, [])
    if values := sorted({r.fact.p for r in incoming if r.fact.p in VALUE_PREDICATES}):
        return "used as a value (" + ", ".join(values) + ")"
    own = g.facts_of(page.subject, history=True)
    if not own and len(incoming) <= 1:
        return "no facts of its own"
    fits = FITS.get(page.type, frozenset())
    if not any(r.fact.p in fits for r in own) and not any(r.fact.p in fits for r in incoming):
        return f"nothing that fits a {page.type}"
    return None


def _support(ctx: Context, page: Page, kind: str) -> bool:
    """A fact or link that says the page is a ``kind`` and not its current type: used as a value
    (for a topic, tool or activity), the object of a predicate that takes only other types
    (``works_at`` X: X is an organisation), or the subject of one (X ``friend_of`` the owner: a
    person). Without one, a verdict rests on the name alone (made-up place names read as
    people's)."""
    g = ctx.graph
    incoming = [
        r for r in g.incoming.get(page.subject, []) if link_target(r.fact.o) == page.subject
    ]
    if kind in VALUE_KINDS:
        return any(r.fact.p in VALUE_PREDICATES for r in incoming)
    for ref in incoming:
        allowed = OBJECT_TYPES.get(ref.fact.p, frozenset())
        if kind in allowed and page.type not in allowed:
            return True
    for ref in g.facts_of(page.subject, history=True):
        allowed = SUBJECT_TYPES.get(ref.fact.p, frozenset())
        if kind in allowed and page.type not in allowed:
            return True
    return False


def _input(ctx: Context, page: Page) -> tuple[str, str]:
    g = ctx.graph
    excerpt, _ = g.excerpt(g.sources_about(page.subject), g.names(page))
    text = load_prompt("lint_entity_kind").format(
        page=g.describe(page), sources=excerpt or "(nothing more)"
    )
    return text, excerpt


def refiles(ctx: Context, page: Page, kind: str) -> list[Operation]:
    ops: list[Operation] = []
    for ref in ctx.graph.incoming.get(page.subject, []):
        if link_target(ref.fact.o) != page.subject:
            continue  # an at:: link: the demotion drops it
        if ref.fact.p in KEEP_AS_VALUE or (ref.history and ref.fact.by):
            continue
        who = "self" if ref.subject == SELF_ID else "other"
        ops.append(RefileFact(fact_id=ref.fact.id, p=AS_VALUE[(who, kind)]))
    return ops


def _a(word: str) -> str:
    return ("an " if word[0] in "aeiou" else "a ") + word


def check_kinds(ctx: Context) -> list[Finding]:
    findings = []
    for page in sorted(ctx.graph.entities.values(), key=lambda p: p.id):
        if page.type in ("topic",) or not ctx.in_scope(page.id) or not ctx.free(page.id):
            continue
        if not (why := _doubtful(ctx, page)):
            continue
        text, excerpt = _input(ctx, page)
        v = ctx.decide(text, KIND, f"types {page.id}")
        if v is None or v.choice == page.type:
            continue
        # demoting is one action whatever the value kind: its probability is their sum
        p = v.p_of(*VALUE_KINDS) if v.choice in VALUE_KINDS else v.p
        if p < ASK_P:
            continue
        word = "organisation" if v.choice == "org" else v.choice
        current = TYPE_WORDS.get(page.type, page.type)
        finding = Finding(
            check="types",
            key=f"type {page.id}",
            summary=f"{page.title!r} ({page.id}, {why}) looks like a {word}, not a {current}",
            about=[page.id],
            p=p,
            decision=v.summary() + (f", value kinds p {p:.2f}" if p != v.p else ""),
        )
        own = ctx.graph.facts_of(page.subject, history=True)
        supported = _support(ctx, page, v.choice)
        acts = v.measured and p >= ACT_P and supported
        if acts and v.choice in VALUE_KINDS and not own and not page.notes.strip():
            finding.ops = [*refiles(ctx, page, v.choice), DemoteEntity(id=page.id)]
            ctx.busy.add(page.id)
        elif acts and v.choice not in VALUE_KINDS:
            finding.ops = [RetypeEntity(id=page.id, type=v.choice)]  # type: ignore[arg-type]
            ctx.busy.add(page.id)
        elif not supported and not excerpt:
            finding.summary += "; nothing but the name supports it: not asked"
        else:
            if acts:
                finding.summary += "; not demoted: it has facts or notes of its own"
            elif v.measured and p >= ACT_P:
                finding.summary += f"; no fact or link fits {_a(word)}"
            finding.ask = Ask(
                text=f"Is {page.title} {_a(current)}, or {_a(word)}?",
                reason=f"lint type {page.id}: {v.choice} p {p:.2f}",
                about=page.id,
                region=ctx.region(page.id),
                private=ctx.private(page.id),
            )
        findings.append(finding)
    return findings
