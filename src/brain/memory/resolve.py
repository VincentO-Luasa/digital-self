"""Resolve the names an episode mentions to existing pages (docs/architecture.md, Flow 3).

1. **Aliases** (code): a name equal (case-folded) to one page's title or alias is that page. A
   name shared by several pages is ambiguous: step 3 decides between them.
2. **Candidates** otherwise: pages of a compatible type whose name contains the mention's words
   ("Dana" → "Dana Whitfield") or the reverse ("the Atlas project" ⊇ "Atlas"), then names
   whose embedding (Qwen3-Embedding, an instruction on both sides) has cosine ≥ ``EMBED_MIN``
   with the mention, best ``MAX_CANDIDATES``. Measured on fictional names: related pairs
   0.58–0.96, unrelated people/orgs ≤ 0.48, two cities 0.73; so embeddings only *propose*.
3. **Decision**: one typed choice question on the episode's lines naming the mention: "who or
   what is X?" over the candidates plus "someone or something else"; p ≥ ``ACT_ABOVE`` picks
   the page (and the name becomes an alias of it). Otherwise the mention stays unresolved and
   extraction may create a new page for it.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from brain.ingest.decisions import evidence
from brain.llm.decide import Choice, ChoiceQuestion
from brain.memory.segment import Mention
from brain.vault.generated import first_sentence
from brain.vault.models import SELF_ID, Page
from brain.vault.store import Vault

log = logging.getLogger(__name__)

ACT_ABOVE = 0.8  # decision: pick a candidate at or above this probability
EMBED_MIN = 0.55  # cosine: a name this close is a candidate (asked about, never merged alone)
MAX_CANDIDATES = 3
EMBEDDINGS = True  # look for candidates by embedding (replay tests switch it off)
INSTRUCT = (
    "Instruct: Retrieve the person, organisation, project or place this name refers to\nQuery: "
)
ELSE = "someone or something else"
# mention type → page types it may resolve to
COMPATIBLE = {
    "person": {"person"},
    "org": {"org"},
    "project": {"project"},
    "place": {"place", "org"},
    "event": {"event", "project"},
}
WORD = re.compile(r"[\w'-]+")
STOP = frozenset({"the", "a", "an", "my", "our", "of", "project", "team", "at", "in"})


@dataclass
class Resolution:
    name: str
    type: str
    entity_id: str | None = None
    how: str = "new"  # alias | decision | new
    p: float | None = None
    candidates: list[str] = field(default_factory=list)

    @property
    def alias(self) -> bool:
        """Resolved from another name: the name may become an alias of the page."""
        return self.entity_id is not None and self.how == "decision"


@dataclass
class ResolveReport:
    resolved: list[Resolution] = field(default_factory=list)
    seconds: float = 0.0
    asked: int = 0
    notes: list[str] = field(default_factory=list)


def _fold(text: str) -> str:
    return " ".join(text.split()).casefold()


def _words(text: str) -> frozenset[str]:
    return frozenset(w for w in WORD.findall(text.casefold()) if w not in STOP)


def _names(page: Page) -> list[str]:
    return [n for n in (page.title, *page.aliases) if n.strip()]


def describe(page: Page) -> str:
    """A short description for a choice: type and the summary's first sentence."""
    summary = first_sentence(page.summary, limit=100) if page.summary else ""
    return f"{page.type}" + (f": {summary}" if summary else "")


def _word_candidates(name: str, pages: Sequence[Page]) -> list[str]:
    words = _words(name)
    if not words:
        return []
    out = []
    for page in pages:
        for other in map(_words, _names(page)):
            if other and (words <= other or other <= words):
                out.append(page.subject)
                break
    return out


def _embed_candidates(
    llm: Any, name: str, pages: Sequence[Page], vectors: dict[str, np.ndarray]
) -> list[tuple[str, float]]:
    texts = [f"{INSTRUCT}{n}" for p in pages for n in _names(p) if n not in vectors]
    if texts:
        fresh = llm.embed(texts)
        for text, vec in zip(texts, fresh, strict=True):
            vectors[text.removeprefix(INSTRUCT)] = vec
    [query] = llm.embed([f"{INSTRUCT}{name}"])
    best: dict[str, float] = {}
    for page in pages:
        score = max(float(query @ vectors[n]) for n in _names(page))
        if score >= EMBED_MIN:
            best[page.subject] = score
    return sorted(best.items(), key=lambda kv: -kv[1])[:MAX_CANDIDATES]


def resolve_mentions(
    vault: Vault,
    llm: Any,
    mentions: Sequence[Mention],
    text: str,
    *,
    owner_names: Sequence[str] = (),
    use_embeddings: bool | None = None,
) -> ResolveReport:
    """Resolve each distinct mention. ``text``: the episode as shown to the model (evidence)."""
    start = time.perf_counter()
    report = ResolveReport()
    use_embeddings = EMBEDDINGS if use_embeddings is None else use_embeddings
    pages = {p.subject: p for p in vault.pages() if p.subject != SELF_ID}
    aliases = vault.alias_index()
    owner = {_fold(n) for n in owner_names}
    vectors: dict[str, np.ndarray] = {}
    seen: set[str] = set()
    for mention in mentions:
        key = _fold(mention.name)
        if not key or key in seen or key in owner:
            continue
        seen.add(key)
        res = Resolution(mention.name.strip(), mention.type)
        report.resolved.append(res)
        ids = [i for i in aliases.get(key, []) if i in pages]
        if len(ids) == 1:
            res.entity_id, res.how = ids[0], "alias"
            continue
        allowed = COMPATIBLE.get(mention.type, set())
        typed = [p for p in pages.values() if p.type in allowed]
        candidates = ids or _word_candidates(mention.name, typed)
        if not candidates and use_embeddings and typed:
            try:
                scored = _embed_candidates(llm, mention.name, typed, vectors)
            except Exception as err:  # noqa: BLE001 - embeddings down: aliases and words only
                report.notes.append(f"resolve: embeddings unavailable ({err})")
                use_embeddings = False
                scored = []
            candidates = [i for i, _ in scored]
            if scored:
                shown = ", ".join(f"{i} {s:.2f}" for i, s in scored)
                report.notes.append(f"resolve: {mention.name!r} looks like {shown}")
        res.candidates = candidates[:MAX_CANDIDATES]
        if res.candidates:
            _decide(llm, res, [pages[i] for i in res.candidates], text, report)
    report.seconds = time.perf_counter() - start
    return report


def _decide(
    llm: Any, res: Resolution, candidates: list[Page], text: str, report: ResolveReport
) -> None:
    choices = [Choice(value=p.title, description=describe(p)) for p in candidates]
    titles = [c.value for c in choices]
    if len(set(map(_fold, titles))) < len(titles):  # same titles: tell them apart by id
        choices = [Choice(value=p.id, description=describe(p)) for p in candidates]
    choices.append(Choice(value=ELSE, description="none of these"))
    question = ChoiceQuestion(
        name="who",
        instructions=f'In this conversation, who or what is "{res.name}"?',
        choices=choices,
    )
    try:
        [decision] = llm.decide(evidence(text, res.name), [question])
    except Exception as err:  # noqa: BLE001 - unresolved is safe: a new page or a review
        report.notes.append(f"resolve: {res.name!r}: no decision ({err})")
        return
    report.asked += 1
    if decision.refusal or decision.choice in (None, ELSE):
        shown = (
            "no answer" if decision.refusal else f"someone else (conf {decision.confidence:.2f})"
        )
        report.notes.append(f"resolve: {res.name!r}: {shown}")
        return
    p = decision.probabilities.get(decision.choice, decision.confidence)
    picked = next(
        page for page, c in zip(candidates, choices, strict=False) if c.value == decision.choice
    )
    if p < ACT_ABOVE:
        report.notes.append(
            f"resolve: {res.name!r} may be {picked.id} (p={p:.2f} < {ACT_ABOVE}): unresolved"
        )
        return
    res.entity_id, res.how, res.p = picked.subject, "decision", p
    report.notes.append(f"resolve: {res.name!r} is {picked.id} (p={p:.2f})")
