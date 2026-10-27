"""Lint doubts in an interview: what they are about and how to ask them.

``questions.md`` items added by ``brain lint`` carry the finding's key (``doubt::``, schema
v0.6; older items have it at the start of their reason). Here:

- ``Doubt``: the key parsed into its kind and ids (``part_of projects/kiln``,
  ``type projects/kiln``, ``merge projects/a projects/b``, ``job f-x2y3z4``…).
- ``EntityInfo`` / ``entity_infos``: what the vault has on each entity a doubt is about (title,
  type, one line, where it came from, the owner's employers ranked by evidence), read once when
  the interview starts. The plan asks one combined question per entity from it, the session
  answers "tell me more" with it, and closing settles the doubts against it.
- ``combined_hint``: the one question for all of an entity's ``type`` and ``part_of`` doubts.

Private pages (or pages of a private region) give no line and no sources unless the interview
is about a private region (``include_private``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date

from brain.lint.edges import likely_jobs, named_jobs, part_of_text
from brain.lint.graph import TYPE_WORDS, Graph, Job
from brain.lint.names import fold
from brain.vault.generated import first_sentence
from brain.vault.models import SELF_ID, Page
from brain.vault.store import Vault

SETTLED_KINDS = frozenset({"part_of", "type", "merge", "job"})  # answers can settle these
LINE_CHARS = 140
MAX_SOURCES = 2
_DOC_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}-")
_EPISODE = re.compile(r"^ep-(\d{4})(\d{2})(\d{2})-")


@dataclass(frozen=True)
class Doubt:
    kind: str  # part_of | type | merge | job | located_in | subject | two | orphan | order…
    ids: tuple[str, ...]

    @classmethod
    def parse(cls, key: str | None) -> Doubt | None:
        parts = (key or "").split()
        if len(parts) < 2:
            return None
        return cls(parts[0], tuple(parts[1:]))

    @property
    def key(self) -> str:
        return " ".join((self.kind, *self.ids))

    @property
    def settles(self) -> bool:
        """An answer can settle it (an option set the owner's words map onto)."""
        return self.kind in SETTLED_KINDS

    @property
    def entities(self) -> tuple[str, ...]:
        """The pages it is about (a ``job`` doubt is about a fact; ``two`` names a predicate)."""
        if self.kind in ("job", "order", "subject"):
            return ()
        if self.kind == "two":
            return self.ids[1:]
        return self.ids


@dataclass(frozen=True)
class EntityInfo:
    id: str
    title: str
    type: str
    line: str = ""  # one line about it (its summary's first sentence, else a fact)
    sources: tuple[str, ...] = ()  # where the brain learned about it, readable
    private: bool = False
    jobs: tuple[str, ...] = ()  # every employer of the owner, most likely first (typed choices)
    named: tuple[str, ...] = ()  # the ones a question names (≤ 2, may be none)
    named_current: bool = False

    @property
    def word(self) -> str:
        return TYPE_WORDS.get(self.type, self.type)

    def brief(self) -> str:
        """``Fraud alerts: a rules engine for card payments (from your document “cv”)``."""
        text = self.title
        if self.line:
            text += f": {self.line.rstrip('.')}"
        if self.sources:
            text += f" (from {' and '.join(self.sources)})"
        return text


def source_name(src: str) -> str | None:
    """A source as the owner would recognise it: ``raw/docs/2026-10-20-cv.md`` → ``your document
    “cv”``, an episode → ``a conversation on 2026-10-15``."""
    if src.startswith("raw/conversations/"):
        return "an earlier conversation"
    if src.startswith("raw/"):
        stem = src.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        stem = _DOC_DATE.sub("", stem).replace("-", " ").replace("_", " ").strip()
        return f"your document “{stem}”" if stem else "a document of yours"
    if m := _EPISODE.match(src):
        return f"a conversation on {m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def _private(vault: Vault, page: Page) -> bool:
    return page.privacy == "private" or vault.lifemap.privacy_for(page.regions) == "private"


def _line(g: Graph, page: Page) -> str:
    if page.summary.strip():
        return first_sentence(g.readable(page.summary), LINE_CHARS)
    facts = [r.fact for r in g.facts_of(page.subject) if r.fact.extra.get("privacy") != "private"]
    facts += [
        r.fact for r in g.incoming.get(page.subject, []) if r.fact.extra.get("privacy") != "private"
    ]
    return first_sentence(g.readable(facts[0].text), LINE_CHARS) if facts else ""


def _sources(g: Graph, page: Page) -> tuple[str, ...]:
    out: list[str] = []
    for src in g.sources_about(page.subject):
        name = source_name(src)
        if name and name not in out:
            out.append(name)
        if len(out) >= MAX_SOURCES:
            break
    return tuple(out)


def _distinct(jobs: Iterable[Job]) -> list[Job]:
    seen: set[str] = set()
    out = []
    for job in jobs:
        if fold(job.name) not in seen:
            seen.add(fold(job.name))
            out.append(job)
    return out


def info_for(g: Graph, page: Page, *, include_private: bool = False) -> EntityInfo:
    private = _private(g.vault, page)
    hide = private and not include_private
    jobs = _distinct(g.jobs)
    ranked = likely_jobs(g, page, jobs) if page.type == "project" else [(j, 0.0) for j in jobs]
    named = named_jobs(ranked) if page.type == "project" else []
    return EntityInfo(
        id=page.id,
        title=page.title,
        type=page.type,
        line="" if hide else _line(g, page),
        sources=() if hide else _sources(g, page),
        private=private,
        jobs=tuple(j.name for j, _ in ranked),
        named=tuple(j.name for j in named),
        named_current=any(j.current for j in named),
    )


def entity_infos(
    vault: Vault,
    doubts: Iterable[str | None],
    *,
    today: date | None = None,
    include_private: bool = False,
    graph: Graph | None = None,
) -> dict[str, EntityInfo]:
    """What the vault has on the entities of these doubts (keys), by entity id."""
    g = graph or Graph(vault, today or date.today())
    wanted: list[str] = []
    for key in doubts:
        if doubt := Doubt.parse(key):
            wanted += [e for e in doubt.entities if e not in wanted]
    return {
        e: info_for(g, g.pages[e], include_private=include_private)
        for e in wanted
        if e in g.pages and e != SELF_ID
    }


def employers(vault: Vault, today: date | None = None) -> list[str]:
    """The owner's employers, current first, then most recent (for doubts about a duty)."""
    return [j.name for j in _distinct(Graph(vault, today or date.today()).jobs)]


# --- questions ---------------------------------------------------------------------------------


def _where(info: EntityInfo) -> str:
    if not info.named:
        return "at one of your jobs, or on your own"
    return f"at {' or '.join(info.named)}, or on your own"


def combined_hint(info: EntityInfo, kinds: set[str]) -> str | None:
    """One question for an entity's ``type`` and ``part_of`` doubts (None: nothing to combine).
    The employers are those with evidence (at most two), never the whole list."""
    if "type" in kinds and "part_of" in kinds:
        return (
            f"What is {info.title} to you: a project ({_where(info)}), "
            "or a topic or tool you work with?"
        )
    if "part_of" in kinds:
        return part_of_text(info.title, _named_jobs(info))
    if "type" in kinds:
        if info.type == "project":
            return f"What is {info.title} to you: a project, or a topic or tool you work with?"
        return f"What is {info.title} to you: {_a(info.word)}, or something else?"
    return None


def merge_hint(a: EntityInfo, b: EntityInfo) -> str:
    return f"Are {a.title} and {b.title} one and the same {a.word}, or two different ones?"


def _a(word: str) -> str:
    return ("an " if word[:1] in "aeiou" else "a ") + word


def _named_jobs(info: EntityInfo) -> list[Job]:
    """The named employers as ``Job``s for ``part_of_text`` (only name and current matter)."""
    return [
        Job(org=None, name=n, start=None, end=None, current=info.named_current, fact_id="")
        for n in info.named
    ]


def infos_text(infos: Mapping[str, EntityInfo], ids: Iterable[str]) -> list[str]:
    return [infos[i].brief() for i in ids if i in infos]
