"""A read-only snapshot of the vault for the lint checks: pages, facts by subject, links, the
owner's jobs, and the source text behind facts (episode summaries, raw documents).

Evidence shown to the model is always an excerpt: the sentences of a source that name what the
question is about, never whole documents. A list item keeps the line that introduces it ("**Fabrikam
Analytics**, Lisbon — Analytics engineer › Built demand forecasting dashboards"), so a CV duty
keeps its employer (``units``).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property
from pathlib import Path

from brain.index.temporal import valid_at
from brain.lint.names import fold
from brain.vault.generated import first_sentence
from brain.vault.models import SELF_ID, WIKILINK, Episode, Fact, Page, link_target
from brain.vault.store import Vault

log = logging.getLogger(__name__)

EXCERPT_CHARS = 360  # evidence per name and source
MAX_SOURCES = 3  # sources read per question
MAX_FACT_LINES = 6
TYPE_WORDS = {
    "person": "person",
    "org": "organisation",
    "project": "project",
    "place": "place",
    "event": "event",
    "topic": "topic",
}


@dataclass(frozen=True)
class Ref:
    """A fact and where it lives."""

    page: Page
    fact: Fact
    history: bool

    @property
    def subject(self) -> str:
        return self.page.subject


@dataclass(frozen=True)
class Job:
    """One of the owner's jobs (a ``works_at`` fact)."""

    org: str | None  # org page id, None for a literal employer
    name: str
    start: str | None
    end: str | None
    current: bool
    fact_id: str

    def period(self) -> str:
        if not (self.start or self.end):
            return "dates unknown"
        return f"{self.start or '?'} to {'now' if self.current else self.end or '?'}"


@dataclass
class Graph:
    vault: Vault
    today: date
    pages: dict[str, Page] = field(init=False)
    _texts: dict[str, str | None] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.pages = {p.id: p for p in self.vault.pages()}

    # -- pages and facts -----------------------------------------------------------------------

    @cached_property
    def entities(self) -> dict[str, Page]:
        """Non-self pages by id."""
        return {p.id: p for p in self.pages.values() if p.subject != SELF_ID}

    @cached_property
    def self_pages(self) -> list[Page]:
        return [p for p in self.pages.values() if p.subject == SELF_ID]

    @cached_property
    def refs(self) -> list[Ref]:
        out = []
        for page in self.pages.values():
            out += [Ref(page, f, False) for f in page.facts]
            out += [Ref(page, f, True) for f in page.history]
        return out

    def facts_of(self, subject: str, *, history: bool = False) -> list[Ref]:
        return [r for r in self.refs if r.subject == subject and (history or not r.history)]

    def current(self, subject: str) -> list[Ref]:
        """Current facts of ``subject`` that are valid today."""
        return [r for r in self.facts_of(subject) if self.valid_now(r.fact)]

    def valid_now(self, fact: Fact) -> bool:
        try:
            return valid_at(fact.valid_from, fact.valid_to, self.today)
        except ValueError:
            return True

    @cached_property
    def incoming(self) -> dict[str, list[Ref]]:
        """Facts whose object (or ``at::``) links to an entity, by entity id."""
        out: dict[str, list[Ref]] = {}
        for ref in self.refs:
            targets = {link_target(ref.fact.o), link_target(ref.fact.extra.get("at", ""))}
            for target in targets - {None}:
                out.setdefault(target, []).append(ref)  # type: ignore[arg-type]
        return out

    def neighbours(self, entity: str) -> set[str]:
        """Entities linked to or from ``entity`` (the owner excluded)."""
        out = {r.subject for r in self.incoming.get(entity, [])}
        for ref in self.facts_of(entity, history=True):
            out |= {
                t for t in (link_target(ref.fact.o), link_target(ref.fact.extra.get("at", ""))) if t
            }
        return out - {entity, SELF_ID, "self/profile"}

    def linked(self, a: str, b: str) -> bool:
        return b in self.neighbours(a)

    def find(self, fact_id: str) -> Ref | None:
        return next((r for r in self.refs if r.fact.id == fact_id), None)

    # -- names ---------------------------------------------------------------------------------

    def names(self, page: Page) -> list[str]:
        return [n for n in (page.title, *page.aliases) if n.strip()]

    def title(self, entity: str) -> str:
        if entity in (SELF_ID, "self/profile"):
            return "the owner"
        page = self.pages.get(entity)
        return page.title if page else entity.rsplit("/", 1)[-1].replace("-", " ")

    def readable(self, value: str) -> str:
        """Wikilinks as names."""

        def repl(m: re.Match[str]) -> str:
            raw = m.group(0)
            if "|" in raw:
                return raw[raw.index("|") + 1 : -2].strip()
            return self.title(m.group(1).strip())

        return " ".join(WIKILINK.sub(repl, value).split())

    @cached_property
    def name_index(self) -> dict[str, list[str]]:
        """Folded titles and aliases → entity ids."""
        out: dict[str, list[str]] = {}
        for page in self.entities.values():
            for name in self.names(page):
                ids = out.setdefault(fold(name), [])
                if page.id not in ids:
                    ids.append(page.id)
        return out

    # -- the owner's jobs ----------------------------------------------------------------------

    @cached_property
    def jobs(self) -> list[Job]:
        out = []
        for ref in self.facts_of(SELF_ID, history=True):
            f = ref.fact
            if f.p != "works_at" or (ref.history and f.by):
                continue
            org = link_target(f.o)
            if org is None:  # a literal employer that names exactly one organisation page
                named = [
                    i for i in self.name_index.get(fold(f.o), []) if self.entities[i].type == "org"
                ]
                org = named[0] if len(named) == 1 else None
            org = org if org in self.entities else None
            current = not ref.history and self.valid_now(f)
            name = self.title(org) if org else self.readable(f.o)
            out.append(Job(org, name, f.valid_from, f.valid_to, current, f.id))
        out.sort(key=lambda j: (not j.current, -(int((j.start or "0")[:4]))))
        return out

    # -- sources -------------------------------------------------------------------------------

    @cached_property
    def episodes(self) -> dict[str, Episode]:
        return {e.id: e for e in self.vault.episodes()}

    def source_text(self, src: str) -> str | None:
        """What a source said: an episode's summary, or a raw document's text."""
        if src in self._texts:
            return self._texts[src]
        text: str | None = None
        if episode := self.episodes.get(src):
            text = "\n\n".join(x for x in (episode.gist, episode.summary) if x)
        elif src.startswith("raw/"):
            path = self.vault.root / src
            if path.is_file():
                text = _read_doc(path)
        self._texts[src] = text
        return text

    def sources_about(self, entity: str) -> list[str]:
        """Sources that speak about an entity: its facts', the facts linking to it, and the
        episodes listing it; most cited first."""
        counts: dict[str, int] = {}
        refs = [*self.facts_of(entity, history=True), *self.incoming.get(entity, [])]
        for ref in refs:
            for s in ref.fact.src:
                counts[s] = counts.get(s, 0) + 1
        for episode in self.episodes.values():
            if entity in episode.entities:
                counts[episode.id] = counts.get(episode.id, 0) + 1
        return sorted(counts, key=lambda s: -counts[s])

    def excerpt(
        self, sources: Iterable[str], names: Sequence[str], *, cap: int = EXCERPT_CHARS
    ) -> tuple[str, list[str]]:
        """The sentences of up to ``MAX_SOURCES`` sources naming any of ``names`` (``""`` when
        none does), and the sources used."""
        parts: list[str] = []
        used: list[str] = []
        names = [n for n in names if n.strip()]
        for src in sources:
            text = self.source_text(src)
            hits = [u for u in units(text or "") if names and _names_in(u, names)]
            if not hits:
                continue
            size = 0
            for unit in hits:
                if size and size + len(unit) > cap * min(len(names), 3):
                    break
                if unit not in parts:
                    parts.append(unit)
                    size += len(unit)
            used.append(src)
            if len(used) >= MAX_SOURCES:
                break
        return "\n".join(f"- {p}" for p in parts), used

    def describe(self, page: Page, *, facts: int = MAX_FACT_LINES) -> str:
        """A short profile: names, the summary's first sentence, own facts, how others mention
        it. What a model needs to tell what the page is about."""
        lines = [f"Name: {page.title}"]
        if page.aliases:
            lines.append("Also called: " + ", ".join(page.aliases))
        if page.summary:
            lines.append("About: " + first_sentence(self.readable(page.summary), 200))
        own = [self.readable(r.fact.text) for r in self.facts_of(page.subject)][:facts]
        if own:
            lines.append("Facts:")
            lines += [f"- {t}" for t in own]
        mentions = []
        for ref in self.incoming.get(page.subject, [])[:facts]:
            who = self.title(ref.subject)
            when = " (ended)" if ref.history else ""
            mentions.append(f"- {who}: {self.readable(ref.fact.text)}{when}")
        if mentions:
            lines.append("Mentioned in:")
            lines += mentions
        return "\n".join(lines)


def _read_doc(path: Path) -> str | None:
    try:
        from brain.ingest.readers import load_document

        return load_document(path).text
    except Exception as err:  # noqa: BLE001 - an unreadable source is just no evidence
        log.debug("lint: cannot read %s: %s", path, err)
        return None


def _names_in(text: str, names: Sequence[str]) -> bool:
    for name in names:
        pattern = r"\s+".join(map(re.escape, name.split()))
        if pattern and re.search(rf"(?<!\w){pattern}(?!\w)", text, re.I):
            return True
    return False


LIST_ITEM = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
MARKUP = re.compile(r"[*_`#>]+")


def _plain(text: str) -> str:
    return " ".join(MARKUP.sub("", text).split())


def units(text: str) -> list[str]:
    """Sentences of ``text``; a list item comes with the line that introduces it ("**Fabrikam
    Analytics**, Lisbon — Analytics engineer › Built demand forecasting dashboards"), so a CV
    duty keeps its employer."""
    out: list[str] = []
    context = ""
    for block in re.split(r"\n\s*\n", text):
        para: list[str] = []
        for line in (ln.strip() for ln in block.splitlines()):
            if not line:
                continue
            if LIST_ITEM.match(line):
                if para:
                    context = _plain(" ".join(para))
                    out += SENTENCE_END.split(context)
                    para = []
                item = _plain(LIST_ITEM.sub("", line))
                out.append(f"{context} › {item}" if context else item)
            else:
                para.append(line)
        if para:
            joined = _plain(" ".join(para))
            out += SENTENCE_END.split(joined)
            context = joined if len(joined) <= 160 else ""
    return [u for u in out if u.strip()]


def names_in(text: str, names: Sequence[str]) -> bool:
    """Whether any of ``names`` occurs in ``text`` as whole words."""
    return _names_in(text, names)
