"""Read-only access to a vault on disk, with per-file mtime caching."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import cached_property
from pathlib import Path
from typing import Any, TypeVar

from brain.vault.lifemap import LifeMap, load_lifemap
from brain.vault.models import SELF_ID, Episode, Fact, Intention, Observation, Page
from brain.vault.parse import (
    ParseError,
    ParseWarning,
    parse_episode,
    parse_intentions,
    parse_journal,
    parse_page,
)

T = TypeVar("T")
Parser = Callable[..., tuple[T, list[ParseWarning]]]

SELF_PAGE = "self/profile"
WIKI_DIRS = ("self", "people", "orgs", "projects", "places", "topics", "events")


class Vault:
    """A vault directory. Cheap to construct; files are parsed lazily and cached by mtime.

    Returned models are shared with the cache: copy them before mutating.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser()
        self._cache: dict[Path, tuple[tuple[int, int], Any, list[ParseWarning]]] = {}
        self._errors: dict[Path, ParseWarning] = {}

    def __repr__(self) -> str:
        return f"Vault({str(self.root)!r})"

    # --- paths ---------------------------------------------------------------------------------

    @property
    def wiki_dir(self) -> Path:
        return self.root / "wiki"

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def episodes_dir(self) -> Path:
        return self.root / "episodes"

    @property
    def journal_dir(self) -> Path:
        return self.root / "journal"

    @property
    def reviews_pending_dir(self) -> Path:
        return self.root / "reviews" / "pending"

    @property
    def reviews_done_dir(self) -> Path:
        return self.root / "reviews" / "done"

    @property
    def brain_dir(self) -> Path:
        return self.root / ".brain"

    @property
    def db_path(self) -> Path:
        """The derived SQLite index (not to be confused with the wiki's ``index.md``)."""
        return self.brain_dir / "index.sqlite"

    @property
    def denylist_path(self) -> Path:
        return self.root / ".privacy" / "denylist.txt"

    @property
    def intentions_path(self) -> Path:
        return self.root / "intentions.md"

    @property
    def log_path(self) -> Path:
        return self.root / "log.md"

    @property
    def core_path(self) -> Path:
        return self.root / "core.md"

    @property
    def index_path(self) -> Path:
        return self.root / "index.md"

    @property
    def questions_path(self) -> Path:
        return self.root / "questions.md"

    @property
    def schema_path(self) -> Path:
        return self.root / "SCHEMA.md"

    def page_path(self, page_id: str) -> Path:
        return self.wiki_dir / f"{page_id}.md"

    def journal_path(self, month: str) -> Path:
        """``month`` is ``YYYY-MM``."""
        return self.journal_dir / f"{month}.md"

    def episode_path(self, episode: Episode) -> Path:
        return self.episodes_dir / episode.started.strftime("%Y/%m") / f"{episode.id}.md"

    @cached_property
    def lifemap(self) -> LifeMap:
        return load_lifemap(self.root)

    # --- loading -------------------------------------------------------------------------------

    def _load(self, path: Path, parser: Parser[T]) -> T | None:
        try:
            st = path.stat()
        except FileNotFoundError:
            return None
        key = (st.st_mtime_ns, st.st_size)
        cached = self._cache.get(path)
        if cached and cached[0] == key:
            return cached[1]
        self._errors.pop(path, None)
        try:
            value, warnings = parser(path.read_text(encoding="utf-8"), path=path)
        except (ParseError, UnicodeDecodeError) as exc:
            self._errors[path] = ParseWarning(path, 0, f"skipped: {exc}")
            self._cache.pop(path, None)
            return None
        self._cache[path] = (key, value, warnings)
        return value

    def _md_files(self, base: Path) -> Iterator[Path]:
        if base.is_dir():
            yield from sorted(p for p in base.rglob("*.md") if p.is_file())

    def pages(self) -> list[Page]:
        loaded = (self._load(p, parse_page) for p in self._md_files(self.wiki_dir))
        return [p for p in loaded if p is not None]

    def page(self, page_id: str) -> Page | None:
        """Page by id; ``self`` resolves to ``self/profile``."""
        return self._load(self.page_path(self.page_id_for(page_id)), parse_page)

    @staticmethod
    def page_id_for(entity: str) -> str:
        return SELF_PAGE if entity == SELF_ID else entity

    def pages_about(self, subject: str) -> list[Page]:
        return [p for p in self.pages() if p.subject == subject]

    def episodes(self) -> list[Episode]:
        loaded = (self._load(p, parse_episode) for p in self._md_files(self.episodes_dir))
        return sorted((e for e in loaded if e is not None), key=lambda e: (e.started, e.id))

    def episode(self, episode_id: str) -> Episode | None:
        return next((e for e in self.episodes() if e.id == episode_id), None)

    def observations(self) -> list[Observation]:
        out: list[Observation] = []
        for path in self._md_files(self.journal_dir):
            out.extend(self._load(path, parse_journal) or [])
        return out

    def intentions(self) -> list[Intention]:
        return self._load(self.intentions_path, parse_intentions) or []

    # --- lookups -------------------------------------------------------------------------------

    def alias_index(self) -> dict[str, list[str]]:
        """Casefolded titles and aliases → entity ids (page subjects)."""
        index: dict[str, set[str]] = {}
        for page in self.pages():
            for name in (page.title, *page.aliases):
                if key := name.strip().casefold():
                    index.setdefault(key, set()).add(page.subject)
        return {k: sorted(v) for k, v in sorted(index.items())}

    def find_fact(self, fact_id: str) -> tuple[Page, Fact, bool] | None:
        """``(page, fact, in_history)`` for a fact block id, or None."""
        for page in self.pages():
            for fact in page.facts:
                if fact.id == fact_id:
                    return page, fact, False
            for fact in page.history:
                if fact.id == fact_id:
                    return page, fact, True
        return None

    def warnings(self) -> list[ParseWarning]:
        """Parse all files and return every warning (for ``brain lint``)."""
        self.pages()
        self.episodes()
        self.observations()
        self.intentions()
        out = [w for path, (_, _, ws) in self._cache.items() if path.exists() for w in ws]
        return out + [w for path, w in self._errors.items() if path.exists()]
