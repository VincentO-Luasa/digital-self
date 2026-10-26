"""Raw capture: immutable copies of ingested documents and chat transcripts under ``raw/``.

``raw/`` is episodic ground truth: files are created exclusively and never overwritten. A
transcript is the one file that grows, turn by turn while its session runs, and every turn is
flushed to disk so a crash loses nothing; the next vault commit picks it up.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from brain.vault import gitops
from brain.vault.parse import split_frontmatter
from brain.vault.render import render_frontmatter
from brain.vault.store import Vault

Role = Literal["user", "assistant", "system"]

TURN_HEADING = re.compile(r"^## (\d+) · (user|assistant|system) · (\S+)$")
_ESCAPED_HEADING = re.compile(r"^\\(## \d+ · (?:user|assistant|system) · \S+)$")
_NON_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 60) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = _NON_SLUG.sub("-", ascii_text.casefold()).strip("-")
    return slug[:max_len].strip("-") or "untitled"


def rel_path(vault: Vault, path: Path) -> str:
    """``raw/docs/...`` relative to the vault root, with forward slashes (used as ``src::``)."""
    return path.resolve().relative_to(vault.root.resolve()).as_posix()


def _create_exclusive(directory: Path, stem: str, suffix: str, data: bytes) -> Path:
    """Write ``stem<suffix>`` (or ``stem-2<suffix>``, ...) without ever replacing a file."""
    directory.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        path = directory / (f"{stem}{suffix}" if n == 1 else f"{stem}-{n}{suffix}")
        try:
            with path.open("xb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            return path
        except FileExistsError:
            n += 1


# --- documents ---------------------------------------------------------------------------------


def docs_dir(vault: Vault) -> Path:
    return vault.raw_dir / "docs"


def find_doc(vault: Vault, data: bytes) -> Path | None:
    """An existing raw copy with exactly these bytes (re-ingesting reuses it)."""
    base = docs_dir(vault)
    if not base.is_dir():
        return None
    for path in sorted(base.iterdir()):
        if path.is_file() and path.stat().st_size == len(data) and path.read_bytes() == data:
            return path
    return None


def save_doc(
    vault: Vault, source: Path, *, today: date | None = None, commit: bool = True
) -> tuple[Path, bool]:
    """Copy ``source`` to ``raw/docs/YYYY-MM-DD-<slug>.<ext>``; return (path, created).

    An identical copy already in ``raw/docs/`` is reused instead of duplicated.
    """
    data = Path(source).read_bytes()
    if existing := find_doc(vault, data):
        return existing, False
    today = today or date.today()
    suffix = Path(source).suffix.lower() or ".txt"
    stem = f"{today.isoformat()}-{slugify(Path(source).stem)}"
    path = _create_exclusive(docs_dir(vault), stem, suffix, data)
    if commit:
        gitops.commit_all(vault.root, f"brain: raw doc {rel_path(vault, path)}")
    return path, True


def save_doc_bytes(
    vault: Vault,
    data: bytes,
    name: str,
    suffix: str,
    *,
    today: date | None = None,
    commit: bool = True,
) -> tuple[Path, bool]:
    """``save_doc`` for content that is not a local file (a fetched web page): saved as
    ``raw/docs/YYYY-MM-DD-<slug of name><suffix>``; an identical copy is reused."""
    if existing := find_doc(vault, data):
        return existing, False
    today = today or date.today()
    suffix = suffix.lower() if suffix.startswith(".") else f".{suffix.lower() or 'txt'}"
    path = _create_exclusive(docs_dir(vault), f"{today.isoformat()}-{slugify(name)}", suffix, data)
    if commit:
        gitops.commit_all(vault.root, f"brain: raw doc {rel_path(vault, path)}")
    return path, True


# --- conversations -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    n: int
    role: Role
    at: datetime
    text: str


def _stamp(at: datetime) -> str:
    return at.replace(microsecond=0).isoformat()


def _escape(text: str) -> str:
    """Keep message lines from looking like turn headings."""
    lines = text.split("\n")
    return "\n".join("\\" + line if TURN_HEADING.match(line) else line for line in lines)


def _unescape(text: str) -> str:
    return "\n".join(_ESCAPED_HEADING.sub(r"\1", line) for line in text.split("\n"))


def render_turn(turn: Turn) -> str:
    body = _escape(turn.text.strip("\n"))
    return f"\n## {turn.n} · {turn.role} · {_stamp(turn.at)}\n\n{body}\n"


class Transcript:
    """A chat or interview transcript at ``raw/conversations/YYYY/MM/<session-id>.md``."""

    def __init__(self, vault: Vault, path: Path, session_id: str, turns: int = 0) -> None:
        self.vault = vault
        self.path = path
        self.session_id = session_id
        self.turns = turns

    @classmethod
    def start(
        cls,
        vault: Vault,
        *,
        channel: str = "chat",
        now: datetime | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Transcript:
        now = (now or datetime.now()).replace(microsecond=0)
        base = f"conv-{now:%Y%m%d-%H%M}"
        directory = vault.raw_dir / "conversations" / now.strftime("%Y/%m")
        directory.mkdir(parents=True, exist_ok=True)
        n = 1
        while True:
            session_id = base if n == 1 else f"{base}-{n}"
            path = directory / f"{session_id}.md"
            fm = [
                ("id", session_id),
                ("type", "transcript"),
                ("channel", channel),
                ("started", now),
                *(meta or {}).items(),
            ]
            header = render_frontmatter(fm) + f"# Conversation {session_id}\n"
            try:
                with path.open("x", encoding="utf-8") as fh:
                    fh.write(header)
                    fh.flush()
                    os.fsync(fh.fileno())
                return cls(vault, path, session_id)
            except FileExistsError:
                n += 1

    @classmethod
    def open(cls, vault: Vault, path: Path | str) -> Transcript:
        """An existing transcript (absolute, or relative to the vault root)."""
        path = Path(path)
        if not path.is_absolute():
            path = vault.root / path
        fm, turns = parse_transcript(path.read_text(encoding="utf-8"))
        session_id = str(fm.get("id") or path.stem)
        return cls(vault, path, session_id, turns=max((t.n for t in turns), default=0))

    def read(self) -> tuple[dict[str, Any], list[Turn]]:
        """Frontmatter and turns as written so far."""
        return parse_transcript(self.path.read_text(encoding="utf-8"))

    @property
    def rel(self) -> str:
        return rel_path(self.vault, self.path)

    def append(self, role: Role, text: str, *, at: datetime | None = None) -> int:
        """Append one turn and flush it to disk; return its 1-based turn number."""
        self.turns += 1
        turn = Turn(self.turns, role, at or datetime.now(), text)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(render_turn(turn))
            fh.flush()
            os.fsync(fh.fileno())
        return self.turns

    def commit(self, message: str | None = None) -> str | None:
        return gitops.commit_all(self.vault.root, message or f"brain: transcript {self.rel}")


def parse_transcript(text: str) -> tuple[dict[str, Any], list[Turn]]:
    """Frontmatter and turns of a transcript written by ``Transcript``."""
    fm, body, _ = split_frontmatter(text)
    turns: list[Turn] = []
    current: tuple[int, Role, datetime] | None = None
    lines: list[str] = []

    def flush() -> None:
        if current:
            turns.append(Turn(*current, _unescape("\n".join(lines).strip("\n"))))

    for line in body.split("\n"):
        if m := TURN_HEADING.match(line):
            flush()
            current = (int(m.group(1)), m.group(2), datetime.fromisoformat(m.group(3)))  # type: ignore[assignment]
            lines = []
        elif current:
            lines.append(line)
    flush()
    return fm or {}, turns
