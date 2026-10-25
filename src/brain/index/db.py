"""SQLite index: derived from the vault, rebuilt by ``brain reindex``.

``items`` holds every retrievable unit (fact line, page summary, episode, observation). FTS5 and
the sqlite-vec table are keyed by ``items.pk``. The access log is *not* derivable from the vault,
so it lives in a separate file (``.brain/access.sqlite``, see ``open_access``) that resets and
full reindexes never touch.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sqlite_vec

from brain.index.temporal import sql_valid_at

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

TABLES = """
CREATE TABLE IF NOT EXISTS meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    schema_version INTEGER NOT NULL,
    embed_model TEXT NOT NULL,
    dim INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    subject TEXT NOT NULL,
    title TEXT NOT NULL,
    privacy TEXT NOT NULL,
    regions_json TEXT NOT NULL,
    path TEXT NOT NULL,
    mtime REAL
);
CREATE TABLE IF NOT EXISTS aliases (
    alias_cf TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY (alias_cf, entity_id)
);
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    p TEXT NOT NULL,
    o TEXT NOT NULL,
    o_entity TEXT,
    text TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    rec TEXT NOT NULL,
    kind TEXT NOT NULL,
    src_json TEXT NOT NULL,
    current INTEGER NOT NULL,
    page_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS facts_subject ON facts (subject, p);
CREATE INDEX IF NOT EXISTS facts_object ON facts (o_entity);
CREATE TABLE IF NOT EXISTS links (
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    PRIMARY KEY (src_id, dst_id, kind)
);
CREATE INDEX IF NOT EXISTS links_dst ON links (dst_id);
CREATE TABLE IF NOT EXISTS items (
    pk INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK (kind IN ('fact', 'summary', 'episode', 'observation')),
    entity_id TEXT,
    text TEXT NOT NULL,
    privacy TEXT NOT NULL,
    importance INTEGER,
    date TEXT,
    src_json TEXT NOT NULL DEFAULT '[]',
    hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS items_entity ON items (entity_id);
CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5 (
    text, content='items', content_rowid='pk', tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS items_ai AFTER INSERT ON items BEGIN
    INSERT INTO items_fts (rowid, text) VALUES (new.pk, new.text);
END;
CREATE TRIGGER IF NOT EXISTS items_ad AFTER DELETE ON items BEGIN
    INSERT INTO items_fts (items_fts, rowid, text) VALUES ('delete', old.pk, old.text);
END;
CREATE TRIGGER IF NOT EXISTS items_au AFTER UPDATE OF text ON items BEGIN
    INSERT INTO items_fts (items_fts, rowid, text) VALUES ('delete', old.pk, old.text);
    INSERT INTO items_fts (rowid, text) VALUES (new.pk, new.text);
END;
"""

DERIVED = ["vec_items", "items_fts", "items", "links", "facts", "aliases", "entities", "meta"]

ACCESS_TABLES = """
CREATE TABLE IF NOT EXISTS access (
    item_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS access_item ON access (item_id, ts);
"""


@dataclass(frozen=True)
class Meta:
    schema_version: int
    embed_model: str
    dim: int


def connect(path: Path | str) -> sqlite3.Connection:
    """Autocommit connection (use ``transaction``) with WAL, sqlite-vec and ``brain_valid_at``."""
    if isinstance(path, Path):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.create_function("brain_valid_at", 3, sql_valid_at, deterministic=True)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def read_meta(conn: sqlite3.Connection) -> Meta | None:
    try:
        row = conn.execute("SELECT schema_version, embed_model, dim FROM meta").fetchone()
    except sqlite3.OperationalError:
        return None
    return Meta(*row) if row else None


def _create_vec(conn: sqlite3.Connection, dim: int) -> None:
    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(embedding float[{dim}])")


def _drop(conn: sqlite3.Connection, names: list[str]) -> None:
    for name in names:
        kind = conn.execute("SELECT type FROM sqlite_master WHERE name = ?", (name,)).fetchone()
        if kind:
            conn.execute(f"DROP {kind[0].upper()} IF EXISTS {name}")


def ensure_schema(conn: sqlite3.Connection, *, embed_model: str, dim: int) -> bool:
    """Create or migrate the schema. Returns True if stored vectors were discarded.

    A schema version change drops every derived table; an embedding model or dimension change
    drops only the vectors, so the next reindex re-embeds everything.
    """
    meta = read_meta(conn)
    reset = meta is None
    if meta and meta.schema_version != SCHEMA_VERSION:
        log.info("index schema %s -> %s: rebuilding", meta.schema_version, SCHEMA_VERSION)
        _drop(conn, DERIVED)
        reset = True
    elif meta and (meta.embed_model, meta.dim) != (embed_model, dim):
        log.info(
            "embedding model changed (%s/%d -> %s/%d): full re-embed",
            meta.embed_model,
            meta.dim,
            embed_model,
            dim,
        )
        _drop(conn, ["vec_items"])
        reset = True
    conn.executescript(TABLES)
    _create_vec(conn, dim)
    conn.execute(
        "INSERT INTO meta (id, schema_version, embed_model, dim) VALUES (1, ?, ?, ?) "
        "ON CONFLICT (id) DO UPDATE SET schema_version = excluded.schema_version, "
        "embed_model = excluded.embed_model, dim = excluded.dim",
        (SCHEMA_VERSION, embed_model, dim),
    )
    return reset


def open_index(path: Path | str, *, embed_model: str, dim: int) -> sqlite3.Connection:
    conn = connect(path)
    ensure_schema(conn, embed_model=embed_model, dim=dim)
    return conn


def open_access(path: Path | str) -> sqlite3.Connection:
    """The access log (``.brain/access.sqlite``): not derivable, so never dropped by reindex."""
    conn = connect(path)
    conn.executescript(ACCESS_TABLES)
    return conn


def access_path(index_path: Path) -> Path:
    return index_path.with_name("access.sqlite")


def vector_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()
