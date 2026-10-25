"""Derived SQLite index (FTS5 + sqlite-vec + graph tables), rebuilt from the vault."""

from brain.index.db import connect, ensure_schema, open_access, open_index
from brain.index.reindex import ReindexStats, VaultSource, reindex
from brain.index.search import Hit, matched_entities, search

__all__ = [
    "Hit",
    "ReindexStats",
    "VaultSource",
    "connect",
    "ensure_schema",
    "matched_entities",
    "open_access",
    "open_index",
    "reindex",
    "search",
]
