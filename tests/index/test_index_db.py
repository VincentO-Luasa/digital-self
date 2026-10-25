from __future__ import annotations

import numpy as np

from brain.index.db import (
    SCHEMA_VERSION,
    access_path,
    connect,
    ensure_schema,
    open_access,
    read_meta,
    vector_blob,
)
from brain.index.temporal import period_end, period_start, valid_at


def tables(conn) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}


def test_schema_creation(db):
    assert ensure_schema(db, embed_model="emb-a", dim=8) is True
    names = tables(db)
    for t in ("meta", "entities", "aliases", "facts", "links", "items", "items_fts", "vec_items"):
        assert t in names
    assert "access" not in names  # the access log lives in its own file
    assert read_meta(db).schema_version == SCHEMA_VERSION
    assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert db.execute("SELECT vec_version()").fetchone()[0].startswith("v")


def test_ensure_schema_is_idempotent(db):
    ensure_schema(db, embed_model="emb-a", dim=8)
    assert ensure_schema(db, embed_model="emb-a", dim=8) is False


def test_model_change_drops_vectors_only(db):
    ensure_schema(db, embed_model="emb-a", dim=4)
    db.execute(
        "INSERT INTO items (id, kind, text, privacy, hash) VALUES ('x', 'fact', 't', 'normal', 'h')"
    )
    db.execute("INSERT INTO vec_items (rowid, embedding) VALUES (1, ?)", (vector_blob(np.ones(4)),))
    assert ensure_schema(db, embed_model="emb-b", dim=8) is True
    assert db.execute("SELECT count(*) FROM vec_items").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM items").fetchone()[0] == 1
    assert read_meta(db).dim == 8
    db.execute("INSERT INTO vec_items (rowid, embedding) VALUES (1, ?)", (vector_blob(np.ones(8)),))


def test_schema_version_change_rebuilds(db):
    ensure_schema(db, embed_model="emb-a", dim=4)
    db.execute(
        "INSERT INTO items (id, kind, text, privacy, hash) VALUES ('x', 'fact', 't', 'normal', 'h')"
    )
    db.execute("UPDATE meta SET schema_version = 0")
    assert ensure_schema(db, embed_model="emb-a", dim=4) is True
    assert db.execute("SELECT count(*) FROM items").fetchone()[0] == 0


def test_fts_triggers_follow_items(db):
    ensure_schema(db, embed_model="emb-a", dim=4)
    db.execute(
        "INSERT INTO items (id, kind, text, privacy, hash) "
        "VALUES ('x', 'fact', 'Café near the office', 'normal', 'h')"
    )
    match = "SELECT count(*) FROM items_fts WHERE items_fts MATCH ?"
    assert db.execute(match, ("cafe",)).fetchone()[0] == 1  # diacritics removed
    db.execute("UPDATE items SET text = 'Tea shop' WHERE id = 'x'")
    assert db.execute(match, ("cafe",)).fetchone()[0] == 0
    assert db.execute(match, ("tea",)).fetchone()[0] == 1
    db.execute("DELETE FROM items")
    assert db.execute(match, ("tea",)).fetchone()[0] == 0


def test_access_log_separate_file(tmp_path):
    index = tmp_path / ".brain" / "index.sqlite"
    conn = open_access(access_path(index))
    assert "access" in tables(conn)
    assert access_path(index).name == "access.sqlite"
    conn.close()


def test_valid_at_function_registered(tmp_path):
    conn = connect(tmp_path / "x.sqlite")
    row = conn.execute("SELECT brain_valid_at('2019', '2024-05', '2020-06')").fetchone()
    assert row[0] == 1
    conn.close()


def test_partial_date_periods():
    assert period_start("2023") == "2023-01-01"
    assert period_end("2023") == "2023-12-31"
    assert period_start("2023-02") == "2023-02-01"
    assert period_end("2023-02") == "2023-02-31"
    assert period_end("2023-02-14") == "2023-02-14"


def test_valid_at_intervals():
    assert valid_at("2019", "2024-05", "2020")
    assert valid_at("2019", "2024-05", "2024-05-20")  # inclusive at month precision
    assert not valid_at("2019", "2024-05", "2024-06")
    assert not valid_at("2024-05", None, "2024-04-30")
    assert valid_at("2024-05", None, "2024")  # as-of year overlaps the start month
    assert valid_at(None, None, "1990")
    assert not valid_at(None, "2018", "2019-01-01")
