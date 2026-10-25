"""raw/ capture: immutable document copies and turn-by-turn transcripts."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from brain.vault import gitops
from brain.vault.raw import Transcript, parse_transcript, rel_path, save_doc, slugify
from brain.vault.store import Vault

TODAY = date(2026, 10, 25)


def test_slugify() -> None:
    assert slugify("Alex Rivera — CV (2026).PDF") == "alex-rivera-cv-2026-pdf"
    assert slugify("Ünïcodé  notes") == "unicode-notes"
    assert slugify("!!!") == "untitled"
    assert len(slugify("x" * 200)) == 60


def test_save_doc_copies_and_commits(tmp_vault: Vault, tmp_path: Path) -> None:
    src = tmp_path / "Alex Rivera CV.md"
    src.write_text("# Alex Rivera\n\nData engineer.\n")
    path, created = save_doc(tmp_vault, src, today=TODAY)
    assert created
    assert rel_path(tmp_vault, path) == "raw/docs/2026-10-25-alex-rivera-cv.md"
    assert path.read_bytes() == src.read_bytes()
    assert not gitops.has_staged_changes(tmp_vault.root)
    log = gitops._git(tmp_vault.root, "log", "--oneline").stdout
    assert "raw doc raw/docs/2026-10-25-alex-rivera-cv.md" in log


def test_save_doc_never_overwrites_and_reuses_identical(tmp_vault: Vault, tmp_path: Path) -> None:
    src = tmp_path / "cv.md"
    src.write_text("first version\n")
    first, _ = save_doc(tmp_vault, src, today=TODAY, commit=False)
    again, created = save_doc(tmp_vault, src, today=date(2026, 11, 2), commit=False)
    assert again == first and not created

    src.write_text("second version\n")
    second, created = save_doc(tmp_vault, src, today=TODAY, commit=False)
    assert created and second != first
    assert second.name == "2026-10-25-cv-2.md"
    assert first.read_text() == "first version\n"


def test_transcript_appends_turn_by_turn(tmp_vault: Vault) -> None:
    started = datetime(2026, 10, 25, 14, 0, 5)
    t = Transcript.start(tmp_vault, now=started, meta={"model": "test-model"})
    assert t.session_id == "conv-20261025-1400"
    assert t.rel == "raw/conversations/2026/10/conv-20261025-1400.md"
    assert t.append("user", "Where do I work?", at=started) == 1
    # each turn is on disk as soon as it is appended
    fm, turns = parse_transcript(t.path.read_text())
    assert fm["id"] == "conv-20261025-1400" and fm["model"] == "test-model"
    assert [x.text for x in turns] == ["Where do I work?"]

    tricky = "Sure.\n## 9 · user · 2026-10-25T14:00:00\nnot a heading\n\n- list"
    t.append("assistant", tricky, at=datetime(2026, 10, 25, 14, 0, 9))
    _, turns = parse_transcript(t.path.read_text())
    assert [(x.n, x.role) for x in turns] == [(1, "user"), (2, "assistant")]
    assert turns[1].text == tricky
    assert turns[1].at == datetime(2026, 10, 25, 14, 0, 9)

    assert t.commit()
    assert not gitops.has_staged_changes(tmp_vault.root)


def test_transcript_ids_never_collide(tmp_vault: Vault) -> None:
    now = datetime(2026, 10, 25, 9, 30)
    a = Transcript.start(tmp_vault, now=now)
    b = Transcript.start(tmp_vault, now=now)
    assert (a.session_id, b.session_id) == ("conv-20261025-0930", "conv-20261025-0930-2")
    assert a.path.exists() and b.path.exists()
