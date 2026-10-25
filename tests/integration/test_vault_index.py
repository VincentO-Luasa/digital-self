"""The index built from a real ``Vault`` (fixture persona), end to end with the fake model."""

from __future__ import annotations

from brain.index.db import open_index
from brain.index.reindex import reindex
from brain.index.search import search
from brain.llm.fake import FakeLLM
from brain.vault import ChangeSet, apply
from brain.vault.ops import AddFact
from brain.vault.store import Vault


def test_reindex_and_search_real_vault(tmp_vault: Vault) -> None:
    llm = FakeLLM()
    db = open_index(tmp_vault.db_path, embed_model=llm.embed_model, dim=llm.embed_dim)
    stats = reindex(tmp_vault, db, llm)
    assert stats.embedded > 0
    assert not tmp_vault.warnings()

    hits = search(db, llm, "Atlas Migration", k=5)
    assert hits
    assert any("atlas" in ((h.entity_id or "") + h.text).lower() for h in hits)


def test_private_pages_hidden_unless_requested(tmp_vault: Vault) -> None:
    llm = FakeLLM()
    db = open_index(tmp_vault.db_path, embed_model=llm.embed_model, dim=llm.embed_dim)
    reindex(tmp_vault, db, llm)
    public = search(db, llm, "Sam Okafor friend", k=20, include_private=False)
    private = search(db, llm, "Sam Okafor friend", k=20, include_private=True)
    assert all(h.privacy != "private" for h in public)
    assert any(h.privacy == "private" for h in private)
    assert len(private) >= len(public)


def test_applied_fact_reaches_index(tmp_vault: Vault) -> None:
    llm = FakeLLM()
    db = open_index(tmp_vault.db_path, embed_model=llm.embed_model, dim=llm.embed_dim)
    reindex(tmp_vault, db, llm)
    cs = ChangeSet(
        ops=[
            AddFact(
                subject="self",
                p="interest",
                o="bouldering",
                text="Alex took up bouldering",
                kind="stated",
                **{"from": "2026-09"},
            )
        ],
        origin="user",
        src=["raw/conversations/2026-10-25-test.md"],
    )
    result = apply(cs, tmp_vault, review=False)
    assert result.applied
    stats = reindex(tmp_vault, db, llm)
    assert stats.embedded >= 1
    hits = search(db, llm, "bouldering", k=5, include_private=True)
    assert any("bouldering" in h.text.lower() for h in hits)
