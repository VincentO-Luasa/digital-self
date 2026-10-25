from __future__ import annotations

import pytest

from brain.index.reindex import load_private_regions, region_is_private, reindex
from brain.llm.fake import FakeLLM
from brain.vault.models import Fact


def run(vault, db, llm, **kw):
    return reindex(
        vault, db, llm, private_regions=frozenset({"relationships.friends", "self.health"}), **kw
    )


def count(db, sql: str, *params) -> int:
    return db.execute(sql, params).fetchone()[0]


def item(db, item_id: str):
    return db.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()


def test_first_run_indexes_everything(vault, db, llm):
    stats = run(vault, db, llm)
    # 3 summaries + 6 current facts + 1 history fact + 3 episodes + 2 observations
    assert stats.items == 15
    assert stats.added == stats.embedded == 15
    assert count(db, "SELECT count(*) FROM items") == 15
    assert count(db, "SELECT count(*) FROM vec_items") == 15
    assert count(db, "SELECT count(*) FROM entities") == 4
    assert count(db, "SELECT count(*) FROM facts") == 7
    assert len(llm.embedded) == 15


def test_graph_tables(vault, db, llm):
    run(vault, db, llm)
    hist = db.execute("SELECT current, valid_to FROM facts WHERE id = 'f-city00'").fetchone()
    assert tuple(hist) == (0, "2024-05")
    work = db.execute("SELECT subject, o_entity FROM facts WHERE id = 'f-work01'").fetchone()
    assert tuple(work) == ("self", "orgs/northwind-labs")
    links = {tuple(r) for r in db.execute("SELECT src_id, dst_id, kind FROM links")}
    assert ("self", "orgs/northwind-labs", "fact") in links
    assert ("self/profile", "orgs/northwind-labs", "wikilink") in links
    assert ("ep-20261025-1412-career", "projects/atlas-migration", "mention") in links
    assert ("o-anx001", "projects/atlas-migration", "mention") in links
    assert ("o-clm001", "ep-20261020-1900-climbing", "source") in links
    aliases = {tuple(r) for r in db.execute("SELECT alias_cf, entity_id FROM aliases")}
    assert ("sam", "people/sam-okafor") in aliases
    assert ("sam okafor", "people/sam-okafor") in aliases


def test_item_text_is_readable(vault, db, llm):
    run(vault, db, llm)
    assert item(db, "f-work01")["text"] == (
        "Works at Northwind Labs as data engineer (since 2023-02)"
    )
    assert item(db, "f-sam001")["text"].startswith("Sam Okafor: Plays the cello")
    assert item(db, "o-anx001")["text"] == "Alex is anxious about the Atlas Migration deadline"
    assert item(db, "o-anx001")["entity_id"] == "projects/atlas-migration"
    ep = item(db, "ep-20261025-1412-career")
    assert ep["text"].splitlines()[0].startswith("Alex worries")
    assert ep["importance"] == 6


def test_unchanged_vault_embeds_nothing(vault, db, llm):
    run(vault, db, llm)
    before = len(llm.embedded)
    stats = run(vault, db, llm)
    assert stats.embedded == stats.added == stats.changed == stats.deleted == 0
    assert len(llm.embedded) == before


def test_edited_fact_reembeds_only_that_item(vault, db, llm):
    run(vault, db, llm)
    pk = item(db, "f-sam001")["pk"]
    sam = vault.page("people/sam-okafor")
    sam.facts[0] = sam.facts[0].model_copy(update={"text": "Plays the viola in a string quartet"})
    llm.embedded.clear()
    stats = run(vault, db, llm)
    assert stats.embedded == stats.changed == 1
    assert llm.embedded == ["Sam Okafor: Plays the viola in a string quartet"]
    assert item(db, "f-sam001")["pk"] == pk
    assert count(db, "SELECT count(*) FROM items_fts WHERE items_fts MATCH 'viola'") == 1
    assert count(db, "SELECT count(*) FROM items_fts WHERE items_fts MATCH 'cello'") == 0
    assert count(db, "SELECT count(*) FROM vec_items") == 15


def test_deleted_page_removes_rows(vault, db, llm):
    run(vault, db, llm)
    vault.page_list = [p for p in vault.page_list if p.id != "projects/atlas-migration"]
    stats = run(vault, db, llm)
    assert stats.deleted == 2  # its two facts (it has no summary)
    # the observation linking to it now renders the raw link target instead of its title
    assert stats.embedded == stats.changed == 1
    assert (
        item(db, "o-anx001")["text"]
        == "Alex is anxious about the projects/atlas-migration deadline"
    )
    assert item(db, "f-atl001") is None
    assert count(db, "SELECT count(*) FROM facts WHERE page_id = 'projects/atlas-migration'") == 0
    assert count(db, "SELECT count(*) FROM entities WHERE id = 'projects/atlas-migration'") == 0
    assert count(db, "SELECT count(*) FROM aliases WHERE alias_cf = 'atlas'") == 0
    assert count(db, "SELECT count(*) FROM vec_items") == 13
    assert count(db, "SELECT count(*) FROM items_fts WHERE items_fts MATCH 'january'") == 0


def test_privacy_propagation(vault, db, llm):
    run(vault, db, llm)
    privacy = {r["id"]: r["privacy"] for r in db.execute("SELECT id, privacy FROM items")}
    # private page → its summary and facts
    assert privacy["summary:people/sam-okafor"] == "private"
    assert privacy["f-sam001"] == privacy["f-sam002"] == "private"
    # episode mentioning a private entity, episode in a private region
    assert privacy["ep-20261020-1900-climbing"] == "private"
    assert privacy["ep-20261018-0900-checkup"] == "private"
    # observation sourced from a private episode
    assert privacy["o-clm001"] == "private"
    # everything else stays normal
    for key in ("f-work01", "summary:orgs/northwind-labs", "ep-20261025-1412-career", "o-anx001"):
        assert privacy[key] == "normal"


def test_privacy_change_updates_without_reembedding(vault, db, llm):
    run(vault, db, llm)
    org = vault.page("orgs/northwind-labs")
    vault.page_list[vault.page_list.index(org)] = org.model_copy(update={"privacy": "private"})
    stats = run(vault, db, llm)
    assert stats.embedded == 0
    assert stats.updated >= 1
    assert item(db, "summary:orgs/northwind-labs")["privacy"] == "private"
    # the career episode mentions the now-private org
    assert item(db, "ep-20261025-1412-career")["privacy"] == "private"


def test_new_fact_added(vault, db, llm):
    run(vault, db, llm)
    atlas = vault.page("projects/atlas-migration")
    atlas.facts.append(
        Fact(
            id="f-atl003",
            text="Status is in testing",
            p="status",
            o="testing",
            rec=atlas.created,
            src=["ep-20261025-1412-career"],
            kind="stated",
        )
    )
    stats = run(vault, db, llm)
    assert stats.added == stats.embedded == 1


def test_full_reindex_reembeds_everything(vault, db, llm):
    run(vault, db, llm)
    stats = run(vault, db, llm, full=True)
    assert stats.embedded == 15
    assert stats.added == 0
    assert count(db, "SELECT count(*) FROM vec_items") == 15


def test_embed_model_change_triggers_reembed(vault, db, llm):
    run(vault, db, llm)
    other = FakeLLM(dim=32)
    other.settings = other.settings.model_copy(update={"embed_model": "other-embedder"})
    stats = run(vault, db, other)
    assert stats.vectors_reset
    assert stats.embedded == 15
    assert db.execute("SELECT embed_model, dim FROM meta").fetchone()[:] == ("other-embedder", 32)


def test_missing_vectors_are_healed(vault, db, llm):
    run(vault, db, llm)
    db.execute("DELETE FROM vec_items")
    stats = run(vault, db, llm)
    assert stats.embedded == 15


def test_embed_failure_leaves_index_untouched(vault, db, llm):
    run(vault, db, llm)
    vault.page_list = vault.page_list[:1]

    class Broken(FakeLLM):
        def embed(self, texts, *, query=False):
            raise ConnectionError("server down")

    broken = Broken(dim=64)
    vault.page_list[0].facts[0] = vault.page_list[0].facts[0].model_copy(update={"text": "New"})
    with pytest.raises(ConnectionError):
        run(vault, db, broken)
    assert count(db, "SELECT count(*) FROM items") == 15


def test_region_privacy_helpers():
    assert region_is_private("relationships.friends", {"relationships.friends"})
    assert region_is_private("self.health.sleep", {"self.health"})
    assert not region_is_private("self.healthy", {"self.health"})
    regions = load_private_regions()
    assert "self.health" in regions
    assert "career.current_job" not in regions
