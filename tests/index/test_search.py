from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from brain.config import RetrievalSettings
from brain.index.reindex import reindex
from brain.index.search import (
    AliasMatcher,
    Filters,
    boostable,
    fts_query,
    fts_ranked,
    matched_entities,
    rrf,
    search,
)
from brain.llm.fake import FakeLLM


@pytest.fixture
def indexed(vault, db, llm):
    reindex(vault, db, llm, private_regions=frozenset({"relationships.friends", "self.health"}))
    return db


def ids(hits) -> list[str]:
    return [h.id for h in hits]


def test_matched_entities(indexed):
    assert matched_entities(indexed, "I saw SAM OKAFOR yesterday") == ["people/sam-okafor"]
    assert matched_entities(indexed, "How is Atlas going at Northwind?") == [
        "projects/atlas-migration",
        "orgs/northwind-labs",
    ]
    assert matched_entities(indexed, "Samantha and Atlases") == []  # word boundaries
    assert matched_entities(indexed, "nothing here") == []


def test_alias_matcher_prefers_longest():
    m = AliasMatcher([("sam", "people/a"), ("sam okafor", "people/b")])
    assert m.match("met sam okafor") == ["people/b"]
    assert m.match("met sam") == ["people/a"]
    assert AliasMatcher([]).match("anything") == []


def test_self_is_not_boosted(indexed):
    assert matched_entities(indexed, "Alex said") == ["self"]  # aliases name page subjects
    assert boostable(indexed, matched_entities(indexed, "Alex")) == []
    assert boostable(indexed, ["people/sam-okafor", "self"]) == ["people/sam-okafor"]


def test_fts_query_is_safe():
    assert fts_query('what is "Atlas" (status)?') == '"atlas" OR "status"'
    assert fts_query("is it?") is None


def test_rrf_fuses_ranks():
    scores = rrf([[1, 2, 3], [3, 1]])
    assert max(scores, key=scores.get) == 1
    assert scores[3] > scores[2]


def test_search_finds_lexical_and_vector_match(indexed, llm):
    hits = search(indexed, llm, "who plays the cello?", rerank=False)
    assert hits[0].id == "f-sam001"
    assert hits[0].privacy == "private"
    assert hits[0].entity_id == "people/sam-okafor"


def test_alias_boost_pulls_in_linked_items(indexed, llm):
    # the climbing episode never says "Sam": it is linked to Sam only through its entities
    no_fts = fts_ranked(indexed, "Sam", Filters(), 40)
    climbing_pk = indexed.execute(
        "SELECT pk FROM items WHERE id = 'ep-20261020-1900-climbing'"
    ).fetchone()[0]
    assert climbing_pk not in no_fts
    hits = search(indexed, llm, "Sam", k=6, rerank=False)
    assert "ep-20261020-1900-climbing" in ids(hits)
    assert all(h.entity_id == "people/sam-okafor" or "climbing" in h.id for h in hits[:4])


def test_alias_boost_follows_fact_objects(indexed, llm):
    hits = search(indexed, llm, "Northwind", k=6, rerank=False)
    # facts whose object is the org (Alex works there, Atlas is owned by it)
    assert {"f-work01", "f-atl002"} <= set(ids(hits))


def test_private_filter(indexed, llm):
    query = "Sam cello bouldering checkup sleep"
    assert any(h.privacy == "private" for h in search(indexed, llm, query))
    public = search(indexed, llm, query, include_private=False, k=50)
    assert public
    assert all(h.privacy == "normal" for h in public)
    assert not {"f-sam001", "ep-20261020-1900-climbing", "o-clm001"} & set(ids(public))


def test_as_of_filters_facts(indexed, llm):
    current = ids(search(indexed, llm, "where does Alex live", k=50, kinds=["fact"]))
    assert "f-city01" in current and "f-city00" not in current

    past = search(indexed, llm, "where does Alex live", k=50, kinds=["fact"], as_of="2020")
    assert "f-city00" in ids(past) and "f-city01" not in ids(past)
    assert next(h for h in past if h.id == "f-city00").current is False

    boundary = ids(search(indexed, llm, "lives", k=50, kinds=["fact"], as_of="2024-05"))
    assert {"f-city00", "f-city01"} <= set(boundary)

    later = ids(search(indexed, llm, "lives", k=50, kinds=["fact"], as_of=date(2025, 1, 1)))
    assert "f-city01" in later and "f-city00" not in later

    # works_at started 2023-02: not valid in 2020
    assert "f-work01" not in ids(search(indexed, llm, "works", k=50, as_of="2020"))


def test_as_of_rejects_bad_dates(indexed, llm):
    with pytest.raises(ValueError):
        search(indexed, llm, "x", as_of="last spring")


def test_kinds_filter(indexed, llm):
    hits = search(indexed, llm, "Atlas deadline", kinds=["episode", "observation"], k=50)
    assert hits
    assert {h.kind for h in hits} <= {"episode", "observation"}


def test_rerank_reorders_head(indexed, llm):
    settings = RetrievalSettings(candidates=40, rerank_top=5)
    hits = search(indexed, llm, "Atlas migration deadline", k=8, settings=settings)
    head = hits[:5]
    assert all(h.rerank is not None for h in head)
    assert [h.score for h in head] == sorted((h.score for h in head), reverse=True)
    assert all(h.rerank is None for h in hits[5:])
    assert ("rerank", {"query": "Atlas migration deadline", "n": 5}) in llm.calls


def test_rerank_failure_falls_back_to_fusion(indexed):
    class NoRerank(FakeLLM):
        def rerank(self, query, documents, top_n=None):
            raise ConnectionError("reranker down")

    hits = search(indexed, NoRerank(dim=64), "cello")
    assert hits[0].id == "f-sam001"
    assert hits[0].rerank is None


def test_query_vector_override(indexed, llm):
    vec = llm.embed(["Lives in Rotterdam"])[0]
    hits = search(indexed, llm, "zzz", query_vector=vec, rerank=False)
    assert hits[0].id == "f-city01"
    assert not any(c[0] == "embed" and c[1]["query"] for c in llm.calls)


def test_hit_sources(indexed, llm):
    hits = search(indexed, llm, "bouldering unwind", kinds=["observation"], rerank=False)
    assert hits[0].id == "o-clm001"
    assert hits[0].sources == ("ep-20261020-1900-climbing",)
    assert hits[0].importance == 4


def test_empty_index(db, llm):
    reindex(
        _EmptyVault(), db, llm, private_regions=frozenset({"relationships.friends", "self.health"})
    )
    assert search(db, llm, "anything") == []
    assert isinstance(llm.embed(["x"]), np.ndarray)


class _EmptyVault:
    def pages(self):
        return []

    def episodes(self):
        return []

    def observations(self):
        return []

    def intentions(self):
        return []

    def alias_index(self):
        return {}
