from __future__ import annotations

import os

from brain.vault.lifemap import load_lifemap, parse_lifemap
from brain.vault.store import Vault


def test_pages_and_lookup(tmp_vault: Vault) -> None:
    ids = [p.id for p in tmp_vault.pages()]
    assert ids == [
        "orgs/northwind-labs",
        "people/sam-okafor",
        "projects/atlas-migration",
        "self/profile",
    ]
    assert tmp_vault.page("self").title == "Alex Rivera"
    assert tmp_vault.page("people/sam-okafor").privacy == "private"
    assert tmp_vault.page("people/nobody") is None
    assert [p.id for p in tmp_vault.pages_about("self")] == ["self/profile"]


def test_alias_index_is_casefolded(tmp_vault: Vault) -> None:
    index = tmp_vault.alias_index()
    assert index["sam"] == ["people/sam-okafor"]
    assert index["samuel okafor"] == ["people/sam-okafor"]
    assert index["alex"] == ["self"]
    assert index["northwind"] == ["orgs/northwind-labs"]
    assert index["atlas migration"] == ["projects/atlas-migration"]


def test_find_fact(tmp_vault: Vault) -> None:
    page, fact, in_history = tmp_vault.find_fact("f-m4n5o6")
    assert (page.id, fact.p, in_history) == ("self/profile", "role", False)
    page, fact, in_history = tmp_vault.find_fact("f-s4t5u6")
    assert in_history and fact.by == "f-m4n5o6"
    assert tmp_vault.find_fact("f-zzzzzz") is None


def test_episodes_observations_intentions(tmp_vault: Vault) -> None:
    episodes = tmp_vault.episodes()
    assert [e.id for e in episodes] == [
        "ep-20261021-2030-friends",
        "ep-20261025-1412-career",
        "ep-20261026-0915-atlas",
    ]
    assert episodes[-1].provider == "claude-code"
    assert tmp_vault.episode_path(episodes[0]).exists()
    assert [o.id for o in tmp_vault.observations()] == ["o-a2b3c4", "o-4hd7sa"]
    assert [i.id for i in tmp_vault.intentions()] == ["i-p2m7wd", "i-q3n4xe"]
    assert tmp_vault.warnings() == []


def test_mtime_cache_reloads_changed_files(tmp_vault: Vault) -> None:
    first = tmp_vault.page("orgs/northwind-labs")
    assert tmp_vault.page("orgs/northwind-labs") is first  # cached
    path = tmp_vault.page_path("orgs/northwind-labs")
    path.write_text(path.read_text().replace("# Northwind Labs", "# Northwind Labs Ltd"))
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert tmp_vault.page("orgs/northwind-labs").title == "Northwind Labs Ltd"


def test_unparseable_file_is_skipped_with_warning(tmp_vault: Vault) -> None:
    bad = tmp_vault.wiki_dir / "topics" / "broken.md"
    bad.parent.mkdir()
    bad.write_text("---\nid: topics/broken\n")
    assert "topics/broken" not in [p.id for p in tmp_vault.pages()]
    warnings = tmp_vault.warnings()
    assert len(warnings) == 1 and warnings[0].path == bad and "skipped" in warnings[0].message


def test_lifemap_privacy_and_slots(tmp_vault: Vault) -> None:
    lifemap = tmp_vault.lifemap
    assert lifemap.privacy_for(["relationships.friends"]) == "private"
    assert lifemap.privacy_for(["relationships.family.cousins"]) == "private"  # inherits
    assert lifemap.privacy_for(["relationships.work", "projects"]) == "normal"
    assert lifemap.privacy_for([]) == "normal"
    assert [s.id for s in lifemap.slots("identity")] == ["name", "city", "hometown", "languages"]
    assert lifemap.slots("nope") == ()
    assert lifemap.region_for("has_flaw", "self") == "self.traits"
    assert lifemap.region_for("goal", "project") == "projects"


def test_lifemap_vault_override(tmp_vault: Vault) -> None:
    (tmp_vault.root / "life-map.yaml").write_text(
        "version: 0.2\nregions:\n  hobbies:\n    label: Hobbies\n    privacy: private\n"
        "    slots:\n      - {id: sport, p: interest, subject: self, multi: true, min: 2}\n"
    )
    lifemap = load_lifemap(tmp_vault.root)
    assert lifemap.version == "0.2"
    assert list(lifemap.regions) == ["hobbies"]
    assert lifemap.privacy_for(["hobbies"]) == "private"
    assert lifemap.slots("hobbies")[0].min == 2


def test_parse_lifemap_defaults() -> None:
    lifemap = parse_lifemap("regions:\n  a: {}\n")
    assert lifemap.regions["a"].privacy == "normal" and lifemap.regions["a"].slots == ()
