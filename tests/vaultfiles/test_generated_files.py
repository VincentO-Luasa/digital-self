"""Generated index.md and core.md: deterministic, privacy-aware, capped, byte-stable."""

from __future__ import annotations

from datetime import date

from brain.tokens import estimate_tokens
from brain.vault import ChangeSet, apply, gitops
from brain.vault.generated import (
    GENERATED_NOTE,
    build_core,
    build_index,
    first_sentence,
    refresh_generated,
)
from brain.vault.ops import AddFact
from brain.vault.store import Vault

TODAY = date(2026, 10, 25)


def test_index_groups_pages_by_region(tmp_vault: Vault) -> None:
    text = build_index(tmp_vault)
    assert text.startswith("# Index\n\n" + GENERATED_NOTE)
    sections = text.split("\n## ")[1:]
    headings = [s.split("\n", 1)[0] for s in sections]
    # life-map order: identity, career.current_job, ..., projects, relationships.friends
    assert headings == ["Identity & basics", "Current job", "Projects", "Friends"]
    assert (
        "- [[projects/atlas-migration]] — Atlas Migration moves Northwind Labs' nightly batch "
        "pipelines from a legacy cluster to a managed warehouse." in text
    )
    assert text.count("- [[") == len(tmp_vault.pages())


def test_index_can_leave_out_private_pages(tmp_vault: Vault) -> None:
    assert "people/sam-okafor" in build_index(tmp_vault)
    assert "people/sam-okafor" not in build_index(tmp_vault, include_private=False)


def test_core_card_contents(tmp_vault: Vault) -> None:
    core = build_core(tmp_vault, today=TODAY)
    assert "Name: Alex Rivera" in core
    assert "Alex is a senior data engineer at Northwind Labs" in core
    situation = core.split("## Current situation\n\n")[1].split("\n\n")[0].splitlines()
    # salience: role and employer before residence, languages and skills
    assert situation[0] == "- Is a senior data engineer (since 2025-03)"
    assert situation[1] == "- Works at Northwind Labs as data engineer (since 2023-02)"
    assert situation.index("- Lives in Porto (since 2024)") < situation.index(
        "- Is an expert in Python"
    )
    assert "Was a data engineer" not in core  # history is not current
    assert "Is called Alex Rivera" not in core  # the name is the header
    entities = core.split("## People and projects\n\n")[1]
    assert "Atlas Migration (project)" in entities  # owned by the owner
    assert "Sam Okafor (person)" in entities  # friend of the owner
    assert "Northwind Labs (org)" in entities  # employer


def test_core_respects_privacy_validity_and_cap(tmp_vault: Vault) -> None:
    assert "Sam Okafor" not in build_core(tmp_vault, today=TODAY, include_private=False)
    # before Alex moved to Porto and before the senior role
    early = build_core(tmp_vault, today=date(2023, 6, 1))
    assert "- Lives in Porto" not in early and "- Is a senior data engineer" not in early
    assert "- Works at Northwind Labs as data engineer (since 2023-02)" in early
    small = build_core(tmp_vault, today=TODAY, max_tokens=120)
    assert estimate_tokens(small) <= 120
    assert "Is a senior data engineer" in small  # highest salience survives the cap


def test_core_is_byte_stable(tmp_vault: Vault) -> None:
    assert build_core(tmp_vault, today=TODAY) == build_core(Vault(tmp_vault.root), today=TODAY)


def test_refresh_writes_and_commits_only_on_change(tmp_vault: Vault) -> None:
    changed = refresh_generated(tmp_vault, today=TODAY)
    assert {p.name for p in changed} == {"core.md", "index.md", "denylist.txt"}
    head = gitops.head(tmp_vault.root)
    assert refresh_generated(tmp_vault, today=TODAY) == []
    assert gitops.head(tmp_vault.root) == head

    cs = ChangeSet(
        ops=[
            AddFact(
                subject="self", p="interest", o="bouldering", text="Goes bouldering", kind="stated"
            )
        ],
        origin="user",
        src=["raw/conversations/2026/10/conv-20261025-1400.md"],
    )
    apply(cs, tmp_vault, review=False)
    changed = refresh_generated(tmp_vault, today=TODAY)
    assert {p.name for p in changed} == {"core.md", "index.md"}  # new page self/interests
    assert "- Goes bouldering" in tmp_vault.core_path.read_text()
    assert gitops.head(tmp_vault.root) != head


def test_first_sentence() -> None:
    assert first_sentence("Lives in Porto. Speaks Spanish.") == "Lives in Porto."
    assert first_sentence("e.g. this one") == "e.g. this one"
    long = first_sentence("word " * 100, limit=30)
    assert len(long) <= 30 and long.endswith("…")


def test_denylist_lists_private_names_and_keeps_manual_lines(tmp_vault: Vault) -> None:
    from brain.vault.generated import DENYLIST_MARKER, build_denylist

    manual = "# my terms\nSecret Street\n"
    text = build_denylist(tmp_vault, manual + "\n" + DENYLIST_MARKER + "\nStale Name\n")
    head, generated = text.split(DENYLIST_MARKER)
    assert head.strip() == manual.strip()
    names = generated.split()
    assert "Sam" in names and "Stale" not in names  # Sam Okafor's page is private
    assert "Northwind" not in generated  # normal pages are not listed
    assert build_denylist(tmp_vault, text) == text  # stable
