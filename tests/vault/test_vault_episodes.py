"""Writer support for the memory pipeline: episodes, confirmations, owner corrections, log
events. Fictional persona only (Alex Rivera, Northwind Labs, Sam Okafor, Atlas Migration)."""

from __future__ import annotations

from datetime import datetime

import pytest

from brain.index.reindex import snapshot
from brain.vault import gitops
from brain.vault.ops import (
    AddEpisode,
    AddObservation,
    ChangeSet,
    ConfirmFact,
    FactSpec,
    SupersedeFact,
    UpdateEpisode,
)
from brain.vault.parse import parse_episode
from brain.vault.render import render_episode
from brain.vault.store import Vault
from brain.vault.writer import apply, correction_problem, log_event, log_has

NOW = datetime(2026, 10, 27, 21, 30)
EP = "ep-20261027-2100-atlas-update"


def episode_op(**kw) -> AddEpisode:
    data = {
        "id": EP,
        "session": "conv-20261027-2100",
        "turns": (1, 3),
        "channel": "chat",
        "regions": ["projects"],
        "entities": ["projects/atlas-migration"],
        "importance": 5,
        "gist": "Alex says the dbt models are migrated",
        "started": datetime(2026, 10, 27, 21, 0),
        "summary": "Alex finished migrating the dbt models. The Airflow DAGs come next.",
    }
    return AddEpisode(**{**data, **kw})


def cs(*ops, src=None, origin="extract") -> ChangeSet:
    return ChangeSet(ops=list(ops), origin=origin, src=src or ["raw/conversations/x.md"])


def test_add_episode_writes_the_schema_file(tmp_vault: Vault) -> None:
    result = apply(cs(episode_op()), tmp_vault, now=NOW)
    assert result.episodes == [EP]
    path = tmp_vault.root / "episodes/2026/10" / f"{EP}.md"
    text = path.read_text()
    assert text.startswith("---\nid: ep-20261027-2100-atlas-update\ntype: episode\n")
    assert 'entities: ["[[projects/atlas-migration]]"]' in text
    assert "status: pending" in text
    episode, warnings = parse_episode(text, path=path)
    assert not warnings
    assert render_episode(episode) == text
    assert tmp_vault.episode(EP).turns == (1, 3)
    assert f"- episode {EP}" in tmp_vault.log_path.read_text()
    assert gitops.head(tmp_vault.root) == result.commit


@pytest.mark.parametrize(
    ("kw", "reason"),
    [
        ({"id": "episode-1"}, "bad episode id"),
        ({"turns": (3, 1)}, "bad turn range"),
        ({"gist": " "}, "needs a session and a gist"),
        ({"summary": "## Facts\n- not here"}, "section headings"),
    ],
)
def test_add_episode_validation(tmp_vault: Vault, kw: dict, reason: str) -> None:
    result = apply(cs(episode_op(**kw)), tmp_vault, now=NOW)
    assert reason in result.skipped[0][1]


def test_episode_ids_are_unique_and_external_cannot_write_episodes(tmp_vault: Vault) -> None:
    apply(cs(episode_op()), tmp_vault, now=NOW)
    again = apply(cs(episode_op()), tmp_vault, now=NOW)
    assert "already exists" in again.skipped[0][1]
    ext = apply(cs(episode_op(id="ep-20261027-2101-x"), origin="external"), tmp_vault, now=NOW)
    assert "memory pipeline only" in ext.skipped[0][1]


def test_update_episode_status_and_entities(tmp_vault: Vault) -> None:
    apply(cs(episode_op()), tmp_vault, now=NOW)
    op = UpdateEpisode(id=EP, status="extracted", add_entities=["[[orgs/northwind-labs]]"])
    result = apply(cs(op, src=[EP]), tmp_vault, now=NOW)
    assert result.applied == [op]
    episode = tmp_vault.episode(EP)
    assert episode.status == "extracted"
    assert episode.entities == ["projects/atlas-migration", "orgs/northwind-labs"]
    unknown = apply(cs(UpdateEpisode(id="ep-20261027-0000-nope")), tmp_vault, now=NOW)
    assert "unknown episode" in unknown.skipped[0][1]


def test_episodes_are_indexed_and_survive_reindex(tmp_vault: Vault) -> None:
    apply(cs(episode_op()), tmp_vault, now=NOW)
    snap, stats = snapshot(Vault(tmp_vault.root), frozenset({"relationships.friends"}))
    item = snap.items[EP]
    assert item.kind == "episode" and item.importance == 5
    assert item.text.startswith("Alex says the dbt models are migrated\nAlex finished")
    assert (EP, "projects/atlas-migration", "mention") in snap.links


def test_confirm_fact_adds_the_source_once(tmp_vault: Vault) -> None:
    result = apply(cs(ConfirmFact(fact_id="f-d5e6f7"), src=[EP]), tmp_vault, now=NOW)
    assert result.confirmed == 1
    _, fact, _ = tmp_vault.find_fact("f-d5e6f7")
    assert fact.src == ["raw/docs/cv.md", EP]
    assert fact.rec.isoformat() == "2026-10-20"  # nothing else changes
    again = apply(cs(ConfirmFact(fact_id="f-d5e6f7"), src=[EP]), tmp_vault, now=NOW)
    assert "already cites" in again.skipped[0][1]
    # an ended fact may be confirmed; a replaced one may not
    assert apply(cs(ConfirmFact(fact_id="f-l6s2b3"), src=[EP]), tmp_vault, now=NOW).confirmed
    replaced = apply(cs(ConfirmFact(fact_id="f-s4t5u6"), src=[EP]), tmp_vault, now=NOW)
    assert "superseded" in replaced.skipped[0][1]


def lives_in(city: str, start: str | None = None) -> FactSpec:
    return FactSpec(p="lives_in", o=city, text=f"Lives in {city}", kind="stated", **{"from": start})


def test_supersession_is_queued_unless_listed_as_an_owner_correction(tmp_vault: Vault) -> None:
    op = SupersedeFact(fact_id="f-d5e6f7", new=lives_in("Braga", "2026-09"))
    queued = apply(cs(op, src=[EP]), tmp_vault, now=NOW)
    assert queued.queued[0][1] == "supersession"
    assert tmp_vault.find_fact("f-d5e6f7")[2] is False


def test_owner_correction_is_applied_and_logged(tmp_vault: Vault) -> None:
    op = SupersedeFact(fact_id="f-d5e6f7", new=lives_in("Braga", "2026-09"))
    result = apply(cs(op, src=[EP]), tmp_vault, now=NOW, corrections={"f-d5e6f7"})
    assert result.corrected == ["f-d5e6f7"] and not result.queued
    page, old, in_history = tmp_vault.find_fact("f-d5e6f7")
    assert in_history and old.valid_to == "2026-09" and old.by == result.new_ids[0]
    assert "- owner correction applied: ^f-d5e6f7" in tmp_vault.log_path.read_text()


def test_an_ended_fact_can_be_corrected_and_stays_in_history(tmp_vault: Vault) -> None:
    new = FactSpec(
        p="works_at",
        o="Fabrikam Analytics",
        text="Worked at Fabrikam Analytics",
        kind="stated",
        **{"from": "2019-06", "to": "2023-03"},
    )
    op = SupersedeFact(fact_id="f-f6b7a2", new=new)
    result = apply(cs(op, src=[EP]), tmp_vault, now=NOW, corrections={"f-f6b7a2"})
    assert result.corrected == ["f-f6b7a2"]
    _, old, in_history = tmp_vault.find_fact("f-f6b7a2")
    assert in_history and old.valid_to == "2023-01" and old.by == result.new_ids[0]
    _, fixed, current = tmp_vault.find_fact(result.new_ids[0])
    assert not current and fixed.valid_to == "2023-03"
    # replaced now: it cannot be corrected again
    again = apply(cs(op, src=[EP]), tmp_vault, now=NOW, corrections={"f-f6b7a2"})
    assert "already superseded" in again.skipped[0][1]


@pytest.mark.parametrize(
    ("fact_id", "new", "origin", "problem"),
    [
        ("f-d5e6f7", lives_in("Braga"), "extract", None),
        ("f-d5e6f7", lives_in("Braga"), "external", "origin external"),
        ("f-n2o3p4", None, "extract", "not the owner"),  # a project's status
        ("f-y5z6a7", None, "extract", "not the owner"),  # Sam's friendship (private page)
        (
            "f-d5e6f7",
            FactSpec(p="hometown", o="Braga", text="Grew up in Braga", kind="stated"),
            "extract",
            "changes the predicate",
        ),
        (
            "f-d5e6f7",
            FactSpec(p="lives_in", o="Braga", text="Lives in Braga", kind="inferred"),
            "extract",
            "inferred replacement",
        ),
        (
            "f-k3x7q2",
            FactSpec(p="works_at", o="[[people/sam-okafor]]", text="x", kind="stated"),
            "extract",
            "private page people/sam-okafor",
        ),
    ],
)
def test_correction_problem(
    tmp_vault: Vault, fact_id: str, new: FactSpec | None, origin: str, problem: str | None
) -> None:
    op = SupersedeFact(fact_id=fact_id, new=new)
    got = correction_problem(op, tmp_vault, origin=origin)  # type: ignore[arg-type]
    assert got == problem if problem is None else problem in got


def test_private_self_region_is_never_auto_corrected(tmp_vault: Vault) -> None:
    health = FactSpec(p="health", o="Asthma", text="Has asthma", kind="stated")
    from brain.vault.ops import AddFact

    added = apply(
        cs(AddFact(subject="self", **health.model_dump(by_alias=True)), src=[EP]),
        tmp_vault,
        now=NOW,
    )
    fact_id = added.new_ids[0]
    op = SupersedeFact(fact_id=fact_id, new=FactSpec(p="health", o="None", text="x", kind="stated"))
    assert "private" in correction_problem(op, tmp_vault)
    result = apply(cs(op, src=[EP]), tmp_vault, now=NOW, corrections={fact_id})
    assert "not auto-applied as an owner correction" in result.queued[0][1]


def test_log_event_and_log_has(tmp_vault: Vault) -> None:
    src = "raw/conversations/2026/10/conv-20261027-1230.md"
    assert not log_has(tmp_vault, "extract", src, "no episode")
    sha = log_event(tmp_vault, "extract", [src], "no episode: small talk", now=NOW)
    assert sha and log_has(tmp_vault, "extract", src, "no episode")
    assert not log_has(tmp_vault, "extract", src + "x", "no episode")


def test_observation_day_is_the_change_set_date(tmp_vault: Vault) -> None:
    op = AddObservation(text="Alex is relieved about the Atlas go-live", importance=5)
    change = ChangeSet(ops=[op], origin="extract", src=[EP], created=datetime(2026, 9, 30, 22, 0))
    apply(change, tmp_vault, now=NOW)
    obs = [o for o in tmp_vault.observations() if o.src == [EP]]
    assert obs[0].day.isoformat() == "2026-09-30"
    assert (tmp_vault.root / "journal/2026-09.md").exists()
