"""Entity resolution: aliases, word and embedding candidates, the typed decision."""

from __future__ import annotations

from memtools import Decider, fake  # type: ignore[import-not-found]

from brain.memory.resolve import resolve_mentions
from brain.memory.segment import Mention
from brain.vault.ops import ChangeSet, CreateEntity
from brain.vault.store import Vault
from brain.vault.writer import apply


def m(name: str, kind: str = "person") -> Mention:
    return Mention(name=name, type=kind)  # type: ignore[arg-type]


def test_alias_resolves_without_asking(mvault: Vault) -> None:
    decider = Decider()
    report = resolve_mentions(
        mvault,
        fake([], decider),
        [m("Sam"), m("Northwind", "org"), m("Alex")],
        "text",
        owner_names=["Alex Rivera", "Alex"],
    )
    assert [(r.name, r.entity_id, r.how) for r in report.resolved] == [
        ("Sam", "people/sam-okafor", "alias"),
        ("Northwind", "orgs/northwind-labs", "alias"),
    ]
    assert not decider.asked


def test_partial_name_is_asked_and_becomes_an_alias(mvault: Vault) -> None:
    decider = Decider(who={"Okafor": "Sam Okafor"})
    text = "[1] OWNER: Okafor says hi."
    report = resolve_mentions(mvault, fake([], decider), [m("Okafor")], text)
    [res] = report.resolved
    assert (res.entity_id, res.how, res.alias) == ("people/sam-okafor", "decision", True)
    assert decider.asked[0][0] == "who" and "Okafor says hi" in decider.asked[0][1]


def test_someone_else_stays_unresolved(mvault: Vault) -> None:
    report = resolve_mentions(mvault, fake([], Decider()), [m("Samuel Okafor Jr")], "x")
    # "Samuel Okafor" is an alias: its words are a subset → asked; the decider says "else"
    assert report.resolved[0].entity_id is None
    assert report.resolved[0].candidates == ["people/sam-okafor"]


def test_ambiguous_alias_is_decided(mvault: Vault) -> None:
    apply(
        ChangeSet(
            ops=[
                CreateEntity(id="people/sam-lee", type="person", title="Sam Lee", aliases=["Sam"])
            ],
            origin="user",
            src=["test"],
        ),
        mvault,
    )
    decider = Decider(who={"Sam": "Sam Lee"})
    report = resolve_mentions(mvault, fake([], decider), [m("Sam")], "[1] OWNER: Sam from work")
    assert report.resolved[0].entity_id == "people/sam-lee"
    assert set(report.resolved[0].candidates) == {"people/sam-okafor", "people/sam-lee"}


def test_embedding_candidates_only_for_compatible_types(mvault: Vault) -> None:
    # "Atlas project" shares the word "atlas" with the project: a word candidate, asked
    decider = Decider(who={"the Atlas project": "Atlas Migration"})
    report = resolve_mentions(mvault, fake([], decider), [m("the Atlas project", "project")], "x")
    assert report.resolved[0].entity_id == "projects/atlas-migration"
    # a person mention never resolves to the project
    report = resolve_mentions(mvault, fake([], Decider()), [m("Atlas", "person")], "x")
    assert "projects/atlas-migration" not in report.resolved[0].candidates
