"""In-memory fake vault (fictional persona Alex Rivera) for index tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pytest

from brain.index.db import connect
from brain.llm.fake import FakeLLM
from brain.vault.models import Episode, Fact, Intention, Observation, Page

TODAY = date(2026, 10, 25)
PRIVATE_REGIONS = frozenset({"relationships.friends", "self.health"})


def fact(fid: str, text: str, p: str, o: str, **kw: object) -> Fact:
    return Fact(
        id=fid,
        text=text,
        p=p,
        o=o,
        rec=kw.pop("rec", TODAY),
        src=kw.pop("src", ["ep-20261025-1412-career"]),
        kind=kw.pop("kind", "stated"),
        **kw,
    )


def page(pid: str, type_: str, title: str, **kw: object) -> Page:
    return Page(
        id=pid,
        type=type_,
        subject=kw.pop("subject", pid),
        title=title,
        created=TODAY,
        updated=TODAY,
        **kw,
    )


def make_pages() -> list[Page]:
    return [
        page(
            "self/profile",
            "self",
            "Profile",
            subject="self",
            aliases=["Alex", "Alex Rivera"],
            summary="Alex Rivera is a data engineer.",
            facts=[
                fact(
                    "f-work01",
                    "Works at [[orgs/northwind-labs|Northwind Labs]] as data engineer",
                    "works_at",
                    "[[orgs/northwind-labs]]",
                    valid_from="2023-02",
                ),
                fact(
                    "f-city01", "Lives in Rotterdam", "lives_in", "Rotterdam", valid_from="2024-05"
                ),
            ],
            history=[
                fact(
                    "f-city00",
                    "Lives in Utrecht",
                    "lives_in",
                    "Utrecht",
                    valid_from="2019",
                    valid_to="2024-05",
                    by="f-city01",
                ),
            ],
        ),
        page(
            "people/sam-okafor",
            "person",
            "Sam Okafor",
            aliases=["Sam", "Samuel Okafor"],
            regions=["relationships.friends"],
            privacy="private",
            summary="Old friend of Alex from university.",
            facts=[
                fact("f-sam001", "Plays the cello in an amateur orchestra", "likes", "cello"),
                fact("f-sam002", "Friend of Alex since 2015", "friend_of", "[[self/profile]]"),
            ],
        ),
        page(
            "orgs/northwind-labs",
            "org",
            "Northwind Labs",
            aliases=["Northwind"],
            regions=["career.current_job"],
            summary="Data platform company where Alex works.",
        ),
        page(
            "projects/atlas-migration",
            "project",
            "Atlas Migration",
            aliases=["Atlas"],
            regions=["projects"],
            facts=[
                fact("f-atl001", "Deadline is January 2027", "deadline", "2027-01"),
                fact(
                    "f-atl002",
                    "Owned by Northwind Labs",
                    "owner",
                    "[[orgs/northwind-labs]]",
                ),
            ],
        ),
    ]


def episode(eid: str, gist: str, **kw: object) -> Episode:
    return Episode(
        id=eid,
        session=kw.pop("session", "conv-20261025-1400"),
        turns=(1, 5),
        channel="chat",
        importance=kw.pop("importance", 5),
        gist=gist,
        started=datetime(2026, 10, 25, 14, 12),
        **kw,
    )


def make_episodes() -> list[Episode]:
    return [
        episode(
            "ep-20261025-1412-career",
            "Alex worries the Atlas migration deadline is unrealistic",
            entities=["[[projects/atlas-migration]]", "[[orgs/northwind-labs]]"],
            regions=["career.current_job"],
            importance=6,
            summary="Talked through the schedule and the staffing gap.",
        ),
        episode(
            "ep-20261020-1900-climbing",
            "Went bouldering together on a rainy evening",
            entities=["[[people/sam-okafor]]"],
            regions=["self.interests"],
        ),
        episode(
            "ep-20261018-0900-checkup",
            "Annual checkup went fine, doctor suggested more sleep",
            regions=["self.health"],
        ),
    ]


def make_observations() -> list[Observation]:
    return [
        Observation(
            id="o-anx001",
            day=TODAY,
            importance=7,
            text="Alex is anxious about the [[projects/atlas-migration]] deadline",
            src=["ep-20261025-1412-career"],
        ),
        Observation(
            id="o-clm001",
            day=date(2026, 10, 20),
            importance=4,
            text="Alex enjoys bouldering as a way to unwind",
            src=["[[ep-20261020-1900-climbing]]"],
        ),
    ]


@dataclass
class FakeVault:
    page_list: list[Page] = field(default_factory=make_pages)
    episode_list: list[Episode] = field(default_factory=make_episodes)
    observation_list: list[Observation] = field(default_factory=make_observations)
    intention_list: list[Intention] = field(default_factory=list)

    def pages(self) -> list[Page]:
        return list(self.page_list)

    def episodes(self) -> list[Episode]:
        return list(self.episode_list)

    def observations(self) -> list[Observation]:
        return list(self.observation_list)

    def intentions(self) -> list[Intention]:
        return list(self.intention_list)

    def alias_index(self) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for p in self.page_list:
            for name in [p.title, *p.aliases]:
                index.setdefault(name.casefold(), []).append(p.subject)
        return index

    def page(self, pid: str) -> Page:
        return next(p for p in self.page_list if p.id == pid)


@pytest.fixture
def private_regions() -> frozenset[str]:
    return PRIVATE_REGIONS


@pytest.fixture
def vault() -> FakeVault:
    return FakeVault()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM(dim=64)


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / ".brain" / "index.sqlite")
    yield conn
    conn.close()
