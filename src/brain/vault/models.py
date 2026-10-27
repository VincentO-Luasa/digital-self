"""In-memory model of the vault. The canonical format is docs/vault-schema.md."""

from __future__ import annotations

import re
import secrets
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "0.6"

Kind = Literal["stated", "observed", "inferred"]
Privacy = Literal["normal", "private"]
EntityType = Literal["person", "org", "project", "place", "topic", "event", "self"]
Channel = Literal["chat", "interview", "external", "import"]
EpisodeStatus = Literal["pending", "extracted", "consolidated"]

PARTIAL_DATE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")
WIKILINK = re.compile(r"\[\[([^\]|#^]+)(?:[#^][^\]|]*)?(?:\|[^\]]*)?\]\]")
_B32 = "abcdefghijklmnopqrstuvwxyz234567"

SELF_ID = "self"


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


# Predicate vocabulary (docs/vault-schema.md). Single-valued per subject unless listed as multi.
SINGLE_VALUED = _words(
    "name born hometown lives_in role team manager met_at last_seen owner status goal deadline "
    "located_in part_of"
)
RELATIONSHIP_PREDICATES = _words("family_of partner_of friend_of colleague_of mentor_of knows")
SELF_PREDICATES = _words(
    "has_strength has_flaw values goal habit interest likes dislikes fears health"
)
MULTI_VALUED = (
    _words(
        "nationality speaks works_at responsibility worked_on studied_at degree certification "
        "has_skill stakeholder uses member_of related_to note"
    )
    | RELATIONSHIP_PREDICATES
    | (SELF_PREDICATES - {"goal"})
)
PREDICATES = SINGLE_VALUED | MULTI_VALUED


def is_multi_valued(p: str, subject: str) -> bool:
    """``goal`` is single-valued for projects but multi-valued for the self."""
    return p in MULTI_VALUED or (subject == SELF_ID and p in SELF_PREDICATES)


def new_id(prefix: str) -> str:
    """Short block id such as ``f-k3x9q2`` (facts), ``o-...`` (observations), ``i-...``."""
    return f"{prefix}-" + "".join(secrets.choice(_B32) for _ in range(6))


def link_target(value: str) -> str | None:
    """Return the target of the first wikilink in ``value``, or None for a literal."""
    m = WIKILINK.search(value)
    return m.group(1).strip() if m else None


def check_partial_date(value: str | None) -> str | None:
    if value is not None and not PARTIAL_DATE.match(value):
        raise ValueError(f"expected YYYY, YYYY-MM or YYYY-MM-DD, got {value!r}")
    return value


class Fact(BaseModel):
    """One fact line: ``- text (p:: ...) (o:: ...) ... ^f-xxxxxx``."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    text: str
    p: str
    o: str
    valid_from: str | None = Field(default=None, alias="from")
    valid_to: str | None = Field(default=None, alias="to")
    rec: date
    src: list[str]  # wikilink targets: episode ids, raw doc paths, review ids
    kind: Kind
    evidence: list[str] = []
    by: str | None = None  # history only: id of the superseding fact
    extra: dict[str, str] = {}  # optional qualifiers such as rel::, level::, at::

    _check_dates = field_validator("valid_from", "valid_to")(check_partial_date)

    @property
    def object_id(self) -> str | None:
        return link_target(self.o)


class Page(BaseModel):
    """A wiki page: frontmatter + summary + current facts + history + human notes."""

    id: str  # path under wiki/ without .md, e.g. "people/sam-okafor"
    type: EntityType
    subject: str  # entity the facts are about; wiki/self/* pages use "self"
    title: str
    aliases: list[str] = []
    regions: list[str] = []
    privacy: Privacy = "normal"
    created: date
    updated: date
    summary: str = ""
    facts: list[Fact] = []
    history: list[Fact] = []
    notes: str = ""
    meta: dict[str, Any] = {}  # unknown frontmatter keys (e.g. Obsidian tags), kept verbatim
    # malformed lines of "facts" / "history", kept verbatim so a rewrite never loses them
    unparsed: dict[str, list[str]] = {}


class Episode(BaseModel):
    """A topic-sized chunk of one conversation, the unit of extraction."""

    id: str  # ep-YYYYMMDD-HHMM-<slug>
    session: str
    turns: tuple[int, int]
    channel: Channel
    provider: str | None = None  # for channel "external", e.g. "claude-code"
    regions: list[str] = []
    entities: list[str] = []  # entity ids
    importance: int = Field(ge=1, le=10)
    gist: str
    started: datetime
    status: EpisodeStatus = "pending"
    privacy: Privacy = "normal"  # private: it holds an answer the owner marked private
    summary: str = ""


def fact_private(fact: Fact) -> bool:
    """A fact from an answer the owner marked private (``privacy:: private``, schema v0.4)."""
    return fact.extra.get("privacy") == "private"


class Observation(BaseModel):
    """A dated journal line: ``- 2026-10-25 · imp 7 · text · [[src]] ^o-xxxxxx``."""

    id: str
    day: date
    importance: int = Field(ge=1, le=10)
    text: str
    src: list[str]


class Intention(BaseModel):
    """Prospective memory: bring something up when a cue appears or after a date."""

    id: str
    text: str
    done: bool = False
    when: list[str] = []  # entity ids that trigger it
    after: str | None = None
    src: list[str] = []

    _check_after = field_validator("after")(check_partial_date)
