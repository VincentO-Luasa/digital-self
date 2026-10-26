"""Operations: the only way memory changes.

Models and the big model *propose* operations; ``brain.vault.writer`` validates and applies them
(or queues them for review), appends to log.md and commits the vault. The JSON shape is mirrored
in vault_template/AGENTS.md.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from brain.vault.models import (
    Channel,
    EntityType,
    EpisodeStatus,
    Kind,
    Privacy,
    check_partial_date,
)


class FactSpec(BaseModel):
    """A fact to add. ``rec``, ``src`` and the block id are filled in by the writer."""

    model_config = {"populate_by_name": True}

    p: str
    o: str
    text: str
    valid_from: str | None = Field(default=None, alias="from")
    valid_to: str | None = Field(default=None, alias="to")
    kind: Kind
    evidence: list[str] = []
    extra: dict[str, str] = {}

    _check_dates = field_validator("valid_from", "valid_to")(check_partial_date)


class CreateEntity(BaseModel):
    op: Literal["create_entity"] = "create_entity"
    id: str
    type: EntityType
    title: str
    aliases: list[str] = []
    regions: list[str] = []
    privacy: Privacy | None = None  # None: derive from regions (config/life-map.yaml)
    summary: str = ""


class AddFact(FactSpec):
    op: Literal["add_fact"] = "add_fact"
    subject: str  # entity id or "self"


class SupersedeFact(BaseModel):
    op: Literal["supersede_fact"] = "supersede_fact"
    fact_id: str
    to: str | None = None  # end of validity of the old fact
    new: FactSpec | None = None

    _check_to = field_validator("to")(check_partial_date)


class MergeEntities(BaseModel):
    op: Literal["merge_entities"] = "merge_entities"
    keep: str
    merge: str


class DemoteEntity(BaseModel):
    """The page names a value, not an entity (a topic, tool or hobby made into a project or
    organisation page): every link to it becomes the literal ``value`` (default: its title) and
    the page is removed. Only a page without facts or notes of its own can be demoted (schema
    v0.5). Always reviewed."""

    op: Literal["demote_entity"] = "demote_entity"
    id: str
    value: str | None = None


class RetypeEntity(BaseModel):
    """The page has the wrong type: it moves to ``new_id`` (default: the type's folder + the
    same slug) with ``type``, and every link is repointed, as in a merge. Always reviewed."""

    op: Literal["retype_entity"] = "retype_entity"
    id: str
    type: EntityType
    new_id: str | None = None


class RefileFact(BaseModel):
    """File a fact differently without changing what it says (schema v0.5): on another
    ``subject``, under another predicate ``p``, with its literal object ``o`` linked to the page
    it names, or with ``at`` (the ``[[orgs/…]]`` a duty, role or degree belongs to). Its text,
    dates, kind and id stay; the change set's sources join ``src::``. Always reviewed."""

    op: Literal["refile_fact"] = "refile_fact"
    fact_id: str
    subject: str | None = None
    p: str | None = None
    o: str | None = None
    at: str | None = None


class UpdateSummary(BaseModel):
    op: Literal["update_summary"] = "update_summary"
    id: str
    summary: str


class AddAlias(BaseModel):
    op: Literal["add_alias"] = "add_alias"
    id: str
    alias: str


class AddIntention(BaseModel):
    op: Literal["add_intention"] = "add_intention"
    text: str
    when: list[str] = []
    after: str | None = None

    _check_after = field_validator("after")(check_partial_date)

    @field_validator("when", mode="before")
    @classmethod
    def _one_or_many(cls, value: object) -> object:
        return [value] if isinstance(value, str) else value


class AddObservation(BaseModel):
    op: Literal["add_observation"] = "add_observation"
    text: str
    importance: int = Field(ge=1, le=10)


class ConfirmFact(BaseModel):
    """The owner restated a fact the vault holds: the change set's sources are added to its
    ``src::`` (nothing else changes). Used by the memory pipeline for ``confirms``."""

    op: Literal["confirm_fact"] = "confirm_fact"
    fact_id: str


class AddEpisode(BaseModel):
    """Write ``episodes/YYYY/MM/<id>.md`` (docs/vault-schema.md, Episodes). Memory pipeline only:
    the big model never proposes episodes."""

    op: Literal["add_episode"] = "add_episode"
    id: str  # ep-YYYYMMDD-HHMM-<slug>
    session: str
    turns: tuple[int, int]
    channel: Channel
    provider: str | None = None
    regions: list[str] = []
    entities: list[str] = []  # entity ids
    importance: int = Field(ge=1, le=10)
    gist: str
    started: datetime
    status: EpisodeStatus = "pending"
    privacy: Privacy = "normal"
    summary: str = ""


class UpdateEpisode(BaseModel):
    """Change an episode's status (pending → extracted → consolidated) and add entity links
    (entities created by its extraction). Memory pipeline only."""

    op: Literal["update_episode"] = "update_episode"
    id: str
    status: EpisodeStatus | None = None
    add_entities: list[str] = []


Operation = Annotated[
    CreateEntity
    | AddFact
    | SupersedeFact
    | MergeEntities
    | DemoteEntity
    | RetypeEntity
    | RefileFact
    | UpdateSummary
    | AddAlias
    | AddIntention
    | AddObservation
    | ConfirmFact
    | AddEpisode
    | UpdateEpisode,
    Field(discriminator="op"),
]

Origin = Literal["ingest", "extract", "consolidate", "external", "user", "lint"]


class ChangeSet(BaseModel):
    """A batch of operations from one source, applied atomically (one vault commit)."""

    ops: list[Operation]
    origin: Origin
    src: list[str]  # provenance stamped on every fact/observation: episode id, raw doc path...
    rationale: str = ""
    created: datetime = Field(default_factory=datetime.now)
