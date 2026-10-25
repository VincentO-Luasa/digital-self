"""Operations: the only way memory changes.

Models and the big model *propose* operations; ``brain.vault.writer`` validates and applies them
(or queues them for review), appends to log.md and commits the vault. The JSON shape is mirrored
in vault_template/AGENTS.md.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from brain.vault.models import EntityType, Kind, Privacy, check_partial_date


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


Operation = Annotated[
    CreateEntity
    | AddFact
    | SupersedeFact
    | MergeEntities
    | UpdateSummary
    | AddAlias
    | AddIntention
    | AddObservation,
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
