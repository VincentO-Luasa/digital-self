"""One structured LLM call per chunk: excerpt + known entities → a flat extraction.

The schema the model fills is deliberately flat (names, not ids or a union of operation types):
a ~3B-active model is reliable at "list the facts", less so at producing discriminated unions.
``brain.ingest.normalise`` turns extractions into vault operations in code.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel

from brain.ingest.chunking import Chunk
from brain.prompts import load_prompt
from brain.vault.models import PREDICATES, SELF_ID, Page

T = TypeVar("T", bound=BaseModel)

# Predicates offered for documents: a subset of the vocabulary. Traits, flaws, health and
# private relationship details are left to conversations, where the owner states them.
INGEST_PREDICATES = (
    "name",
    "born",
    "hometown",
    "lives_in",
    "nationality",
    "speaks",
    "works_at",
    "role",
    "team",
    "manager",
    "responsibility",
    "worked_on",
    "studied_at",
    "degree",
    "certification",
    "has_skill",
    "values",
    "interest",
    "likes",
    "goal",
    "owner",
    "status",
    "deadline",
    "stakeholder",
    "uses",
    "friend_of",
    "family_of",
    "colleague_of",
    "mentor_of",
    "knows",
    "member_of",
    "located_in",
    "part_of",
)
assert set(INGEST_PREDICATES) <= PREDICATES

Predicate = Literal[INGEST_PREDICATES]  # type: ignore[valid-type]
EntityKind = Literal["person", "org", "project", "place", "topic", "event"]

MAX_KNOWN = 40
MAX_TOKENS = 6144


class ExtractedEntity(BaseModel):
    name: str
    type: EntityKind
    id: str
    aliases: list[str]
    summary: str


class ExtractedFact(BaseModel):
    subject: str
    predicate: Predicate
    object: str
    text: str
    start: str
    end: str


class Extraction(BaseModel):
    reasoning: str
    owner_names: list[str]
    entities: list[ExtractedEntity]
    facts: list[ExtractedFact]


class StructuredModel(Protocol):
    def structured(
        self,
        messages: Sequence[Mapping[str, object]],
        schema: type[T],
        *,
        temperature: float | None = None,
        max_tokens: int = 2048,
    ) -> T: ...


@dataclass(frozen=True)
class KnownEntity:
    id: str
    names: tuple[str, ...]


def known_entities(pages: Iterable[Page]) -> dict[str, KnownEntity]:
    """Entity id → its title and aliases (the same names ``Vault.alias_index()`` indexes).

    Self pages are left out: the owner is always listed as ``self``.
    """
    names: dict[str, list[str]] = {}
    for page in pages:
        if page.subject == SELF_ID:
            continue
        bucket = names.setdefault(page.subject, [])
        for name in (page.title, *page.aliases):
            if name.strip() and name.casefold() not in {n.casefold() for n in bucket}:
                bucket.append(name.strip())
    return {i: KnownEntity(i, tuple(n)) for i, n in sorted(names.items())}


def owner_names(pages: Iterable[Page]) -> list[str]:
    """Names of the owner from the self pages (the default title "Profile" is not a name)."""
    out: list[str] = []
    for page in pages:
        if page.subject != SELF_ID:
            continue
        for name in (page.title, *page.aliases):
            if name.strip() and name != "Profile" and name.casefold() not in map(str.casefold, out):
                out.append(name.strip())
    return out


def relevant_known(
    known: Mapping[str, KnownEntity], text: str, always: Iterable[str] = ()
) -> list[KnownEntity]:
    """Known entities named in ``text`` (plus ``always``), capped at ``MAX_KNOWN``.

    Sending the whole alias index would not scale and would bloat the prompt.
    """
    folded = text.casefold()
    keep = set(always)
    for entity_id, entity in known.items():
        if entity_id == SELF_ID:
            continue
        names = (n.casefold() for n in entity.names if n)
        if any(re.search(rf"(?<!\w){re.escape(n)}(?!\w)", folded) for n in names):
            keep.add(entity_id)
    return [known[i] for i in sorted(keep) if i in known and i != SELF_ID][:MAX_KNOWN]


def render_known(entities: Sequence[KnownEntity], owner_names: Sequence[str]) -> str:
    owner = ", ".join(owner_names) if owner_names else "name unknown"
    lines = [f"- self: the owner ({owner})"]
    lines += [f"- {e.id}: {', '.join(e.names)}" for e in entities]
    return "\n".join(lines)


def build_messages(
    chunk: Chunk,
    *,
    doc_title: str,
    known: Sequence[KnownEntity],
    owner_names: Sequence[str],
    today: date,
) -> list[dict[str, str]]:
    user = (
        f"Today is {today.isoformat()}.\n"
        f"Document: {doc_title}\n"
        f"Section: {chunk.heading or '(start)'}\n\n"
        f"KNOWN ENTITIES (id: names):\n{render_known(known, owner_names)}\n\n"
        f"EXCERPT:\n<<<\n{chunk.text}\n>>>"
    )
    return [
        {"role": "system", "content": load_prompt("ingest_extract")},
        {"role": "user", "content": user},
    ]


def extract_chunk(
    llm: StructuredModel,
    chunk: Chunk,
    *,
    doc_title: str,
    known: Sequence[KnownEntity],
    owner_names: Sequence[str],
    today: date,
) -> Extraction:
    messages = build_messages(
        chunk, doc_title=doc_title, known=known, owner_names=owner_names, today=today
    )
    return llm.structured(messages, Extraction, max_tokens=MAX_TOKENS)
