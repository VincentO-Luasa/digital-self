"""The vault: parse/render markdown, read-only store, and the single writer."""

from brain.vault.generated import build_core, build_index, refresh_generated
from brain.vault.init import InitResult, init_vault
from brain.vault.lifemap import LifeMap, Region, Slot, load_lifemap
from brain.vault.models import Episode, Fact, Intention, Observation, Page
from brain.vault.ops import ChangeSet, Operation
from brain.vault.parse import ParseError, ParseWarning
from brain.vault.raw import Transcript, parse_transcript, save_doc
from brain.vault.store import Vault
from brain.vault.writer import (
    ApplyResult,
    Review,
    ReviewNotFound,
    apply,
    approve,
    needs_review,
    pending_reviews,
    reject,
)

__all__ = [
    "ApplyResult",
    "ChangeSet",
    "Episode",
    "Fact",
    "InitResult",
    "Intention",
    "LifeMap",
    "Observation",
    "Operation",
    "Page",
    "ParseError",
    "ParseWarning",
    "Region",
    "Review",
    "ReviewNotFound",
    "Slot",
    "Transcript",
    "Vault",
    "apply",
    "approve",
    "build_core",
    "build_index",
    "init_vault",
    "load_lifemap",
    "needs_review",
    "parse_transcript",
    "pending_reviews",
    "refresh_generated",
    "reject",
    "save_doc",
]
