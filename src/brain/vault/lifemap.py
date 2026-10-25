"""The life map: regions and slots (config/life-map.yaml, overridable per vault)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from brain.config import REPO_DIR
from brain.vault.models import Privacy

DEFAULT_LIFEMAP = REPO_DIR / "config" / "life-map.yaml"


class Slot(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    p: str
    subject: str  # "self" or an entity type
    ask: str = ""
    stale_days: int | None = None
    multi: bool = False
    min: int = 1


class Region(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    label: str = ""
    weight: float = 1.0
    privacy: Privacy = "normal"
    per_entity: str | None = None
    slots: tuple[Slot, ...] = ()


class LifeMap(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str = "0.1"
    regions: dict[str, Region] = {}

    def region(self, region_id: str) -> Region | None:
        """Exact region, or the nearest dotted ancestor (``a.b.c`` → ``a.b`` → ``a``)."""
        parts = region_id.split(".")
        for n in range(len(parts), 0, -1):
            if found := self.regions.get(".".join(parts[:n])):
                return found
        return None

    def privacy_for(self, regions: Iterable[str]) -> Privacy:
        """Private if any region (or its ancestor) is private."""
        for region_id in regions:
            region = self.region(region_id)
            if region and region.privacy == "private":
                return "private"
        return "normal"

    def slots(self, region_id: str) -> tuple[Slot, ...]:
        region = self.regions.get(region_id)
        return region.slots if region else ()

    def region_for(self, p: str, subject: str) -> str | None:
        """First region with a slot for predicate ``p`` about ``subject`` (``self`` or a type)."""
        for region in self.regions.values():
            if any(s.p == p and s.subject == subject for s in region.slots):
                return region.id
        return None


def parse_lifemap(text: str) -> LifeMap:
    data = yaml.safe_load(text) or {}
    regions = {
        rid: Region.model_validate({"id": rid, **(body or {})})
        for rid, body in (data.get("regions") or {}).items()
    }
    return LifeMap(version=str(data.get("version", "0.1")), regions=regions)


def load_lifemap(vault_root: Path | None = None, default: Path = DEFAULT_LIFEMAP) -> LifeMap:
    """``<vault>/life-map.yaml`` if present, else the repo's config/life-map.yaml."""
    override = vault_root / "life-map.yaml" if vault_root else None
    path = override if override and override.exists() else default
    return parse_lifemap(path.read_text(encoding="utf-8"))
