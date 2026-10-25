"""Record real request/response pairs as JSON fixtures, and load them back for replay.

Use only with fictional or generic inputs: fixtures end up in the public repo
(``tests/fixtures/llm/``). Each file holds one exchange::

    {"kind": "structured", "request": {...}, "response": {...}}

``FakeLLM.from_fixtures`` replays the recorded chat/structured outputs.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Exchange:
    kind: str  # chat | structured | rerank
    request: dict[str, Any]
    response: dict[str, Any]

    @property
    def content(self) -> str:
        """Assistant text of a recorded chat/structured completion."""
        choices = self.response.get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content") or ""


class Recorder:
    def __init__(self, directory: Path, *, prefix: str = "") -> None:
        self.directory = directory
        self.prefix = prefix

    def save(self, kind: str, request: dict[str, Any], response: dict[str, Any]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        body = json.dumps(request, sort_keys=True, default=str)
        digest = hashlib.sha1(body.encode()).hexdigest()[:8]
        name = re.sub(r"[^a-z0-9_-]+", "-", f"{self.prefix}{kind}".lower()).strip("-")
        path = self.directory / f"{name}-{digest}.json"
        payload = {"kind": kind, "request": request, "response": response}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
        return path


def load_exchange(path: Path) -> Exchange:
    data = json.loads(path.read_text())
    return Exchange(kind=data["kind"], request=data["request"], response=data["response"])
