"""Replay identical structured requests from disk.

Ingest uses it so that changing normalisation or the writer doesn't mean re-running the model
over every chunk (minutes per document). The key covers the model, schema, messages and
sampling, so a new prompt or excerpt is a miss. Lives under the vault's ``.brain/`` (derived,
never committed).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from brain.llm.decide import Decision, Question

DECISIONS = TypeAdapter(list[Decision])

T = TypeVar("T", bound=BaseModel)


class StructuredCache:
    """Wraps a model: ``structured`` and ``decide`` are cached, everything else is passed
    through."""

    def __init__(self, llm: Any, directory: Path) -> None:
        self.llm = llm
        self.directory = directory
        self.hits = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.llm, name)

    def _path(self, messages: Sequence[Mapping[str, Any]], schema: type[BaseModel], **kw) -> Path:
        settings = getattr(self.llm, "settings", None)
        request = {
            "model": getattr(settings, "chat_model", None),
            "schema": schema.model_json_schema(),
            "messages": [dict(m) for m in messages],
            **kw,
        }
        blob = json.dumps(request, sort_keys=True, default=str).encode()
        return self.directory / f"{schema.__name__}-{hashlib.sha256(blob).hexdigest()[:24]}.json"

    def structured(
        self,
        messages: Sequence[Mapping[str, Any]],
        schema: type[T],
        *,
        temperature: float | None = None,
        max_tokens: int = 2048,
    ) -> T:
        path = self._path(messages, schema, temperature=temperature, max_tokens=max_tokens)
        if path.exists():
            try:
                out = schema.model_validate_json(path.read_text(encoding="utf-8"))
            except ValidationError:
                path.unlink(missing_ok=True)
            else:
                self.hits += 1
                return out
        out = self.llm.structured(messages, schema, temperature=temperature, max_tokens=max_tokens)
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(out.model_dump_json(), encoding="utf-8")
        tmp.replace(path)
        return out

    def decide(self, text: str, questions: Sequence[Question]) -> list[Decision]:
        if not questions:
            return []
        settings = getattr(self.llm, "settings", None)
        request = {
            "model": getattr(settings, "chat_model", None),
            "input": text,
            "questions": [q.model_dump() for q in questions],
        }
        blob = json.dumps(request, sort_keys=True, default=str).encode()
        path = self.directory / f"decide-{hashlib.sha256(blob).hexdigest()[:24]}.json"
        if path.exists():
            try:
                out = DECISIONS.validate_json(path.read_text(encoding="utf-8"))
            except ValidationError:
                path.unlink(missing_ok=True)
            else:
                self.hits += 1
                return out
        out = list(self.llm.decide(text, questions))
        if not any(d.refusal for d in out):  # a refusal may be a transient no-answer: re-ask
            self.directory.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(DECISIONS.dump_json(out))
            tmp.replace(path)
        return out
