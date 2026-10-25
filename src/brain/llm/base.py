"""Types shared by the real client and the fake: results, errors, the index-facing protocol."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

Message = Mapping[str, Any]
Messages = Sequence[Message]

QUERY_INSTRUCT = (
    "Instruct: Given a question about the owner's life, retrieve relevant memories\nQuery: "
)
EMBED_BATCH = 32


class StructuredOutputError(ValueError):
    """The model's JSON did not validate against the schema, even after one retry."""

    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass(frozen=True)
class ChatResult:
    """A non-streamed completion that may contain tool calls (returned when ``tools`` is set)."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None

    def message(self) -> dict[str, Any]:
        """The assistant message to append to the history before sending tool results."""
        msg: dict[str, Any] = {"role": "assistant"}
        if self.content is not None:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.raw_arguments or json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


def parse_arguments(raw: str | None) -> dict[str, Any]:
    """Tool-call arguments as a dict; malformed JSON yields an empty dict (raw is kept)."""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class RetrievalModel(Protocol):
    """What the index needs from a model client (``LLM`` and ``FakeLLM`` both satisfy it)."""

    @property
    def embed_model(self) -> str: ...

    @property
    def embed_dim(self) -> int: ...

    def embed(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray: ...

    def rerank(
        self, query: str, documents: Sequence[str], top_n: int | None = None
    ) -> list[tuple[int, float]]: ...
