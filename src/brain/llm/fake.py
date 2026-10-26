"""Deterministic stand-in for ``LLM`` in tests: no server, same interface.

- embeddings: hashed bag of words (texts sharing words have high cosine similarity);
- chat / structured: scripted responses, consumed in order (or replayed from fixtures);
- rerank: lexical overlap with the query;
- decide: scripted answers (a list of ``Decision``s, a recorded ``/v1/decisions`` response, or a
  ``decider(text, questions)`` callable); unscripted questions are refusals ("no answer").
"""

from __future__ import annotations

import hashlib
import re
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel, ValidationError

from brain.config import LLMSettings
from brain.llm.base import ChatResult, Messages, StructuredOutputError, l2_normalise
from brain.llm.decide import Decision, Question, parse_answers, refusals
from brain.llm.recorder import load_exchange

T = TypeVar("T", bound=BaseModel)
WORD = re.compile(r"\w+")

Scripted = str | dict[str, Any] | BaseModel | Exception  # an Exception is raised
Decider = Callable[[str, Sequence[Question]], Sequence[Decision | dict[str, Any]]]
# one decide() call: its decisions, a raw /v1/decisions response, or an Exception to raise
ScriptedDecisions = Sequence[Decision | dict[str, Any]] | dict[str, Any] | Exception


def words(text: str) -> list[str]:
    return WORD.findall(text.casefold())


def _bucket(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


def hash_embedding(text: str, dim: int) -> np.ndarray:
    """Hashed bag of words. Of an instructed text ("Instruct: …\nQuery: …") only the query part
    is embedded: an instruction shared by every text would otherwise dominate the bag."""
    if text.startswith("Instruct:") and "\nQuery: " in text:
        text = text.split("\nQuery: ", 1)[1]
    vec = np.zeros(dim, dtype=np.float32)
    for token in words(text) or [text or "<empty>"]:
        index, sign = _bucket(token, dim)
        vec[index] += sign
    return vec


class FakeLLM:
    def __init__(
        self,
        settings: LLMSettings | None = None,
        *,
        dim: int = 1024,
        chat_responses: Iterable[str | ChatResult] = (),
        structured_responses: Iterable[Scripted] = (),
        decisions: Iterable[ScriptedDecisions] = (),
        decider: Decider | None = None,
        healthy: bool = True,
        default_reply: str = "ok",
    ) -> None:
        self.settings = settings or LLMSettings()
        self.dim = dim
        self.healthy = healthy
        self.default_reply = default_reply
        self.chat_queue: deque[str | ChatResult] = deque(chat_responses)
        self.structured_queue: deque[Scripted] = deque(structured_responses)
        self.decision_queue: deque[ScriptedDecisions] = deque(decisions)
        self.decider = decider
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.embedded: list[str] = []  # every text ever embedded, in order

    @classmethod
    def from_fixtures(cls, paths: Iterable[Path], **kwargs: Any) -> FakeLLM:
        """Replay recorded chat/structured completions (see ``brain.llm.recorder``)."""
        fake = cls(**kwargs)
        for exchange in map(load_exchange, paths):
            if exchange.kind == "chat":
                fake.chat_queue.append(exchange.content)
            elif exchange.kind == "structured":
                fake.structured_queue.append(exchange.content)
            elif exchange.kind == "decide":
                fake.decision_queue.append(exchange.response)
        return fake

    @property
    def embed_model(self) -> str:
        return self.settings.embed_model

    @property
    def embed_dim(self) -> int:
        return self.dim

    def chat(
        self,
        messages: Messages,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 1024,
        tools: Sequence[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> Any:
        self.calls.append(("chat", {"messages": list(messages), "tools": tools, "stream": stream}))
        reply = self.chat_queue.popleft() if self.chat_queue else self.default_reply
        if tools is not None:
            return reply if isinstance(reply, ChatResult) else ChatResult(reply, [], "stop")
        text = (reply.content or "") if isinstance(reply, ChatResult) else reply
        return self._stream(text) if stream else text

    @staticmethod
    def _stream(text: str) -> Iterator[str]:
        yield from re.findall(r"\S+\s*|\s+", text)

    def structured(
        self,
        messages: Messages,
        schema: type[T],
        *,
        temperature: float | None = None,
        max_tokens: int = 2048,
    ) -> T:
        self.calls.append(("structured", {"messages": list(messages), "schema": schema}))
        if not self.structured_queue:
            raise LookupError(f"FakeLLM: no scripted response for {schema.__name__}")
        item = self.structured_queue.popleft()
        if isinstance(item, Exception):
            raise item
        try:
            if isinstance(item, BaseModel):
                return schema.model_validate(item.model_dump())
            if isinstance(item, dict):
                return schema.model_validate(item)
            return schema.model_validate_json(item)
        except ValidationError as err:
            raise StructuredOutputError(str(err), str(item)) from err

    def decide(self, text: str, questions: Sequence[Question]) -> list[Decision]:
        if not questions:
            return []
        self.calls.append(("decide", {"text": text, "questions": list(questions)}))
        if self.decision_queue:
            item: Any = self.decision_queue.popleft()
        elif self.decider is not None:
            item = self.decider(text, questions)
        else:
            return refusals(questions)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, dict):
            return parse_answers(item, questions)
        given = {d.name: d for d in (Decision.model_validate(x) for x in item)}
        return [given.get(q.name) or refusals([q])[0] for q in questions]

    def embed(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray:
        self.calls.append(("embed", {"n": len(texts), "query": query}))
        if not query:
            self.embedded.extend(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalise(np.stack([hash_embedding(t, self.dim) for t in texts]))

    def rerank(
        self, query: str, documents: Sequence[str], top_n: int | None = None
    ) -> list[tuple[int, float]]:
        self.calls.append(("rerank", {"query": query, "n": len(documents)}))
        q = set(words(query))
        scored = [(i, len(q & set(words(doc))) / (len(q) or 1)) for i, doc in enumerate(documents)]
        scored.sort(key=lambda r: (-r[1], r[0]))
        return scored[: top_n or len(documents)]

    def health(self) -> bool:
        return self.healthy

    def close(self) -> None:
        pass
