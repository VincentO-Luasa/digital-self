"""The only module that talks to the model server (OpenAI-compatible; oMLX by default).

Runtime rules (see the ``local-llm`` skill): thinking is always disabled, sampling is always sent
explicitly, structured output uses a strict ``json_schema`` response format, embeddings for
queries get the Qwen3-Embedding instruct prefix, and rerank is a plain ``POST /v1/rerank``.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Iterator, Sequence
from typing import Any, Literal, TypeVar, overload

import httpx
import numpy as np
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from brain.config import LLMSettings
from brain.llm.base import (
    EMBED_BATCH,
    QUERY_INSTRUCT,
    CallCancelled,
    ChatResult,
    Messages,
    OutputTruncated,
    StructuredOutputError,
    ToolCall,
    l2_normalise,
    parse_arguments,
)
from brain.llm.decide import (
    Decision,
    Question,
    fallback_decisions,
    fallback_messages,
    fallback_schema,
    parse_answers,
    request_payload,
)
from brain.llm.recorder import Recorder

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
MAX_ERROR_CHARS = 1500
NO_ENDPOINT = (404, 405, 501)  # the runtime has no /v1/decisions


def server_root(base_url: str) -> str:
    """``http://host:port/v1`` → ``http://host:port`` (``/health`` lives at the root)."""
    return re.sub(r"/v1/?$", "", base_url.rstrip("/"))


def response_format(schema: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "strict": True,
            "schema": schema.model_json_schema(),
        },
    }


def retry_message(error: ValidationError) -> dict[str, str]:
    details = "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in error.errors()
    )
    return {
        "role": "user",
        "content": (
            "Your JSON did not validate: "
            f"{details[:MAX_ERROR_CHARS]}\nReply again with corrected JSON only."
        ),
    }


def _no_endpoint(resp: httpx.Response) -> bool:
    """The route does not exist (as opposed to e.g. an unknown model, a 404 with an API error)."""
    if resp.status_code not in NO_ENDPOINT:
        return False
    try:
        body = resp.json()
    except ValueError:
        return True
    return not (isinstance(body, dict) and "error" in body)


class LLM:
    """Client for chat, structured output, embeddings and rerank.

    ``transport`` lets tests inject an ``httpx.MockTransport``; ``recorder`` saves exchanges as
    replayable fixtures.
    """

    def __init__(
        self,
        settings: LLMSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        recorder: Recorder | None = None,
        max_retries: int = 1,
    ) -> None:
        self.settings = settings
        self.recorder = recorder
        self._http = httpx.Client(timeout=settings.timeout_s, transport=transport)
        self._client = OpenAI(
            base_url=settings.base_url,
            api_key="local",
            http_client=self._http,
            max_retries=max_retries,
        )
        self._dim: int | None = None
        self._decisions: bool | None = None  # does the server have /v1/decisions? (learned)

    # -- lifecycle ---------------------------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LLM:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def embed_model(self) -> str:
        return self.settings.embed_model

    @property
    def embed_dim(self) -> int:
        """Embedding size, learned from the server on first use."""
        if self._dim is None:
            self._dim = int(self.embed(["dimension probe"]).shape[1])
        return self._dim

    # -- chat --------------------------------------------------------------------------------

    def _chat_kwargs(
        self,
        messages: Messages,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        s = self.settings
        return {
            "model": s.chat_model,
            "messages": [dict(m) for m in messages],
            "temperature": s.chat_temperature if temperature is None else temperature,
            "top_p": s.chat_top_p if top_p is None else top_p,
            "max_tokens": max_tokens,
            "extra_body": {**NO_THINKING, "top_k": s.chat_top_k},
        }

    @overload
    def chat(
        self,
        messages: Messages,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 1024,
        tools: None = None,
        stream: Literal[False] = False,
    ) -> str: ...

    @overload
    def chat(
        self,
        messages: Messages,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 1024,
        tools: None = None,
        stream: Literal[True],
    ) -> Iterator[str]: ...

    @overload
    def chat(
        self,
        messages: Messages,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 1024,
        tools: Sequence[dict[str, Any]],
        stream: Literal[False] = False,
    ) -> ChatResult: ...

    def chat(
        self,
        messages: Messages,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 1024,
        tools: Sequence[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> str | Iterator[str] | ChatResult:
        """Text completion. ``stream=True`` yields text deltas; ``tools`` returns a ChatResult."""
        kwargs = self._chat_kwargs(messages, temperature, top_p, max_tokens)
        if tools:
            if stream:
                raise ValueError("streaming with tools is not supported")
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = "auto"
        if stream:
            return self._stream(kwargs)
        resp = self._complete("chat", kwargs)
        choice = resp.choices[0]
        if tools is None:
            return choice.message.content or ""
        calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=parse_arguments(tc.function.arguments),
                raw_arguments=tc.function.arguments or "",
            )
            for tc in choice.message.tool_calls or []
            if tc.type == "function"
        ]
        return ChatResult(choice.message.content, calls, choice.finish_reason)

    def _complete(self, kind: str, kwargs: dict[str, Any]) -> Any:
        start = time.perf_counter()
        resp = self._client.chat.completions.create(**kwargs)
        usage = resp.usage
        log.debug(
            "%s %s: %.2fs prompt=%s completion=%s finish=%s",
            kind,
            kwargs["model"],
            time.perf_counter() - start,
            usage.prompt_tokens if usage else "?",
            usage.completion_tokens if usage else "?",
            resp.choices[0].finish_reason,
        )
        if self.recorder:
            self.recorder.save(kind, kwargs, resp.model_dump(exclude_none=True))
        return resp

    def _stream(self, kwargs: dict[str, Any]) -> Iterator[str]:
        start = time.perf_counter()
        first: float | None = None
        pieces = 0
        for chunk in self._client.chat.completions.create(**kwargs, stream=True):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                if first is None:
                    first = time.perf_counter() - start
                pieces += 1
                yield delta
        log.debug(
            "chat-stream %s: ttft=%.2fs total=%.2fs chunks=%d",
            kwargs["model"],
            first or 0.0,
            time.perf_counter() - start,
            pieces,
        )

    def _cancellable(self, kwargs: dict[str, Any], cancel: threading.Event) -> tuple[str, Any]:
        """A streamed completion: (text, finish reason); ``CallCancelled`` once ``cancel`` is
        set (closing the stream makes the server stop generating)."""
        if cancel.is_set():
            raise CallCancelled
        start = time.perf_counter()
        stream = self._client.chat.completions.create(**kwargs, stream=True)
        pieces: list[str] = []
        finish = None
        try:
            for chunk in stream:
                if cancel.is_set():
                    log.debug("structured stream cancelled: %.2fs", time.perf_counter() - start)
                    raise CallCancelled
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.delta.content:
                    pieces.append(choice.delta.content)
                finish = choice.finish_reason or finish
        finally:
            stream.close()
        log.debug("structured stream %s: %.2fs", kwargs["model"], time.perf_counter() - start)
        return "".join(pieces), finish

    # -- structured --------------------------------------------------------------------------

    def structured(
        self,
        messages: Messages,
        schema: type[T],
        *,
        temperature: float | None = None,
        max_tokens: int = 2048,
        cancel: threading.Event | None = None,
    ) -> T:
        """Grammar-enforced JSON parsed into ``schema``; one retry with the validation error.

        Raises ``OutputTruncated`` at once when the reply hits ``max_tokens``. With ``cancel``
        the reply is streamed and the call raises ``CallCancelled`` as soon as the event is set
        (checked per chunk: a prefill under way finishes first); background work uses it to
        get out of the way of the owner.
        """
        temp = self.settings.extract_temperature if temperature is None else temperature
        kwargs = self._chat_kwargs(messages, temp, None, max_tokens)
        kwargs["response_format"] = response_format(schema)
        raw = ""
        for attempt in range(2):
            if cancel is not None:
                raw, finish = self._cancellable(kwargs, cancel)
            else:
                resp = self._complete("structured", kwargs)
                raw = resp.choices[0].message.content or ""
                finish = resp.choices[0].finish_reason
            if finish == "length":
                raise OutputTruncated(
                    f"{schema.__name__}: reply cut off at max_tokens={max_tokens}", raw
                )
            try:
                return schema.model_validate_json(raw)
            except ValidationError as err:
                log.debug("structured %s: attempt %d invalid: %s", schema.__name__, attempt, err)
                kwargs["messages"] = [
                    *kwargs["messages"],
                    {"role": "assistant", "content": raw},
                    retry_message(err),
                ]
        raise StructuredOutputError(f"{schema.__name__}: invalid JSON after retry", raw)

    # -- decisions ---------------------------------------------------------------------------

    def decide(self, text: str, questions: Sequence[Question]) -> list[Decision]:
        """Answer closed questions about ``text``: one ``Decision`` per question, in order.

        Uses ``POST /v1/decisions`` (a probability per option, one token per question). A
        server without it (404/405/501) gets one ``structured`` call instead, answers marked
        ``fallback``; that is remembered for the client's lifetime.
        """
        if not questions:
            return []
        if self._decisions is not False:
            payload = request_payload(self.settings.chat_model, text, questions)
            start = time.perf_counter()
            resp = self._http.post(f"{self.settings.base_url.rstrip('/')}/decisions", json=payload)
            if not _no_endpoint(resp) or self._decisions:
                resp.raise_for_status()
                data = resp.json()
                self._decisions = True
                log.debug(
                    "decide %d questions: %.2fs input=%s",
                    len(questions),
                    time.perf_counter() - start,
                    (data.get("usage") or {}).get("input_tokens", "?"),
                )
                if self.recorder:
                    self.recorder.save("decide", payload, data)
                return parse_answers(data, questions)
            log.info("no /v1/decisions (HTTP %d): structured fallback", resp.status_code)
            self._decisions = False
        schema = fallback_schema(questions)
        reply = self.structured(
            fallback_messages(text, questions), schema, temperature=0.0, max_tokens=512
        )
        return fallback_decisions(reply, questions)

    # -- embeddings / rerank -----------------------------------------------------------------

    def embed(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray:
        """Float32, L2-normalised, shape ``(len(texts), dim)``."""
        if not texts:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        inputs = [QUERY_INSTRUCT + t if query else t for t in texts]
        rows: list[list[float]] = []
        start = time.perf_counter()
        for i in range(0, len(inputs), EMBED_BATCH):
            resp = self._client.embeddings.create(
                model=self.settings.embed_model,
                input=inputs[i : i + EMBED_BATCH],
                encoding_format="float",
            )
            rows.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        out = l2_normalise(np.asarray(rows, dtype=np.float32))
        self._dim = out.shape[1]
        log.debug(
            "embed %d texts (query=%s): %.2fs", len(texts), query, time.perf_counter() - start
        )
        return out

    def rerank(
        self, query: str, documents: Sequence[str], top_n: int | None = None
    ) -> list[tuple[int, float]]:
        """``[(document index, relevance)]``, best first."""
        if not documents:
            return []
        payload = {
            "model": self.settings.rerank_model,
            "query": query,
            "documents": list(documents),
            "top_n": top_n or len(documents),
        }
        start = time.perf_counter()
        resp = self._http.post(f"{self.settings.base_url.rstrip('/')}/rerank", json=payload)
        resp.raise_for_status()
        data = resp.json()
        log.debug(
            "rerank %d docs: %.2fs tokens=%s",
            len(documents),
            time.perf_counter() - start,
            (data.get("usage") or {}).get("total_tokens", "?"),
        )
        if self.recorder:
            self.recorder.save("rerank", payload, data)
        results = [(int(r["index"]), float(r["relevance_score"])) for r in data["results"]]
        return sorted(results, key=lambda r: r[1], reverse=True)

    def health(self) -> bool:
        try:
            resp = self._http.get(f"{server_root(self.settings.base_url)}/health", timeout=3.0)
        except httpx.HTTPError:
            return False
        return resp.status_code == 200
