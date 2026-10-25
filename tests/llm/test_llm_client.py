"""Request building and response handling of ``brain.llm.LLM`` against a mocked server."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import numpy as np
import pytest
from pydantic import BaseModel

from brain.config import LLMSettings
from brain.llm import LLM, ChatResult, StructuredOutputError
from brain.llm.base import QUERY_INSTRUCT
from brain.llm.recorder import Recorder, load_exchange

Handler = Callable[[httpx.Request], httpx.Response]


class Answer(BaseModel):
    reasoning: str
    answer: str


def completion(content: str | None = "hello", **message: Any) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", **message}
    if content is not None:
        msg["content"] = content
    finish = "tool_calls" if "tool_calls" in message else "stop"
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "chat-model",
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


class Server:
    """Records requests; replies from a queue of payloads (or a callable)."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if callable(reply):
            return reply(request)
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(200, json=reply)

    def body(self, i: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[i].content)


def make_llm(server: Handler, **kw: Any) -> LLM:
    return LLM(LLMSettings(), transport=httpx.MockTransport(server), **kw)


def test_chat_payload_disables_thinking_and_sends_sampling():
    server = Server(completion("hi there"))
    llm = make_llm(server)
    assert llm.chat([{"role": "user", "content": "hello"}]) == "hi there"
    body = server.body()
    assert server.requests[0].url.path == "/v1/chat/completions"
    assert body["model"] == "Qwen3.6-35B-A3B-4bit"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["temperature"] == 0.7
    assert body["top_p"] == 0.8
    assert body["max_tokens"] == 1024
    assert "tools" not in body and "response_format" not in body


def test_chat_overrides_sampling():
    server = Server(completion())
    make_llm(server).chat([{"role": "user", "content": "x"}], temperature=0.1, max_tokens=64)
    assert server.body()["temperature"] == 0.1
    assert server.body()["max_tokens"] == 64


def test_chat_with_tools_parses_calls_without_content():
    call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"entity": "projects/atlas-migration"}'},
    }
    server = Server(completion(None, tool_calls=[call]))
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    result = make_llm(server).chat([{"role": "user", "content": "x"}], tools=tools)
    assert isinstance(result, ChatResult)
    assert result.finish_reason == "tool_calls"
    assert result.content is None
    assert result.tool_calls[0].name == "lookup"
    assert result.tool_calls[0].arguments == {"entity": "projects/atlas-migration"}
    assert "content" not in result.message()
    assert result.message()["tool_calls"][0]["function"]["name"] == "lookup"
    body = server.body()
    assert body["tools"] == tools and body["tool_choice"] == "auto"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_chat_stream_yields_deltas():
    def sse(_: httpx.Request) -> httpx.Response:
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]},
        ]
        base = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m"}
        lines = [f"data: {json.dumps(base | c)}\n\n" for c in chunks] + ["data: [DONE]\n\n"]
        return httpx.Response(
            200, content="".join(lines).encode(), headers={"content-type": "text/event-stream"}
        )

    server = Server(sse)
    out = make_llm(server).chat([{"role": "user", "content": "x"}], stream=True)
    assert "".join(out) == "Hello"
    assert server.body()["stream"] is True
    assert server.body()["chat_template_kwargs"] == {"enable_thinking": False}


def test_stream_with_tools_rejected():
    with pytest.raises(ValueError):
        make_llm(Server(completion())).chat([], tools=[{"type": "function"}], stream=True)


def test_structured_payload():
    server = Server(completion('{"reasoning": "r", "answer": "Atlas"}'))
    out = make_llm(server).structured([{"role": "user", "content": "q"}], Answer)
    assert out == Answer(reasoning="r", answer="Atlas")
    body = server.body()
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "Answer"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == Answer.model_json_schema()
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.8
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_structured_retries_once_with_error():
    server = Server(
        completion('{"reasoning": "r"}'), completion('{"reasoning": "r", "answer": "a"}')
    )
    out = make_llm(server).structured([{"role": "user", "content": "q"}], Answer)
    assert out.answer == "a"
    assert len(server.requests) == 2
    retry = server.body(1)["messages"]
    assert retry[-2] == {"role": "assistant", "content": '{"reasoning": "r"}'}
    assert retry[-1]["role"] == "user" and "answer" in retry[-1]["content"]


def test_structured_gives_up_after_retry():
    server = Server(completion("not json"))
    with pytest.raises(StructuredOutputError) as err:
        make_llm(server).structured([{"role": "user", "content": "q"}], Answer)
    assert err.value.raw == "not json"
    assert len(server.requests) == 2


def embeddings_reply(request: httpx.Request) -> httpx.Response:
    texts = json.loads(request.content)["input"]
    data = [
        {"object": "embedding", "index": i, "embedding": [float(len(t)), 1.0, 0.0, 0.0]}
        for i, t in enumerate(texts)
    ]
    data.reverse()  # the client must order by index
    return httpx.Response(200, json={"object": "list", "data": data, "model": "e"})


def test_embed_batches_normalises_and_orders():
    server = Server(embeddings_reply)
    llm = make_llm(server)
    texts = [f"text {'x' * i}" for i in range(70)]
    out = llm.embed(texts)
    assert out.shape == (70, 4) and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert [len(json.loads(r.content)["input"]) for r in server.requests] == [32, 32, 6]
    body = server.body(0)
    assert body["model"] == "Qwen3-Embedding-0.6B-8bit"
    assert body["encoding_format"] == "float"
    assert body["input"][0] == "text "  # documents get no prefix
    expected = np.array([5.0, 1.0, 0, 0]) / np.linalg.norm([5.0, 1.0])
    assert np.allclose(out[0], expected)
    assert llm.embed_dim == 4


def test_embed_query_prefix():
    server = Server(embeddings_reply)
    make_llm(server).embed(["where does Alex work"], query=True)
    assert server.body()["input"] == [QUERY_INSTRUCT + "where does Alex work"]


def test_embed_empty():
    server = Server(embeddings_reply)
    assert make_llm(server).embed([]).shape[0] == 0
    assert server.requests == []


def test_rerank():
    reply = {
        "results": [
            {"index": 1, "relevance_score": 0.9, "document": {"text": "b"}},
            {"index": 0, "relevance_score": 0.1, "document": {"text": "a"}},
        ],
        "usage": {"total_tokens": 10},
    }
    server = Server(reply)
    assert make_llm(server).rerank("q", ["a", "b"], top_n=2) == [(1, 0.9), (0, 0.1)]
    assert server.requests[0].url.path == "/v1/rerank"
    assert server.body() == {
        "model": "Qwen3-Reranker-0.6B-mxfp8",
        "query": "q",
        "documents": ["a", "b"],
        "top_n": 2,
    }


def test_rerank_http_error():
    server = Server(httpx.Response(507, json={"error": "memory guard"}))
    with pytest.raises(httpx.HTTPStatusError):
        make_llm(server).rerank("q", ["a"])


def test_health():
    server = Server({"status": "healthy"})
    assert make_llm(server).health() is True
    assert server.requests[0].url.path == "/health"

    def down(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert make_llm(down).health() is False
    assert make_llm(Server(httpx.Response(503))).health() is False


def test_recorder_saves_exchanges(tmp_path):
    server = Server(completion('{"reasoning": "r", "answer": "a"}'))
    llm = make_llm(server, recorder=Recorder(tmp_path, prefix="demo-"))
    llm.structured([{"role": "user", "content": "q"}], Answer)
    [path] = tmp_path.glob("demo-structured-*.json")
    exchange = load_exchange(path)
    assert exchange.kind == "structured"
    assert exchange.request["response_format"]["json_schema"]["name"] == "Answer"
    assert exchange.content == '{"reasoning": "r", "answer": "a"}'
