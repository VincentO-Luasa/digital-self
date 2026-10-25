---
name: local-llm
description: How to call the brain's local model server (oMLX, OpenAI-compatible on 127.0.0.1:8000) — model ids, non-thinking mode, enforced JSON schema, tool calls, embeddings, rerank, sampling, prompt caching, and known quirks. Use when writing or debugging code in src/brain/llm, any prompt or structured-output call, embedding/rerank code, or when the model server misbehaves.
---

# Local LLM runtime (oMLX)

Only `brain.llm` talks to the server. Everything else goes through it.

## Endpoint and models

- Base URL `http://127.0.0.1:8000/v1`, no API key (send any string to satisfy the client).
- Health check: `curl 127.0.0.1:8000/health`. Admin UI: `http://127.0.0.1:8000/admin`.

| role | model id |
|---|---|
| chat / extraction / tools | `Qwen3.6-35B-A3B-4bit` |
| embeddings (1024-d, L2-normalised) | `Qwen3-Embedding-0.6B-8bit` |
| reranker | `Qwen3-Reranker-0.6B-mxfp8` |

Model ids come from config (`~/.config/digital-self/config.toml`). Never hard-code them outside
`brain.config` defaults.

## Request rules (all verified 2026-10-25)

- **Always disable thinking**: `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.
  Thinking is the model default and wastes tokens and latency.
- **Always send sampling explicitly**. The server defaults are thinking-mode (temp 1.0).
  Chat: `temperature=0.7, top_p=0.8`. Extraction / classification: `temperature=0.2`.
- **Structured output**: `response_format={"type": "json_schema", "json_schema": {"name": ..., "strict": True, "schema": Model.model_json_schema()}}`.
  Decoding is grammar-enforced (xgrammar), and the schema is then *not* shown to the model.
  **The prompt must describe every field and enum meaning itself.** Put a `reasoning` field first.
- **Tool calls**: standard OpenAI `tools` + `tool_choice="auto"`; `finish_reason == "tool_calls"`.
  Assistant tool-call messages omit `content` (not `null`), so handle a missing key.
- **Embeddings**: `client.embeddings.create(model=..., input=[...])`. Prefix *queries* (not
  documents) with `"Instruct: Given a question about the owner's life, retrieve relevant memories\nQuery: "`.
- **Rerank**: `POST /v1/rerank` with `{"model", "query", "documents": [...], "top_n"}` →
  `results[{index, relevance_score, document}]`, sorted. Not part of the OpenAI SDK; use `httpx`.

## Performance (base M4, 32GB)

- Decode ~45 tok/s. Prefill ~500 tok/s cold; a cached prefix gives TTFT ~0.15 s (45× faster).
- **Keep prompt prefixes byte-stable**: system prompt, then L0 card, then history, then per-turn
  memory, then the user turn. Anything that changes early in the prompt invalidates the cache.
- Cold model load ~20 s; the server keeps it resident.
- Memory: ~21 GB with all three models loaded. Don't load extra models.

## Testing

- Unit tests must not need the server. Record real responses as JSON fixtures in
  `tests/fixtures/llm/` (fictional persona only) and replay them with a fake client.
- Tests that hit the server are marked `@pytest.mark.llm` and skipped when `/health` fails.

## Operations

- The service is a user LaunchAgent `ai.omlx.server` (`~/Library/LaunchAgents/ai.omlx.server.plist`):
  restart with `launchctl kickstart -k gui/$(id -u)/ai.omlx.server`.
- Config: `~/.omlx/settings.json`, `~/.omlx/model_settings.json`. Logs: `~/.omlx/logs/`.
- HTTP 507 means the memory guard refused to load. Check memory pressure and close apps
  (the guard is pinned at 24 GB).
- Fallbacks if oMLX breaks: LM Studio (`lms server start`), then change only `base_url`/model ids.
