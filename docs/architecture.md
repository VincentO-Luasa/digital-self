# Architecture

A personal "second brain" that runs on a 32GB Apple-silicon laptop. A local model handles small
talk, recall and advice that can be inferred from the knowledge graph; anything harder is clearly
handed to Claude. Every conversation — local or Claude — flows back into memory.

```
            ┌───────────── you ─────────────┐
            │                               │
   brain chat / interview (local)    Claude Code / Desktop, opened in the vault
            │                               │  MCP: privacy-filtered read + propose
   ┌────────▼─────────┐                     │  SessionEnd hook → raw/claude/
   │ Router            │──"Ask Claude"──────▶│
   │ rules + retrieval │   handoff packet
   │ + answer check    │   (you confirm)
   └────────┬─────────┘
            │ OpenAI-compatible HTTP (127.0.0.1)
   ┌────────▼─────────┐     ┌──────────────────────────────┐
   │ oMLX              │     │ VAULT  ~/brain-vault         │
   │ chat: Qwen3.6-    │◀───▶│ markdown + local git = truth │
   │   35B-A3B 4-bit   │     │ .brain/index.sqlite = derived│
   │ embed + rerank    │     └──────────────────────────────┘
   └──────────────────┘                ▲
                     memory pipeline: segment → extract → review → consolidate (nightly)
```

## Components (planned `src/brain/` packages)

| package | responsibility |
|---|---|
| `config` | settings (vault path, endpoint, model ids, thresholds) from `~/.config/local-brain/config.toml` + env |
| `llm` | thin OpenAI-compatible client: chat, JSON-schema calls, tool calls, embeddings, rerank. The only module that knows about the runtime. |
| `vault` | parse/render pages, fact lines, episodes, journal; **the single writer** (all mutations go through operations applied here, then a git commit) |
| `index` | SQLite: FTS5, sqlite-vec embeddings, `entities`, `facts`, `links`, `access_log`; `reindex` rebuilds from the vault |
| `retrieval` | hybrid search (BM25 + vector + alias match) → rerank → graph expansion (1 hop over links/facts) → scoring by relevance × recency-of-access × importance |
| `router` | pre-route rules, coverage check, answer validation, escalation decision + reason |
| `chat` | the conversation loop, prompt assembly (`core.md` + retrieved facts + recent turns), commands |
| `interview` | coverage scoring over `life-map.yaml`, region selection, question planning |
| `memory` | topic segmentation, extraction to operations, entity resolution, review queue, consolidation, lint, decay |
| `handoff` | build/preview/send handoff packets; `claude -p` or clipboard |
| `mcp_server` | stdio MCP server for Claude, enforcing privacy |
| `cli` | `brain init-vault | chat | interview | ingest | review | consolidate | lint | reindex | ask-claude | status` |

No agent framework: the control flow is explicit Python with Pydantic schemas.

## Flow 1 — a chat turn

1. **Pre-route** (deterministic, ~20 ms): explicit `/claude` or `/local`; kNN over labelled example
   queries + keyword features → category. Categories that always escalate: coding, maths /
   multi-step reasoning, current events, long-document analysis, high-stakes
   (medical/legal/financial), long-form writing.
2. **Retrieve**: hybrid search + rerank; record access (reinforces recency).
3. **Coverage check**: if the question needs personal knowledge and too little relevant memory is
   found, escalate (or offer to learn it: "I don't know that yet — want to tell me?").
4. **Answer** with an enforced JSON schema:
   `{answer, used_fact_ids[], answerable: yes|partial|no, needs: none|reasoning|world_knowledge|fresh_info|expertise}`.
   Categorical labels, never self-rated numeric confidence (unreliable in small models).
5. **Validate**: cited ids must exist; advice with no cited facts, `answerable != yes` or
   `needs != none` → escalate.
6. **Escalate**: banner `→ Ask Claude: <reason>`; `brain ask-claude` builds the packet (question,
   facts it would share — private ones flagged — local draft, reason), shows it, and on confirmation
   copies it or opens Claude Code in the vault workspace.
7. **Log** the decision with its signals; one keystroke marks "should have escalated" /
   "didn't need to". Thresholds are tuned on this log; later a small learned router replaces rules.

## Flow 2 — interview mode ("let the brain ask")

Populating the brain should feel like a conversation, not writing an autobiography.

1. **Coverage** per region of `life-map.yaml` = filled slots (non-stale) / slots, using the facts
   table. Per-entity regions (projects, people) score each known entity.
2. **Pick a region**: `brain interview` chooses `argmax(weight × (1 − coverage))`, boosted by items
   in `questions.md`; `brain interview projects` forces one.
3. **Ask one thing at a time**, starting from what is already known ("Your site says you moved to
   the platform team in March — still true?"), following up on answers, ~10–15 minutes or until the
   region is saturated. The user can say "skip", "private", or "stop".
4. **Close the session**: the transcript is saved, segmented into topic episodes (usually one per
   region touched), and extraction runs per episode → many small, reviewable updates.

Free chat also gets topic-segmented the same way, and may surface at most one gap question per
session (configurable).

## Flow 3 — memory pipeline

| stage | when | model | output |
|---|---|---|---|
| capture | every turn | — | `raw/conversations/...` transcript |
| segment | session end | local | episodes with regions, entities, importance |
| extract | session end, per episode | local, JSON-schema, reasoning-field-first | journal observations + candidate operations |
| resolve | with extract | embeddings + aliases → local LLM decides | ops target existing entities, or create new ones |
| apply / queue | with extract | code | safe ops applied + git commit; others → `reviews/pending/` |
| consolidate ("sleep") | nightly, launchd | local | page summaries, reflections (→ review), `core.md`, `index.md`, decay/archive, git commit |
| lint | nightly | code + local | format errors, duplicates, orphans, drift, gaps → `questions.md` |

The local model does all writes. Claude only *proposes* (through MCP or by its transcript being
ingested), which keeps personal data local and avoids automated use of a work seat.

## Flow 4 — Claude integration

- The vault doubles as a Claude Code workspace: `brain init-vault` installs `vault_template/`
  (CLAUDE.md for the "big brain" role, skills, `.mcp.json`, SessionEnd hook) into it.
- MCP tools (privacy-filtered): `brain_search`, `brain_get_page`, `brain_core_profile`,
  `brain_recent_episodes`, `brain_capture` (append to inbox), `brain_propose` (create a review).
- The SessionEnd hook copies the transcript to `raw/claude/`; the next pipeline run ingests it like
  any other conversation (channel `claude`).
- claude.ai web chats can't reach a local MCP server: `brain capture` imports pasted or exported
  chats into `raw/inbox/`.

## Privacy model

- Repo is public: code, docs, generic config and fictional fixtures only. Enforced by
  `.gitignore`, the pre-commit hook (`scripts/privacy_check.sh`, which also reads the vault's
  generated denylist), and the rule in `CLAUDE.md`.
- Vault is local: its git repo has no remote by default.
- Escalation may go through an organisation-administered Claude Enterprise seat, whose transcripts
  are retained and reachable through the Compliance API. Hence: private pages are never served over
  MCP, and handoff packets are shown for confirmation before anything is sent.

## Runtime budget (M4, 32GB)

| component | memory |
|---|---|
| Qwen3.6-35B-A3B MLX 4-bit + runtime | ~19–20 GB |
| KV cache at 16k context | ~1–1.5 GB |
| Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B | ~2 GB |
| GPU wired limit | raise to 24 GB (`iogpu.wired_limit_mb`) |
| macOS + apps | ~8 GB left — tight; fallback model Gemma 4 26B-A4B (~14 GB) |

Prefill (~400 tok/s on base M4) dominates latency: keep the system prompt stable (prefix-cached)
and retrieved context under ~1.5k tokens.

## Phases

| phase | deliverable |
|---|---|
| 0 | repo, CLAUDE.md, docs, skills, privacy guardrails, runtime installed + benchmarked |
| 1 | vault init + schema, parser/renderer, SQLite index, `brain ingest` (seed docs), read-only `brain chat` |
| 2 | episodic capture, topic segmentation, extraction, entity resolution, review queue, **interview mode** |
| 3 | router + escalation, handoff packets, MCP server, vault workspace + SessionEnd hook |
| 4 | nightly consolidation, reflections, `core.md`, lint, decay, launchd jobs |
| 5 | personal eval harness (updates, temporal, abstention, multi-hop), tuning; web UI |
| 6 | "graph of my life" visualisation (interactive, time slider, region filters) |
