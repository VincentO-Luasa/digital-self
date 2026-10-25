# Architecture

A personal "second brain" that runs on a 32GB Apple-silicon laptop. A local model handles small
talk, recall and advice that can be inferred from the knowledge graph; anything harder is clearly
handed to **the big model**, a frontier model the owner already has access to (System 2 to the
local model's System 1). The big model is pluggable: the author uses Claude via Claude Code, but
OpenAI, Kimi K3 or any OpenAI-compatible endpoint work too (Flow 4). Every conversation, local or
big-model, flows back into memory.

```
            ┌───────────── you ─────────────┐
            │                               │
   brain chat / interview (local)    the big model (pluggable provider)
            │                               │  MCP: privacy-filtered read + propose
   ┌────────▼─────────┐                     │  transcript → raw/external/<provider>/
   │ Router            │──"→ Ask <model>"───▶│
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
                     memory pipeline: segment → extract → review → consolidate ("sleep" when idle)
```

## Components (planned `src/brain/` packages)

| package | responsibility |
|---|---|
| `config` | settings (vault path, endpoint, model ids, thresholds) from `~/.config/digital-self/config.toml` + env |
| `llm` | thin OpenAI-compatible client: chat, JSON-schema calls, tool calls, embeddings, rerank. The only module that knows about the runtime. |
| `vault` | parse/render pages, fact lines, episodes, journal; **the single writer** (all mutations go through operations applied here, then a git commit) |
| `index` | SQLite: FTS5, sqlite-vec embeddings, `entities`, `facts`, `links`, `access_log`; `reindex` rebuilds from the vault |
| `retrieval` | human-memory-inspired retrieval (`docs/retrieval.md`): schema card (L0), working memory (L1), cues → spreading activation (PPR) + base-level activation + similarity + importance (L3), surfacing (L4); no LLM call in the hot path |
| `router` | pre-route rules, feeling-of-knowing gate, effort tiers (local fast → local thinking → big model), conflict monitor, answer validation |
| `chat` | the conversation loop, prompt assembly (`core.md` + retrieved facts + recent turns), commands |
| `interview` | coverage scoring over `life-map.yaml`, region selection, question planning |
| `memory` | topic segmentation, extraction to operations, entity resolution, review queue, consolidation, lint, decay |
| `handoff` | build/preview/send handoff packets through the configured escalation provider (`claude-code`, `clipboard`, `openai-compatible`; Flow 4) |
| `mcp_server` | stdio MCP server for any MCP-capable client, enforcing privacy |
| `cli` | `brain init-vault | chat | interview | ingest | review | consolidate | lint | reindex | ask | capture | status` |

No agent framework: the control flow is explicit Python with Pydantic schemas.

## Flow 1 — a chat turn

1. **Pre-route** (deterministic, ~20 ms): explicit `/big` or `/local`; kNN over labelled example
   queries + keyword features → category. Categories that always escalate: coding, maths /
   multi-step reasoning, current events, long-document analysis, high-stakes
   (medical/legal/financial), long-form writing.
2. **Retrieve** (`docs/retrieval.md`): extract cues, update working memory, score candidates by
   activation, pack ≤1k tokens of gists and facts, maybe surface one related memory; record access.
3. **Feeling-of-knowing gate** (`docs/brain-map.md` #3): two signals computed in code,
   *familiarity* (are the cue entities known?) and *recollection* (did a fact for what is asked
   come back?). Familiar but not recalled → search deeper, then **ask the owner** and add the
   gap to `questions.md`. Unfamiliar personal cue → "who's X?". Only missing *world* knowledge
   goes to the big model. Personal gaps never do, because the big model doesn't know your life
   either.
3b. **Effort tier**: T1 local fast; T2 local *thinking mode* for private multi-step reasoning;
   T3 the big model. A conflict monitor in code (contradictory current facts, a cited superseded
   fact) raises the tier.
4. **Answer** with an enforced JSON schema:
   `{answer, used_fact_ids[], answerable: yes|partial|no, needs: none|reasoning|world_knowledge|fresh_info|expertise}`.
   Categorical labels, never self-rated numeric confidence (unreliable in small models).
5. **Validate**: cited ids must exist; advice with no cited facts, `answerable != yes` or
   `needs != none` → escalate.
6. **Escalate**: banner `→ Ask <display_name>: <reason>` (`→ Ask Claude` in the author's setup);
   `brain ask` builds the packet (question, facts it would share — private ones flagged — local
   draft, reason), shows it, and on confirmation hands it to the configured provider (Flow 4).
7. **Prefetch** (`docs/brain-map.md` #1): while the owner reads, predict the next retrieval and
   warm the prompt cache with a prefill-only request; on submit only a small "late" memory block
   is added. History is trimmed in chunks of 4 turns so the cached prefix survives.
8. **Log** the decision with its signals; one keystroke marks "should have escalated" /
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
| segment | session end | local | episodes with regions, entities, gist, importance *labels* (emotion, goal link, future-dated, "remember this") → composite importance computed in code |
| extract | session end, per episode | local, JSON-schema, reasoning-field-first, **predict-calibrate**: sees current facts of the resolved entities and emits only `new` / `contradicts` / `confirms` / `noop` | journal observations + candidate operations; congruent updates auto-apply, surprises go to review |
| resolve | with extract | embeddings + aliases → local LLM decides | ops target existing entities, or create new ones |
| apply / queue | with extract | code | safe ops applied + git commit; others → `reviews/pending/` |
| consolidate ("sleep") | **opportunistic**: idle ≥10 min on mains power, or when consolidation pressure is high; queue ordered by need × gain, checkpointed | local | page summaries, reflections (→ review), briefing cards for likely topics, slow `core.md` updates (≥3 episodes over ≥2 weeks, diff confirmed), `index.md`, edge downscaling, decay/archive, git commit |
| lint | with consolidation | code + local | format errors, duplicates, orphans, drift, gaps → `questions.md` |

The local model does all writes. The big model only *proposes* (through MCP or by its transcript
being ingested), which keeps personal data local and avoids automated use of a seat that may be
administered by an employer.

## Flow 4 — Big-model integration

The big model is whatever frontier model the owner already uses. The local brain only needs a way
to hand it a confirmed packet and a way to get the conversation back. That is the escalation
provider, set in `~/.config/digital-self/config.toml`:

```toml
[escalation]
provider = "claude-code"     # claude-code | clipboard | openai-compatible
display_name = "Claude"       # shown in the banner: "→ Ask Claude"
# openai-compatible only:
base_url = "https://api.openai.com/v1"   # or Moonshot (Kimi), OpenRouter, a self-hosted server...
model = "..."
api_key_env = "OPENAI_API_KEY"
```

The example is the author's setup. The code defaults are `provider = "clipboard"` and
`display_name = "the big model"`, so nothing is sent anywhere until the owner configures it.

| provider | what `brain ask` does after you confirm | how the conversation comes back |
|---|---|---|
| `claude-code` | opens Claude Code in the vault workspace with the packet; richest integration (MCP tools, skills, permission rules, SessionEnd hook) | the SessionEnd hook copies the transcript to `raw/external/claude-code/` |
| `clipboard` | copies the packet for any chat UI (ChatGPT, Kimi, Claude.ai, …) | paste or export the reply into `brain capture`; it is filed under `raw/external/clipboard/` |
| `openai-compatible` | sends the packet to any OpenAI-compatible API (OpenAI, Moonshot/Kimi, OpenRouter, a self-hosted server) | the reply is stored as a transcript in `raw/external/openai-compatible/` |

- **Vault workspace.** `brain init-vault` installs `vault_template/` into the vault, so any agent
  opened there gets its instructions: `AGENTS.md` is the canonical, model-agnostic runtime role
  ("you are the brain's big model"); `CLAUDE.md` is a thin Claude Code wrapper that imports it;
  `.claude/` holds Claude Code's skills, settings and hook; `.mcp.json` registers the MCP server.
- **MCP server** (privacy-filtered, standard protocol, so any MCP-capable client works: Claude Code
  or Desktop, other agents and IDEs that support MCP): `brain_search`, `brain_get_page`,
  `brain_core_profile`, `brain_recent_episodes`, `brain_capture` (append to inbox),
  `brain_propose` (create a review).
- **Transcripts.** Every big-model conversation lands in `raw/external/<provider>/YYYY/MM/`. The
  next pipeline run ingests it like any other conversation; its episodes get `channel: external`
  and `provider: <provider>`.
- **Web chats** can't reach a local MCP server: with the `clipboard` provider, or for any chat you
  want to keep, `brain capture` imports pasted or exported chats.

## Privacy model

- Repo is public: code, docs, generic config and fictional fixtures only. Enforced by
  `.gitignore`, the pre-commit hook (`scripts/privacy_check.sh`, which also reads the vault's
  generated denylist), and the rule in `CLAUDE.md`.
- Vault is local: its git repo has no remote by default.
- Whatever goes to a hosted big model is retained by its provider, and the seat may be
  organisation-administered (in the author's case a Claude Enterprise seat, whose transcripts are
  reachable through the Compliance API). Hence, for every provider: private pages are never served
  over MCP, and handoff packets are shown for confirmation before anything is sent.

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

| phase | deliverable | status |
|---|---|---|
| 0 | repo, CLAUDE.md, docs, skills, privacy guardrails, runtime installed + benchmarked | done |
| 1 | vault init + schema, parser/renderer, SQLite index, `brain ingest` (seed docs), read-only `brain chat` | done |
| 2 | episodic capture, topic segmentation, extraction, entity resolution, review queue, **interview mode** | next |
| 3 | router + escalation, handoff packets, escalation providers, MCP server, vault workspace + SessionEnd hook | |
| 4 | opportunistic "sleep" scheduler, consolidation, reflections, briefing cards, person models, `core.md`, lint, decay | |
| 5 | personal eval harness (updates, temporal, abstention, multi-hop), tuning; web UI | |
| 6 | "graph of my life" visualisation (interactive, time slider, region filters) | |

## Open items (from Phase 0 review)

- `brain init-vault` substitutes an absolute `uv` path into `.mcp.json` and the hook, because
  Claude Code launched from the desktop app or an IDE may not have `uv` on PATH.
- Deduplicate the big model's `brain_propose` operations against later extraction of the same
  session's transcript (`raw/external/<provider>/`). Tag reviews with the session id and skip facts
  already proposed in that session.
- The handoff packet format in `vault_template/AGENTS.md` is the target for the `handoff` module.
- Documents ingested by the big model (`brain-ingest-doc`) cite its transcript as their source.
  Phase 3 decides whether a copy also goes to `raw/docs/`.
- The predicate list is mirrored in `vault_template/AGENTS.md`. Keep it in sync when the schema
  version changes (a `brain lint` check).

## Open items (from the Phase 1 writer review)

- Enforce `evidence::` on inferred facts (the schema requires it; `AGENTS.md` asks for at least two).
- After `merge_entities`, reconcile clashing facts from the two pages and repoint pending reviews
  that still name the merged id.
- Vault commits use `git add -A`, so uncommitted hand edits ride along in the brain's commit.
  Commit only the paths the writer touched, and commit hand edits separately (`brain lint`).
- `models.check_partial_date` should reject impossible calendar dates itself (the writer does it
  today); `gitops` needs a public helper to unstage paths (rollback uses a private one).
- An approved fact's `rec::` is the approval date, while an observation keeps the change set's
  date. Decide which "learned at" the brain means.

## Open items (from the first real ingest)

Quality issues seen on a real personal homepage (about 3.8k tokens, 10 chunks, 100 facts, ~6 min):

- `core.md` lists every current fact about the owner. It needs per-section caps, dedup of
  near-identical lines and a real salience ranking (Phase 4 consolidation; a stopgap in Phase 2).
- HTML loaders read animated counters at their initial value ("0 years of…"). Prefer `data-*`
  target values or drop numeric claims that read as 0.
- The model creates pages for tools and products (an AI assistant, a code host, a database
  record id) and for teams. Normalise those to literal values, as the prompt already asks.
- `works_at` is misused for responsibilities ("Works on models that…"); they get merged into the
  employer fact and lost. Map verb-led descriptions to `responsibility`.
- Fact text sometimes uses pronouns ("his") or "their toolkit"; render from a template or
  post-edit.
- Dated past residences without dates contradict the current city and go to review; the
  extractor should set `end` when the text says "previously" or uses the past tense.
