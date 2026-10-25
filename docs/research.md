# Research notes (October 2026)

Condensed findings that shaped the design. Numbers are as reported by their sources; most memory
benchmark figures are vendor-reported and not comparable across setups.

## 1. Karpathy's LLM Wiki

- X post "LLM Knowledge Bases" (2 Apr 2026) and the "LLM Wiki" idea file
  (<https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>).
- Three layers: **raw sources** (immutable) → **wiki** (LLM-owned markdown: summaries, entity and
  concept pages) → **schema** (CLAUDE.md/AGENTS.md conventions, co-evolved with the human).
- Operations: **ingest** (one source touches 10–15 pages), **query** (read `index.md` first, answer
  with citations, file good answers back), **lint** (contradictions, stale claims, orphans, gaps).
- `index.md` (catalog) + `log.md` (append-only timeline) replace RAG at ~100 sources / hundreds of
  pages. "LLMs don't get bored" — they do the bookkeeping that kills personal wikis.
- Related: LLMs have "anterograde amnesia"; "where's the hippocampus?"; sleep-like distillation is
  the missing phase (Dwarkesh interview, Oct 2025).
- Failure modes at scale (MPT Solutions, Aug 2026): stale body text, silent entity duplicates,
  orphans around ~300 pages; no temporal or provenance model.

## 2. Agent-memory landscape

| system | storage | notes |
|---|---|---|
| Graphiti (Zep) | Neo4j / FalkorDB(Lite) | best temporal model (bi-temporal edges, invalidation); small local models produce schema errors (#912), mis-retire facts when contradiction search isn't scoped (#1728: 41% invalidated, 3/4 wrong); reasoning-field-first raised contradiction accuracy 7/15 → 14/15 (#1666) |
| Mem0 v2 | vector store | removed graph memory (Apr 2026); add-only extraction, weak supersession |
| Letta / letta-code | git-tracked markdown "MemFS" | core memory always in context; "dreaming" consolidation; file+grep agent scored 74% on LoCoMo vs Mem0-graph 68.5% |
| Cognee | Ladybug + LanceDB + SQLite | fully embedded; GLiNER extraction without an LLM |
| Hindsight | embedded Postgres | experiences → observations → mental models; retain leaderboard: Qwen3.6-35B-A3B 49% vs best API 50% extraction accuracy |
| Basic Memory | markdown + SQLite | MCP server, Obsidian-compatible; AGPL, pre-1.0 |
| LightRAG / GraphRAG / HippoRAG 2 | various | document-corpus oriented; LightRAG names Qwen3-30B-A3B as minimum for local extraction |

Storage: Kuzu archived (Oct 2025) → LadybugDB fork; sqlite-vec still alpha but widely used.

## 3. Memory design principles

- **Generative Agents** (arXiv 2304.03442): retrieval = recency + importance + relevance; recency
  decays from *last access*; reflection when summed importance crosses a threshold.
- **MemGPT** (2310.08560): tiered memory, core blocks always in context.
- **Zep/Graphiti** (2501.13956): valid-time vs record-time; invalidate, don't delete.
- **LongMemEval** (2410.10813): tests extraction, multi-session, temporal reasoning, knowledge
  updates, abstention; fact-granular storage and time-aware query expansion help.
- **StateMemBench** (2608.19652): systems often answer with *superseded* state; explicit
  supersession tracking gives large gains.
- **Mastra observational memory**: dated, prioritised observation log maintained by background
  observer/reflector agents.
- Practitioner advice: prefer explicit supersession with history over decay curves for facts;
  keep evidence chains instead of numeric confidence; decide fact vs opinion at ingest.

## 4. Local inference on a base M4 (32GB, ~120 GB/s)

- Dense ~27B 4-bit ≈ 7 tok/s → only MoE models are usable interactively.
- Qwen3.6-35B-A3B 4-bit MLX: ~45–48 tok/s, 19–20 GB peak at ≤4k context; 22 GB at 32k (oMLX
  community benchmarks). Gemma 4 26B-A4B: 31–38 tok/s, 13–15 GB.
- Prefill ~350–425 tok/s → a 4k-token prompt costs ~10 s uncached. Stable prompt prefixes and
  small retrieved context matter more than decode speed.
- macOS GPU wired limit defaults to roughly 21–24 GB on 32 GB machines; raise with
  `sudo sysctl iogpu.wired_limit_mb=24576` (resets on reboot).
- Runtimes: oMLX (chat + embed + rerank, JSON schema, SSD prefix cache), LM Studio, Ollama (MLX
  backend for safetensors), llama.cpp (most mature grammars), mlx-lm server (tool-call parser gaps
  for Qwen 3.5/3.6, mlx-lm #1293).

## 5. Routing / escalation

- Learned routers (RouteLLM 2406.18665) need preference data; cascades (FrugalGPT 2305.05176,
  AutoMix 2310.12963) hinge on the verifier.
- Verbalised confidence is unreliable in small models (2412.14737, 2502.11028).
- Production routers use rules over domain/keyword/embedding signals (vLLM Semantic Router,
  2603.04444).
- → Rules + retrieval coverage + categorical answer labels + validation; log decisions to train a
  router later.

## 6. Claude integration

- Enterprise seats include Claude Code; `claude -p` works headless under the subscription login.
  The Agent SDK is documented as API-key only.
- Local stdio MCP servers work in Claude Code (`.mcp.json`) and Claude Desktop, not in claude.ai web
  (custom connectors must be public HTTPS).
- Hooks: `SessionEnd` gets `transcript_path` (copy it, do heavy work later). Transcripts live in
  `~/.claude/projects/<dir>/<session>.jsonl`, retained 30 days by default.
- Enterprise Claude Code sessions are retained server-side (6 years by default) and retrievable by
  admins with Compliance API access — relevant when personal data is escalated.
- Reused: `kepano/obsidian-skills` (MIT) → `.claude/skills/obsidian-markdown`.

## Sources

Karpathy: <https://x.com/karpathy/status/2039805659525644595>,
<https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>,
<https://www.dwarkesh.com/p/andrej-karpathy>,
<https://www.mpt.solutions/karpathys-llm-wiki-works-heres-what-broke-when-i-ran-it-at-real-scale/> ·
Memory: <https://github.com/getzep/graphiti>, <https://github.com/mem0ai/mem0/releases/tag/v2.0.0>,
<https://www.letta.com/blog/benchmarking-ai-agent-memory>, <https://www.letta.com/blog/context-repositories>,
<https://github.com/vectorize-io/hindsight>, <https://github.com/topoteretes/cognee>,
<https://github.com/basicmachines-co/basic-memory>, <https://mastra.ai/research/observational-memory>,
<https://github.com/LadybugDB/ladybug>, <https://github.com/tobi/qmd> ·
Inference: <https://github.com/jundot/omlx>, <https://omlx.ai/benchmarks/pcfbnl7n>,
<https://unsloth.ai/docs/models/tutorials/qwen3.6.md>, <https://github.com/ml-explore/mlx-lm/issues/1293> ·
Routing: <https://arxiv.org/abs/2406.18665>, <https://github.com/vllm-project/semantic-router> ·
Claude: <https://code.claude.com/docs/en/hooks.md>, <https://code.claude.com/docs/en/mcp.md>,
<https://code.claude.com/docs/en/skills.md>, <https://platform.claude.com/docs/en/manage-claude/compliance-api.md>,
<https://github.com/kepano/obsidian-skills>
