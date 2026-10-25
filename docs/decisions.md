# Decision log

Short entries, newest last. Format: date — decision — why — alternatives considered.

## 2026-10-25 — Markdown vault is the source of truth; SQLite is a derived index
Why: at personal scale (thousands of notes) file + grep/FTS agents match graph-DB memory on
LoCoMo (Letta); markdown is readable, diffable and editable by the owner, which matters for a
self-model holding "flaws"; Karpathy's LLM-Wiki pattern; any index can be rebuilt after a model or
schema change. Alternatives: Graphiti on FalkorDB, Cognee, Mem0 (dropped graph memory in v2),
Hindsight. Kept as later options as *derived* indexes if multi-hop/temporal evals fail.

## 2026-10-25 — Borrow graph-memory semantics into markdown
Bi-temporal fact lines (`from/to` + `rec`), supersede-never-delete, mandatory `src::` provenance,
`aliases:` for entity resolution, stated/observed/inferred split. Why: these are what the memory
literature shows matters (Zep/Graphiti, StateMemBench, LongMemEval), independent of storage.

## 2026-10-25 — Local model performs all memory writes; the big model only proposes
Why: the big model's seat may be organisation-administered (the author's is a Claude Enterprise
seat, whose transcripts are retained and reachable via the Compliance API); automated nightly use
of a seat conflicts with "ordinary use" terms; Qwen3.6-35B-A3B is close to API models on extraction benchmarks (Hindsight retain board).

## 2026-10-25 — Runtime: oMLX, model Qwen3.6-35B-A3B (MLX 4-bit), non-thinking
Why: oMLX serves chat + embeddings + rerank in one process with enforced JSON schema and a
persistent prefix cache; MoE with ~3B active params is the only class fast enough on a base M4
(~45 tok/s vs ~7 tok/s for dense 27B). Fallbacks: LM Studio (runtime), Gemma 4 26B-A4B (model).
The client is OpenAI-compatible only, so swapping is configuration.

## 2026-10-25 — Escalation by rules + retrieval coverage + categorical answer labels
Why: small models' verbalised confidence is unreliable; learned routers need labelled data we do
not have yet. The routing log collects that data for a later learned router.

## 2026-10-25 — Public repo, private vault
Why: the owner wants to share how the brain is built. Code/docs/fictional fixtures only in the
repo; vault outside the repo with a local-only git; pre-commit hook with path rules, a
vault-generated denylist, and gitleaks.

## 2026-10-25 — Interview mode with topic-chunked episodes
Why: populating the graph should be a conversation, not an autobiography. Coverage over a life map
(regions × slots) picks what to ask; each conversation is segmented into topic episodes so
extraction is small, focused and reviewable.

## 2026-10-25 — Human-memory-inspired retrieval (proposed)
Schema card always in context (impression + current situation + marking facts), a 4-slot
working memory, cue-driven spreading activation (Personalized PageRank) combined with ACT-R
base-level activation, similarity and importance, gist-first packing, and gated proactive
surfacing. No LLM call in the hot path. Why: lets a ~3B-active model work from a small, ranked,
pre-digested context. HippoRAG 2, Synapse and EM-LLM report gains from the same mechanisms.
Weights are priors to be fitted on the personal eval. Details in `docs/retrieval.md`.

## 2026-10-25 — Big model is pluggable (Claude in the author's setup)
The local model escalates to "the big model", chosen by an `[escalation]` provider in the config:
`claude-code` (Claude Code in the vault workspace, the author's choice), `clipboard` (any chat UI,
reply brought back with `brain capture`), or `openai-compatible` (OpenAI, Moonshot/Kimi,
OpenRouter, a self-hosted server). Big-model transcripts live in `raw/external/<provider>/` and
their episodes use `channel: external` + `provider:`. The vault's runtime instructions are
`AGENTS.md`; `CLAUDE.md` only imports it and adds Claude-Code-specific notes.
Why: the owner wants the project usable with OpenAI, Kimi K3 and other big models, not only
Claude. The integration points are already cross-vendor conventions: MCP is an open protocol many
clients speak, `AGENTS.md` is read by many coding agents, and skills use the open Agent Skills
`SKILL.md` format. Privacy rules (confirmed packets, no private content over MCP) apply to every
provider, since any hosted provider may retain transcripts. Earlier entries' "Claude" wording was
generalised to "the big model".
Alternatives: Claude-only (simpler, but ties a public project to one vendor); a multi-provider SDK
or agent framework (conflicts with the explicit-control-flow rule; the OpenAI-compatible API
already covers most providers).

## 2026-10-25 — Ingest: flat extraction schema, resolution and policy in code
One structured call per heading-bounded chunk (~1.2k tokens). The model fills a flat schema
(`reasoning`, `owner_names`, `entities[{name,type,id,aliases,summary}]`,
`facts[{subject,predicate,object,text,start,end}]`) with names, not ids; the predicate is an enum
over a document subset of the vocabulary (no traits, flaws, health). Code resolves names to ids
(known aliases → entities named earlier in the document → new slug ids), normalises dates,
merges duplicates (union of their periods), turns `(self, owner|relation, X)` around to
`(X, p, self)`, keeps tools/skills/languages as literals, drops entities no fact uses, derives
new pages' regions (hence privacy) from the facts that link them, and adds the owner's names as
aliases of `self/profile`. One `ChangeSet(origin="ingest")` per document; the writer's policy
decides what is queued. The prompt lists only known entities named in the chunk (cap 40) so it
scales with the vault. Why: a ~3B-active model is reliable at "list the facts", not at producing
a discriminated union of operations with ids; everything it returns is a proposal checked in code.
Alternatives: emitting `Operation` JSON directly (likely more invalid ids and wrong directions;
the first live runs already showed wrong directions); sending the whole alias index (does not scale).

## 2026-10-25 — Documents the owner wrote yield `stated` facts
`brain ingest` treats a document as the owner's own (CV, homepage) and records `kind: stated`;
`--observed` marks documents written by others. Why: a first-person homepage is the owner
speaking, and `stated` facts with no contradiction auto-apply. The schema's "observed (in a
document)" now means documents by third parties.

## 2026-10-25 — Phase 1 core.md and index.md are deterministic
`core.md` (L0 card) is built in code from current self facts valid today plus the entities tied
to the owner by a current fact, ranked by a salience prior (predicate weight, stated > observed >
inferred, newest first on ties) and cut at `core_max_tokens`. No timestamps, so the file is
byte-stable and the chat prefix cache survives until a contributing fact changes. Both files
are regenerated and committed after ingest and review approvals, and include private pages
(they are local; the big model's instructions forbid reading them). `build_core(...,
include_private=False)` exists for the MCP profile tool. Phase 4 consolidation replaces the card
builder (impression, marking facts, open loops).

## 2026-10-25 — Raw capture: exclusive files, turn-by-turn transcripts
`raw/docs/YYYY-MM-DD-<slug>.<ext>` and `raw/conversations/YYYY/MM/conv-YYYYMMDD-HHMM.md` are
created with exclusive opens (a suffix `-2` on collision), never overwritten; re-ingesting an
identical document reuses its copy. Transcript turns are `## <n> · <role> · <timestamp>` blocks
appended and fsynced one by one (lines that look like turn headings are escaped), committed at
the end of the session. Why: raw/ is episodic ground truth; a crash must lose nothing, and
Phase 2 segmentation needs stable turn numbers for episode `turns: [a, b]`.

## 2026-10-25 — Chat prompt: memory in the latest user message, history trimmed in blocks
System prompt + core card form the cached prefix. History holds the plain user text (never the
memory block) and drops the oldest 4 exchanges once it exceeds 8. The per-turn block (today's
date + search hits packed gist-first under `retrieval.budget_tokens`, skipping facts already in
the card) is prepended to the latest user message. Token budgets use a chars/3.5 estimate (no
tokenizer dependency; errs on the safe side).

## 2026-10-25 — core.md is a friend's profile card, not a ranked fact dump
The L0 card is now laid out the way people hold a close friend in mind: an identity line, then
Who they are (values, strengths, flaws, habits; "Not known yet: …" for missing aspects), Now
(active projects, responsibilities, goals, recent important journal lines, facts new in the last
90 days), Life story (one line per lifetime period: job + roles + city, studies + degree), People
who matter (≤6, inner circle first), Interests and skills (comma lists with "+n more") and Open
loops (deadlines within 30 days, undone intentions, top `questions.md` items). Still deterministic
and byte-stable; near-duplicates (token Jaccard ≥ 0.75) merge and pool their sources; each
section has a share of a ~600-token target and spare budget flows to the others; `core_max_tokens`
stays the hard cap. Why: the flat builder gave one line per skill, tool and side project (~60
lines on a real homepage; 64 lines / ~600 tokens on a 68-fact synthetic vault, ~1k tokens at 128
facts), which costs prefill and dilutes a small model's attention; the new card is 20 lines /
~440 tokens on both. Grounded in McAdams (1995) three levels, Conway & Pleydell-Pearce (2000)
lifetime periods and working self, trait summaries stored apart from episodes (Park 1986; Klein,
Loftus & Kihlstrom), and Dunbar's inner layers (docs/retrieval.md, "L0: schema card").
Alternatives: an LLM-written profile now (non-deterministic, breaks byte stability; kept for the
Phase 4 impression paragraph); a fixed per-predicate line cap without sections (still a list).
