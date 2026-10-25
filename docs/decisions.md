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

## 2026-10-25 — Local model performs all memory writes; Claude only proposes
Why: a Claude Enterprise seat may be organisation-administered (Enterprise transcripts are retained
and reachable via the Compliance API); automated nightly use of a seat conflicts with "ordinary
use" terms; Qwen3.6-35B-A3B is close to API models on extraction benchmarks (Hindsight retain board).

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
