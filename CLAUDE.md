# local_brain — developer guide for Claude

A local "second brain": a small local LLM (oMLX on an M4 32GB Mac) that chats with its owner,
remembers conversations like a human (episodes → consolidated knowledge), keeps a markdown
knowledge graph of their life, interviews them to fill gaps, and hands hard questions to Claude.

This file is for **building** the brain. Claude's *runtime* role (answering escalated questions
inside the vault) is defined in `vault_template/CLAUDE.md`, not here.

## Hard rules

1. **This repo is PUBLIC. No personal data, ever.** No real names, facts, transcripts, documents
   or vault content in code, tests, docs, commit messages, or PR text. Examples and fixtures use
   the fictional persona **Alex Rivera** (data engineer at the fictional **Northwind Labs**, friend
   **Sam Okafor**, project **Atlas Migration**). If you need real data to debug, read it from the
   vault at runtime and never copy it into the repo.
2. **Never read the vault unless the task needs it**, and never paste vault content into chat
   output that will end up in the repo. The vault is at `$BRAIN_VAULT` (default `~/brain-vault`).
3. **The pre-commit hook must stay on** (`git config core.hooksPath .githooks`). Never use
   `--no-verify`. Run `scripts/privacy_check.sh --all` before every push.
4. **Markdown is the source of truth; SQLite is derived.** Any feature must survive
   `brain reindex` from the vault alone.
5. **One writer.** All vault mutations go through `brain.vault` operations (which also append to
   `log.md` and git-commit the vault). The LLM proposes operations as JSON; code applies them.
   The LLM never free-writes `## Facts` sections.
6. **The runtime is swappable.** Only `brain.llm` talks to the model server, through the
   OpenAI-compatible API. No runtime-specific calls elsewhere.
7. **Privacy filter in every path to Claude.** MCP tools and handoff packets must exclude
   `privacy: private` content unless the user explicitly confirmed it.

## Key docs (read before changing the related area)

- `docs/architecture.md` — components, flows (chat turn, interview, memory pipeline, Claude), phases
- `docs/vault-schema.md` — **canonical** vault format: layout, page frontmatter, fact lines, episodes,
  predicates, reviews, privacy. Change the schema there first, bump its version.
- `docs/retrieval.md` — human-memory-inspired retrieval model (activation, working memory, surfacing)
- `docs/research.md` — research findings and sources behind the design
- `docs/decisions.md` — decision log; append an entry for every non-obvious choice
- `config/life-map.yaml` — regions and slots driving interview mode and coverage

## Stack

- Python 3.12, `uv` for env and deps, `src/` layout, package `brain`, CLI entry point `brain`.
- Pydantic v2 models for every LLM input/output schema; `openai` client against
  `http://127.0.0.1:<port>/v1`; SQLite with FTS5 + `sqlite-vec`; `mcp` Python SDK for the server;
  `typer` + `rich` for the CLI; `pytest`, `ruff`.
- No LangChain / LangGraph / agent frameworks: explicit control flow.
- Local model: Qwen3.6-35B-A3B (MLX 4-bit), **non-thinking** for chat, extraction and tool calls;
  embeddings Qwen3-Embedding-0.6B; reranker Qwen3-Reranker-0.6B. See the `local-llm` skill for
  endpoint, model ids and request quirks.

## Commands

```bash
uv sync                          # install
uv run pytest                    # tests (must not need the model server unless marked `llm`)
uv run pytest -m llm             # tests that call the local model
uv run ruff check . && uv run ruff format .
scripts/privacy_check.sh --all   # before every push
```

## Conventions

- Small modules, typed, docstrings only where intent isn't obvious. Match surrounding code.
- Every LLM call: pinned prompt template in `src/brain/prompts/`, Pydantic output schema, enforced
  JSON schema decoding, a `reasoning` field placed **before** answer fields, and a test with a
  recorded fixture response so tests run without the model.
- Keep prompts short and stable (the prefix is cached; prefill on a base M4 is ~400 tok/s).
  Retrieved memory in a prompt stays under ~1.5k tokens.
- Dates: ISO 8601; valid-time precision may be `YYYY` or `YYYY-MM`.
- Tests build vaults from `tests/fixtures/` (fictional persona) into a tmp dir.
- Commits: conventional style (`feat(memory): ...`), one logical change each; push at the end of
  each phase or when asked.

## Working style for this project

- The lead session plans and integrates; independent modules are delegated to subagents with a
  precise brief (files they own, interfaces, tests to pass). Avoid two agents editing one file.
- Before building a phase, re-read its section in `docs/architecture.md` and update the phase
  table when done.
- When unsure about a design choice that affects the user's data or privacy, ask.

## Status

Phase 0 (scaffolding, docs, skills, runtime). See the phase table in `docs/architecture.md`.
