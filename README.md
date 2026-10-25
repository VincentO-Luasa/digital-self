# local_brain

A **local AI "second brain"** that runs on a MacBook (Apple M4, 32GB). It knows its owner's job,
projects, CV, relationships, habits, strengths and flaws. It remembers conversations roughly the
way a person does and keeps a markdown knowledge graph of their life. It interviews them to fill
gaps, and when a question is too hard for a small local model it says **"ask Claude"**.

> This repository is public and contains **no personal data**. The brain's memory lives in a
> separate local-only folder (the *vault*). Examples use a fictional persona, Alex Rivera.

## Why

- **Private by default.** A personal model of your life should run on your own machine.
- **Cheap where possible.** Small talk and "what did I say about X?" don't need a frontier model.
  A ~3B-active MoE model on a laptop handles those. Claude is kept for reasoning, coding,
  expertise and long-form work.
- **Memory that compounds.** Following Karpathy's *LLM Wiki* idea, knowledge is compiled into an
  interlinked wiki once and kept current, rather than rediscovered by RAG on every question.

## How it works

```
 you ──► brain chat / brain interview ──► router ──► local model (oMLX, Qwen3.6-35B-A3B)
                                            │               ▲
                                            │ "→ Ask Claude" │ retrieval (activation over the graph)
                                            ▼               │
                         Claude Code in the vault ◄──► VAULT: markdown + git (truth)
                         (MCP, privacy-filtered)           + SQLite index (derived)
                                                            ▲
                         memory pipeline: segment → extract → review → nightly consolidate
```

- **Vault**: Obsidian-compatible markdown. Raw transcripts and documents are the episodic
  memory. Wiki pages about people, projects and you are the semantic memory. Facts carry valid
  time, record time and their source, and are superseded rather than deleted. See
  [docs/vault-schema.md](docs/vault-schema.md).
- **Interview mode**: the brain finds the least-covered regions of your life map and asks about
  them in short, topic-focused conversations. Each conversation is split into topic episodes and
  turned into small, reviewable graph updates.
- **Retrieval inspired by human memory**: an always-present impression of you, a working memory
  that follows the conversation, and associative recall that brings back relevant past episodes.
  See [docs/retrieval.md](docs/retrieval.md).
- **Escalation**: rule-based pre-routing, a retrieval coverage check, and categorical answer
  labels (not self-rated confidence). The handoff packet is shown to you before anything is sent.
- **Claude integration**: the vault doubles as a Claude Code workspace with a privacy-filtered
  MCP server, and a SessionEnd hook feeds Claude conversations back into memory.

Full design: [docs/architecture.md](docs/architecture.md). Research behind it:
[docs/research.md](docs/research.md). Decision log: [docs/decisions.md](docs/decisions.md).

## Privacy guardrails for a public repo

- The vault lives outside the repo (`~/brain-vault`) and its git history stays on the machine.
- `.gitignore` blocks vault-like paths and data files.
- `.githooks/pre-commit` → `scripts/privacy_check.sh` blocks commits that contain vault paths,
  terms from the vault's generated denylist (names and aliases of private pages), or secrets
  (gitleaks).
- `CLAUDE.md` forbids personal data in code, tests, docs and commits. Fixtures are fictional.

After cloning: `git config core.hooksPath .githooks`.

## Status

Under construction. Phase 0: design, docs, skills, runtime setup. Build roadmap is in
[docs/architecture.md#phases](docs/architecture.md#phases).

## Credits

Andrej Karpathy's [LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f);
Generative Agents, MemGPT/Letta, Zep/Graphiti, HippoRAG and the other work cited in
[docs/research.md](docs/research.md); [kepano/obsidian-skills](https://github.com/kepano/obsidian-skills)
(MIT) for the Obsidian markdown skill. Built with Claude Code.

## License

MIT
