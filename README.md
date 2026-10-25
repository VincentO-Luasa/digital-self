<div align="center">

# 🧠 Digital Self

### A brain of your own: it lives on your laptop and remembers the way a human brain does.

You talk to it. It remembers you the way a person would.<br/>
When a question is beyond it, it says so and asks a bigger brain.

**📚 Karpathy's LLM Wiki &nbsp;➕&nbsp; 🧠 how humans remember &nbsp;=&nbsp; 🪞 a local digital self**

`runs on a MacBook M4 · 32GB` · `memory stays on your machine` · `plain markdown you can read` · `any big model for hard questions`

</div>

---

<p align="center">
  <img src="docs/img/overview.svg" alt="You chat with a local model on your laptop (System 1). It recalls episodes and wiki knowledge; a gate decides whether it knows the answer, asks you about gaps in your life, and sends hard questions to a big model (System 2) once you approve. Sleep consolidates episodes into the wiki." width="100%">
</p>

## The idea

A brain you can talk to, one that knows **you**, is a digital self. So it should be built like one.

1. **Knowledge compounds.** Following Karpathy's [LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f),
   conversations and documents are compiled into an interlinked markdown wiki that the model keeps
   up to date. Nothing is rediscovered from scratch on every question.
2. **Memory works like a person's.** When you talk with someone you know, you don't search a
   database. You already have an impression of them, a few things that marked you, and the
   conversation itself brings memories back. The brain copies that.
3. **It knows its limits.** Like Kahneman's two systems: a small local model is the fast, intuitive
   **System 1** (small talk, recall, everyday advice), a big model is the slow, deliberate
   **System 2**. When the brain lacks knowledge *of your life*, it asks you. When a problem is hard,
   it tells you to ask the big model, and you approve exactly what gets shared.

## How a human remembers → how the brain does it

| a person… | the brain… |
|---|---|
| has an impression of you, always present | keeps a **schema card** (temperament, current situation, marking facts) in every prompt |
| holds only a few things in mind at once | has a **4-slot working memory** that follows the conversation |
| is reminded of things by what you just said | uses **spreading activation**: names, topics and dates in your message light up related memories in the graph |
| remembers what they use, and forgets the rest | uses **ACT-R activation**: recall strengthens a memory, unused ones fade, important ones never do |
| turns the day into knowledge while asleep | runs **"sleep"** when the laptop is idle and plugged in: episodes → facts → impressions |
| knows when they know something | has a **feeling-of-knowing gate**: answer, ask you, or escalate |
| gets to know you by talking | has an **interview mode**: short, topic-focused conversations that fill the gaps it notices |

The full mapping, with the neuroscience and how strong the evidence is for each, is in
[docs/brain-map.md](docs/brain-map.md). The retrieval model is in [docs/retrieval.md](docs/retrieval.md).

## What a conversation looks like

```text
you    ▸ I'm dreading tomorrow's Atlas review.
brain  ▸ The go-live date again? Last month you said the backfill was the real risk,
         and Sam offered to help with it. Did that happen?        ← recall, "this reminds me of…"
you    ▸ Yes, it's done. Should I push the date anyway? Weigh the risks for me.
brain  ▸ → Ask the big model: needs multi-step reasoning.
         Packet ready: your question, 3 facts, my draft (0 private items). Send? [y/N]
```

*(Fictional persona. This repository contains no personal data.)*

## Under the hood

```
 you ─► brain chat / interview ─► router ─► local model  (oMLX · Qwen3.6-35B-A3B · ~45 tok/s)
                                    │              ▲
                                    │              │ ~1k tokens of activated memories
                     "→ ask the     ▼              │
                      big model"  handoff     VAULT  ~/brain-vault  (markdown + local git = truth)
                                  (you confirm)    + SQLite index (FTS · vectors · facts · links)
                                                   ▲
                           memory pipeline: segment → extract → review → sleep
```

- **Vault**: Obsidian-compatible markdown. Raw conversations are the episodic memory; wiki pages
  about people, projects and you are the semantic memory. Each fact records when it was true,
  when it was learned and where it came from. Facts are superseded, never deleted.
  See [docs/vault-schema.md](docs/vault-schema.md).
- **Local model**: a mixture-of-experts model with about 3B active parameters, fast enough for
  chat on a base M4. Retrieval is pure math (about 50 ms), so the model's time goes into answering.
- **Big model, pluggable**: Claude Code, any chat UI through the clipboard, or any
  OpenAI-compatible API (OpenAI, Kimi, …). The author uses Claude. The vault exposes a
  privacy-filtered MCP server, and big-model conversations flow back into memory.
- **Design and research**: [architecture](docs/architecture.md) · [research notes](docs/research.md)
  · [decision log](docs/decisions.md).

## Privacy

- Memory lives in `~/brain-vault`, a local git repo with no remote. This public repo holds only
  code, docs and a fictional test persona.
- Private regions (family, friends, health, traits) are never served to the big model. Every
  handoff is shown to you before it is sent.
- A pre-commit hook blocks vault paths, names from the vault's private pages, and secrets
  (gitleaks). After cloning, run `git config core.hooksPath .githooks`.

## Status

| phase | | |
|---|---|---|
| 0 | design, research, privacy guardrails, local runtime | ✅ |
| 1 | vault, index, ingest, read-only chat | 🚧 |
| 2 | episodes, extraction, review queue, **interview mode** | |
| 3 | router, effort tiers, handoff, MCP server | |
| 4 | "sleep": consolidation, briefing cards, person models | |
| 5 | personal memory eval, tuning, web UI | |
| 6 | 🕸️ "graph of my life" visualisation | |

## Credits

Andrej Karpathy's [LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) ·
Generative Agents · MemGPT/Letta · Zep/Graphiti · HippoRAG · EM-LLM · Sleep-time Compute, and the
cognitive science cited in [docs/brain-map.md](docs/brain-map.md) ·
[kepano/obsidian-skills](https://github.com/kepano/obsidian-skills) (MIT). Built with Claude Code.

MIT License
