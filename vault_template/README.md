# vault_template

`brain init-vault` copies this folder into the owner's private vault (default `~/brain-vault`).
The vault is then also a **workspace for the big model**. When the local model shows
`→ Ask <name>`, the owner opens their agent in the vault (the author uses Claude Code; any agent
that reads `AGENTS.md` and speaks MCP works), and it acts as the brain's big model. It reads memory
only through the privacy-filtered `brain` MCP server and changes memory only by proposing reviews.

- `AGENTS.md` is the canonical, model-agnostic runtime instructions. Many agents read it.
- `CLAUDE.md` is what Claude Code reads. It imports `AGENTS.md` (`@AGENTS.md`) and adds only
  Claude-Code-specific notes, so the two never drift.
- Skills use the open Agent Skills format (`<name>/SKILL.md` with `name` / `description`
  frontmatter). They sit in `.claude/skills/`, where Claude Code loads them; other agents can be
  pointed at the same files.

This README is not copied into the vault.

## What gets installed

| file | becomes | purpose |
|---|---|---|
| `AGENTS.md` | `<vault>/AGENTS.md` | the big model's runtime role, hard privacy rules, tool list, operation and predicate reference |
| `CLAUDE.md` | `<vault>/CLAUDE.md` | Claude Code only: imports `AGENTS.md`, notes on skills, hook and permissions |
| `.mcp.json` | `<vault>/.mcp.json` | registers the `brain` stdio MCP server (`brain mcp`) for MCP clients that read `.mcp.json` |
| `.claude/settings.json` | `<vault>/.claude/settings.json` | Claude Code: SessionEnd hook (transcript → `raw/external/claude-code/YYYY/MM/`), permission rules |
| `.claude/skills/brain-answer/` | same | answering an escalated question end to end |
| `.claude/skills/brain-review/` | same | helping the owner judge pending reviews (never applies them) |
| `.claude/skills/brain-reflect/` | same | on-request reflection that ends in inferred-fact proposals |
| `.claude/skills/brain-ingest-doc/` | same | turning a non-private document into proposals |

## Placeholders

`init-vault` replaces these in `.mcp.json` and `.claude/settings.json`:

| placeholder | value |
|---|---|
| `{{REPO_DIR}}` | absolute path of this code repository (where `uv run brain …` works) |
| `{{VAULT_DIR}}` | absolute path of the vault (also exported as `BRAIN_VAULT` to the MCP server) |

Substitute absolute, `~`-expanded paths and JSON-escape them. In the hook command the repo path
sits inside double quotes, so paths with spaces work.

## Permission rules, and why (Claude Code)

These live in `.claude/settings.json` and apply to Claude Code. If you use another agent, give it
equivalent rules if it supports them; `AGENTS.md` states the same rules in prose either way.

- `deny Read` on `raw/`, `wiki/`, `journal/`, `episodes/`, `reviews/`, `.brain/`, `.privacy/`,
  `.git/` and the generated root files (`core.md`, `index.md`, `log.md`, `questions.md`,
  `intentions.md`). All of these can hold private content. The agent must go through MCP, which
  filters it out. A Read deny also blocks Edit/Write on the same path.
- `deny Edit(/**)`: the brain has a single writer. The big model never writes in the vault.
  (Claude Code ignores `Write(path)` rules, and `Edit` covers every file-editing tool.)
- `deny Grep`, `deny Glob`, `ask Bash`: path rules don't stop `grep -r` from the vault root or
  scripts that open files, so searching is removed and every shell command needs the owner's OK.
- `allow mcp__brain__*` and `enabledMcpjsonServers: ["brain"]` take effect only after the owner
  accepts Claude Code's workspace-trust dialog for the vault. Deny rules apply immediately.

These rules are a guardrail, not a sandbox. The primary guarantee is that the MCP server never
returns `privacy: private` content.

## SessionEnd hook (Claude Code)

`brain hook session-end` reads the hook JSON from stdin (`session_id`, `transcript_path`, `cwd`,
`reason`) and copies `transcript_path` to `raw/external/claude-code/YYYY/MM/<session_id>.jsonl`. It must finish
well within the 30 s timeout (only copy; ingestion happens later in the pipeline), and be
idempotent, because `/clear` and `/resume` also fire SessionEnd. Other clients have no such hook:
their conversations come back through `brain capture` (see `docs/architecture.md`, Flow 4).

## Changing the template

Edit it here, in the repo, not in a vault. Re-running `init-vault` on an existing vault should
update these files. Keep every example on the fictional persona (Alex Rivera, Northwind Labs,
Sam Okafor, Atlas Migration). This repository is public.
