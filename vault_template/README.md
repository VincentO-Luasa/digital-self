# vault_template

`brain init-vault` copies this folder into the owner's private vault (default `~/brain-vault`).
The vault is then also a **Claude Code workspace**. When the local model shows `→ Ask Claude`, the
owner opens Claude Code in the vault, and Claude acts as the brain's "big brain". It reads memory
only through the privacy-filtered `brain` MCP server and changes memory only by proposing reviews.

This README is not copied into the vault.

## What gets installed

| file | becomes | purpose |
|---|---|---|
| `CLAUDE.md` | `<vault>/CLAUDE.md` | Claude's runtime role, hard privacy rules, tool list, operation and predicate reference |
| `.mcp.json` | `<vault>/.mcp.json` | registers the `brain` stdio MCP server (`brain mcp`) |
| `.claude/settings.json` | `<vault>/.claude/settings.json` | SessionEnd hook (transcript → `raw/claude/YYYY/MM/`), permission rules |
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

## Permission rules, and why

- `deny Read` on `raw/`, `wiki/`, `journal/`, `episodes/`, `reviews/`, `.brain/`, `.privacy/`,
  `.git/` and the generated root files (`core.md`, `index.md`, `log.md`, `questions.md`,
  `intentions.md`). All of these can hold private content. Claude must go through MCP, which
  filters it out. A Read deny also blocks Edit/Write on the same path.
- `deny Edit(/**)`: the brain has a single writer. Claude never writes in the vault. (Claude Code
  ignores `Write(path)` rules, and `Edit` covers every file-editing tool.)
- `deny Grep`, `deny Glob`, `ask Bash`: path rules don't stop `grep -r` from the vault root or
  scripts that open files, so searching is removed and every shell command needs the owner's OK.
- `allow mcp__brain__*` and `enabledMcpjsonServers: ["brain"]` take effect only after the owner
  accepts Claude Code's workspace-trust dialog for the vault. Deny rules apply immediately.

These rules are a guardrail, not a sandbox. The primary guarantee is that the MCP server never
returns `privacy: private` content.

## SessionEnd hook

`brain hook session-end` reads the hook JSON from stdin (`session_id`, `transcript_path`, `cwd`,
`reason`) and copies `transcript_path` to `raw/claude/YYYY/MM/<session_id>.jsonl`. It must finish
well within the 30 s timeout (only copy; ingestion happens later in the pipeline), and be
idempotent, because `/clear` and `/resume` also fire SessionEnd.

## Changing the template

Edit it here, in the repo, not in a vault. Re-running `init-vault` on an existing vault should
update these files. Keep every example on the fictional persona (Alex Rivera, Northwind Labs,
Sam Okafor, Atlas Migration). This repository is public.
