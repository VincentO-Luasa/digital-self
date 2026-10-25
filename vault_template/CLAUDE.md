@AGENTS.md

# Claude Code notes

Everything in `AGENTS.md` (imported above) applies. This file only adds what is specific to Claude
Code.

- **Skills** live in `.claude/skills/` (`brain-answer`, `brain-review`, `brain-reflect`,
  `brain-ingest-doc`) and load when a task matches their description.
- **SessionEnd hook** (`.claude/settings.json`): when the session ends (also on `/clear`), the hook
  copies this transcript to `raw/external/claude-code/YYYY/MM/<session-id>.jsonl` for the local
  pipeline. Nothing to save by hand.
- **Permissions** (`.claude/settings.json`): Read is denied on the memory folders and the generated
  root files, Edit is denied everywhere, Grep and Glob are off, and every Bash command asks the
  owner. A denied call means "use the `brain` MCP tools", never "find another way in".
- The `brain` MCP server comes from `.mcp.json` and is enabled once the owner accepts the
  workspace-trust dialog for this folder.
