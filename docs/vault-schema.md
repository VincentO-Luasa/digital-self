# Vault schema (v0.1)

The vault is the brain's long-term memory and the **single source of truth**. It is a folder of
Obsidian-compatible markdown, versioned with a *local-only* git repo, living **outside** this code
repository (default `~/brain-vault`, override with `BRAIN_VAULT`). Everything else — the SQLite
index, embeddings, coverage scores — is derived from it and can be rebuilt with `brain reindex`.

Design lineage: Karpathy's *LLM Wiki* (raw sources → LLM-maintained wiki → schema file; ingest /
query / lint), plus the fixes the agent-memory literature says matter: bi-temporal facts,
supersession instead of deletion, mandatory provenance, entity resolution with aliases, and an
explicit stated-vs-inferred split. See `docs/research.md`.

> All examples in this repository use a fictional persona (**Alex Rivera**, a data engineer at the
> fictional **Northwind Labs**). Never put real personal data in this repo.

## Layout

```
brain-vault/
  CLAUDE.md            # workspace instructions for Claude (installed from vault_template/)
  .claude/             # skills + settings (SessionEnd hook) for escalation sessions
  .mcp.json            # brain MCP server, privacy-filtered
  SCHEMA.md            # copy of this spec's version tag + vault-local conventions
  core.md              # ≤1.5k-token profile, always in the local model's context (generated)
  index.md             # catalog: one line per wiki page, grouped by region (generated)
  log.md               # append-only timeline of every operation (generated)
  questions.md         # open knowledge gaps the brain wants to ask about (generated + manual)
  life-map.yaml        # optional override of config/life-map.yaml (regions + slots)
  raw/                 # IMMUTABLE sources — episodic ground truth
    docs/              #   imported documents: CV, website snapshot, exports
    conversations/YYYY/MM/<session-id>.md   # full local chat / interview transcripts
    claude/YYYY/MM/<session-id>.jsonl       # Claude Code transcripts copied by the SessionEnd hook
    inbox/             #   dropped files / pasted claude.ai chats waiting for ingest
  episodes/YYYY/MM/<episode-id>.md          # topic-sized chunks of conversations (see below)
  wiki/                # SEMANTIC memory — LLM-maintained, human-editable
    self/              #   profile.md, traits.md, goals.md, routine.md, health.md, …
    people/  orgs/  projects/  places/  topics/  events/
  journal/YYYY-MM.md   # dated observations with importance scores
  reviews/
    pending/<id>.md    # proposed changes awaiting approval
    done/YYYY/MM/<id>.md
  .brain/              # derived, git-ignored inside the vault: index.sqlite, logs, caches
  .privacy/denylist.txt  # terms the repo pre-commit hook refuses to commit (generated + manual)
```

## Pages (wiki/)

Every wiki page has YAML frontmatter:

```yaml
---
id: people/sam-okafor        # stable; equals path under wiki/ without .md
type: person                 # person | org | project | place | topic | event | self
subject: people/sam-okafor   # entity the facts on this page are about (wiki/self/* pages use `self`)
aliases: [Sam, Samuel Okafor]
regions: [relationships.friends]
privacy: private             # normal | private — private pages never leave the machine unconfirmed
created: 2026-10-25
updated: 2026-10-25
---
```

Body sections, in order:

1. `# Title`
2. **Summary** — free prose written by the LLM at consolidation time. Never the only place a fact
   lives.
3. `## Facts` — current facts, one per line, machine-parseable (below). Written by code only.
4. `## History` — superseded facts, same format, with `to::` and `by::` set.
5. `## Notes` — optional free-form human notes; the pipeline reads them as a source but never
   rewrites them.

### Fact lines

```markdown
- Works at Northwind Labs as data engineer (p:: works_at) (o:: [[orgs/northwind-labs]]) (from:: 2023-02) (rec:: 2026-10-25) (src:: [[ep-20261025-1412-career]]) (kind:: stated) ^f-k3x9q2
```

| field | required | meaning |
|---|---|---|
| leading text | yes | human-readable sentence, rendered by code |
| `p::` | yes | predicate from the controlled vocabulary below |
| `o::` | yes | object: a wikilink to an entity, or a literal value |
| `from::` / `to::` | no | **valid time** — when it was true in the world (`YYYY`, `YYYY-MM`, or `YYYY-MM-DD`) |
| `rec::` | yes | **record time** — when the brain learned it |
| `src::` | yes | provenance: episode, raw doc, or review id. **No source, no fact.** |
| `kind::` | yes | `stated` (you said it), `observed` (in a document), `inferred` (the brain concluded it) |
| `evidence::` | inferred only | list of sources supporting an inference (no fake confidence numbers) |
| `by::` | history only | block id of the superseding fact |
| `^f-xxxxxx` | yes | block id: `f-` + 6 lowercase base32 chars; Obsidian can link to it |

Rules:
- **Supersede, never delete.** A contradicted fact moves to `## History` with `to::` (end of
  validity) and `by::`. Queries default to current facts; history answers "when / used to".
- **Contradiction checks are scoped** to the same `subject` + `p` (+ `o` for multi-valued
  predicates). Unscoped invalidation is a known failure mode (Graphiti #1728).
- **Inferred facts** (traits, patterns, flaws) always go through the review queue.
- Hand-edits in Obsidian are allowed; `brain lint` validates and normalises the format.

### Predicate vocabulary (v0.1)

Single-valued per subject unless marked *(multi)*:

- identity: `name`, `born`, `hometown`, `lives_in`, `nationality` *(multi)*, `speaks` *(multi)*
- work: `works_at` *(multi)*, `role`, `team`, `manager`, `responsibility` *(multi)*, `worked_on` *(multi)*
- education: `studied_at` *(multi)*, `degree` *(multi)*, `certification` *(multi)*
- skills: `has_skill` *(multi, o = topic page or literal; optional `level::`)*
- relationships: `family_of`, `partner_of`, `friend_of`, `colleague_of`, `mentor_of`, `knows` *(all multi; optional `rel::` e.g. sister)*; `met_at`, `last_seen`
- projects: `owner`, `status`, `goal`, `stakeholder` *(multi)*, `deadline`, `uses` *(multi)*
- self: `has_strength`, `has_flaw`, `values`, `goal`, `habit`, `interest`, `likes`, `dislikes`, `fears`, `health` *(all multi)*
- generic: `member_of`, `located_in`, `part_of`, `related_to` *(multi)*, `note` *(multi, last resort)*

New predicates are added here first (bump the schema version), never invented ad hoc by the LLM.

## Episodes (episodes/)

Conversations are segmented by **topic** into episodes — the unit of extraction. One chat may
yield several episodes; an interview session usually yields one per region it covered.

```yaml
---
id: ep-20261025-1412-career
type: episode
session: conv-20261025-1400        # raw/conversations/... transcript
turns: [3, 17]                     # inclusive turn range in that transcript
channel: interview                 # chat | interview | claude | import
regions: [career.current_job]
entities: ["[[orgs/northwind-labs]]", "[[people/sam-okafor]]"]
importance: 6                      # 1–10, scored at segmentation time
started: 2026-10-25T14:12
status: extracted                  # pending | extracted | consolidated
---
<3–6 sentence summary of what was said>
```

## Journal (journal/YYYY-MM.md)

Append-only dated observations — the bridge between episodes and wiki pages:

```markdown
- 2026-10-25 · imp 7 · Alex is anxious about the Q1 platform migration deadline · [[ep-20261025-1412-career]] ^o-4hd8sa
```

Recency for retrieval is measured from **last access**, so recalled memories are reinforced.
Low-importance, never-accessed observations are summarised and archived at consolidation; they
are never silently dropped.

## Generated files

- `core.md` — rewritten at consolidation from the highest-salience current facts: identity, current
  role and projects, key people, current goals and worries. Hard cap 1.5k tokens.
- `index.md` — `- [[id]] — one-line summary`, grouped by region. Read first by any agent.
- `log.md` — `## [2026-10-25 14:30] extract | ep-20261025-1412-career | +3 facts, 1 superseded`.
- `questions.md` — knowledge gaps with region, reason and priority; interviews draw from it.

## Reviews (reviews/pending/)

Every change the pipeline is not allowed to auto-apply becomes a review file: a JSON block of
operations (`create_entity`, `add_fact`, `supersede_fact`, `merge_entities`, `update_summary`,
`add_alias`) plus a human-readable diff. Auto-applied: plain stated/observed facts with no
contradiction. Queued: inferred facts, relationship changes, supersessions, merges, anything on a
`private` page created by Claude.

## Privacy

- Default `privacy: private` for regions `relationships.*`, `self.health`, `self.traits`, and any
  page you mark. The MCP server never returns private content to Claude; private facts reach Claude
  only inside a handoff packet you confirmed.
- `.privacy/denylist.txt` (one term per line, generated from private pages' titles and aliases +
  manual entries) is read by the code repo's pre-commit hook to block accidental leaks.
