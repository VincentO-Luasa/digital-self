# Vault schema (v0.3)

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
  AGENTS.md            # runtime instructions for the big model working in the vault (from vault_template/)
  CLAUDE.md            # thin Claude Code wrapper: imports AGENTS.md + Claude-Code-specific notes
  .claude/             # Claude Code only: skills + settings (SessionEnd hook, permissions)
  .mcp.json            # brain MCP server, privacy-filtered, for any MCP-capable client
  SCHEMA.md            # copy of this spec's version tag + vault-local conventions
  core.md              # ~300–600-token profile card, always in the local model's context (generated)
  index.md             # catalog: one line per wiki page, grouped by region (generated)
  log.md               # append-only timeline of every operation (generated)
  questions.md         # open knowledge gaps the brain wants to ask about (generated + manual)
  intentions.md        # prospective memory: "bring up X when Y comes up / on date D"
  life-map.yaml        # optional override of config/life-map.yaml (regions + slots)
  raw/                 # IMMUTABLE sources — episodic ground truth
    docs/              #   imported documents: CV, website snapshot, exports
    conversations/YYYY/MM/<session-id>.md   # full local chat / interview transcripts
    external/<provider>/YYYY/MM/<session-id>.*   # big-model conversations (see below)
    inbox/             #   dropped files / pasted web chats (ChatGPT, Kimi, Claude.ai, …) waiting for ingest
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
4. `## History` — superseded facts, same format, with `to::` set (and `by::` when a newer fact
   replaced them).
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
| `by::` | history only | block id of the superseding fact; absent when a fact simply ended |
| `level::`, `rel::`, `at::` | no | qualifiers of some predicates (see the vocabulary): a skill's level, a relation's kind, the school a degree is from |
| `^f-xxxxxx` | yes | block id: `f-` + 6 lowercase base32 chars; Obsidian can link to it |

Rules:
- **Supersede, never delete.** A contradicted fact moves to `## History` with `to::` (end of
  validity) and `by::`. Queries default to current facts; history answers "when / used to".
  When an approved `add_fact` contradicts a current fact, the old one is retired this way:
  `to::` is the new fact's `from::` if that is later, otherwise today.
- **Restating is not contradicting.** A new fact with the same subject, `p` and `o` as a current
  fact is a duplicate (skipped) when its dates are missing, equal or less precise. Different dates
  (e.g. a new `to::`) make it a contradiction, which goes to review.
- **Back-to-back is not overlap.** A fact ending in `2025-03` and one starting in `2025-03` hand over;
  they don't clash.
- **Contradiction checks are scoped** to the same `subject` + `p` (+ `o` for multi-valued
  predicates). Unscoped invalidation is a known failure mode (Graphiti #1728).
- **Inferred facts** (traits, patterns, flaws) always go through the review queue.
- Hand-edits in Obsidian are allowed; `brain lint` validates and normalises the format.

### Predicate vocabulary (v0.3)

Single-valued per subject unless marked *(multi)*:

- identity: `name`, `born`, `hometown`, `lives_in`, `nationality` *(multi)*, `speaks` *(multi)*
- work: `works_at` *(multi)*, `role`, `team`, `manager`, `responsibility` *(multi)*, `worked_on` *(multi)*
- education: `studied_at` *(multi)*, `degree` *(multi; optional `at::` = wikilink to the school's org page, e.g. `(at:: [[orgs/westbridge-university]])`)*, `certification` *(multi)*
- skills: `has_skill` *(multi, o = topic page or literal; optional `level::`)*
- relationships: `family_of`, `partner_of`, `friend_of`, `colleague_of`, `mentor_of`, `knows` *(all multi; optional `rel::` e.g. sister)*; `met_at`, `last_seen`
- projects: `owner`, `status`, `goal`, `stakeholder` *(multi)*, `deadline`, `uses` *(multi)*
- self: `has_strength`, `has_flaw`, `values`, `goal`, `habit`, `interest`, `likes`, `dislikes`, `fears`, `health` *(all multi)*
- generic: `member_of` *(multi: clubs, communities, groups)*, `located_in`, `part_of`, `related_to` *(multi)*, `note` *(multi, last resort)*

New predicates are added here first (bump the schema version), never invented ad hoc by the LLM.

`at::` (v0.3) ties a degree to the `studied_at` school it came from. Ingest sets it from the line
of the document that names both; the profile card pairs a degree with its school by this link
and never by overlapping dates alone. A degree without it (older vaults, hand edits) still
works: the card falls back to equal or clearly overlapping study periods. Entity merges repoint
it like `o::`.

## Episodes (episodes/)

Conversations are segmented by **topic** into episodes — the unit of extraction. One chat may
yield several episodes; an interview session usually yields one per region it covered.

```yaml
---
id: ep-20261025-1412-career
type: episode
session: conv-20261025-1400        # raw/conversations/... transcript
turns: [3, 17]                     # inclusive turn range in that transcript
channel: interview                 # chat | interview | external | import
# provider: claude-code            # channel external only: the escalation provider id
regions: [career.current_job]
entities: ["[[orgs/northwind-labs]]", "[[people/sam-okafor]]"]
importance: 6                      # 1–10, scored at segmentation time
gist: Alex worries the Atlas migration deadline is unrealistic   # ≤30 tokens, used by retrieval
started: 2026-10-25T14:12
status: extracted                  # pending | extracted | consolidated
---
<3–6 sentence summary of what was said>
```

## Big-model conversations (raw/external/)

Conversations with the big model (`docs/architecture.md`, Flow 4) are stored per escalation
provider, so the pipeline knows where a transcript came from and how to parse it:

| provider | path | written by |
|---|---|---|
| `claude-code` | `raw/external/claude-code/YYYY/MM/<session-id>.jsonl` | the SessionEnd hook (Claude Code's own transcript format) |
| `openai-compatible` | `raw/external/openai-compatible/YYYY/MM/<session-id>.md` | `brain ask` (packet + reply; frontmatter records `model` and `base_url` host) |
| `clipboard` | `raw/external/clipboard/YYYY/MM/<session-id>.md` | `brain capture` (pasted or exported chat) |

Episodes cut from these transcripts have `channel: external` and `provider: <provider>`.

## Intentions (intentions.md)

Prospective memory — things to bring up when a cue appears or a date arrives:

```markdown
- [ ] Ask how the Atlas go-live went (when:: [[projects/atlas-migration]]) (after:: 2026-11-15) (src:: [[ep-20261025-1412-career]]) ^i-p2m7wd
```

## Access log (derived, not in markdown)

Retrieval keeps per-item access timestamps in `.brain/access.sqlite` (base-level activation,
co-use edges). It cannot be rebuilt from markdown. Losing it only resets activation to
creation time, so it is backed up but not versioned.

## Journal (journal/YYYY-MM.md)

Append-only dated observations — the bridge between episodes and wiki pages:

```markdown
- 2026-10-25 · imp 7 · Alex is anxious about the Q1 platform migration deadline · [[ep-20261025-1412-career]] ^o-4hd8sa
```

Recency for retrieval is measured from **last access**, so recalled memories are reinforced.
Low-importance, never-accessed observations are summarised and archived at consolidation; they
are never silently dropped.

## Generated files

- `core.md` — the profile card, rebuilt deterministically from current facts (no LLM, no
  timestamps): an identity line (`Name: … · role at employer (team) · lives in … · speaks …`),
  then `## Who they are`, `## Now`, `## Life story`, `## People who matter`,
  `## Interests and skills`, `## Open loops`. Multi-valued facts are grouped into comma lists,
  jobs/studies/cities into lifetime periods; empty sections are left out. Aims at ~600 tokens,
  hard cap `core_max_tokens`. Layout and rules: `docs/retrieval.md` ("L0: schema card").
- `index.md` — `- [[id]] — one-line summary`, grouped by region. Read first by any agent.
- `log.md` — `## [2026-10-25 14:30] extract | ep-20261025-1412-career | +3 facts, 1 superseded`.
- `questions.md` — knowledge gaps with region, reason and priority; interviews draw from it.

## Reviews (reviews/pending/)

Every change the pipeline is not allowed to auto-apply becomes a review file: a JSON block of
operations (`create_entity`, `add_fact`, `supersede_fact`, `merge_entities`, `update_summary`,
`add_alias`, `add_intention`). The canonical JSON shape of each operation is the Pydantic model in
`brain.vault.ops`; `vault_template/AGENTS.md` mirrors it. Each review also carries a
human-readable diff. Auto-applied: plain stated/observed facts with no contradiction. Queued:
inferred facts, relationship changes, contradictions, supersessions, merges, and any change from
the big model that touches a `private` page (as subject, as the object of a fact, or through a
region the life map marks private).

## Privacy

- Default `privacy: private` for the regions marked `privacy: private` in `config/life-map.yaml`
  (family, friends, traits, health) and any page you mark. Colleagues are `normal` by default. The MCP server never returns private content to the big model; private facts reach it
  only inside a handoff packet you confirmed.
- `.privacy/denylist.txt` (one term per line, generated from private pages' titles and aliases +
  manual entries) is read by the code repo's pre-commit hook to block accidental leaks.
