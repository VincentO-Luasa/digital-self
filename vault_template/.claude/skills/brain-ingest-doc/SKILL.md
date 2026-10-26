---
name: brain-ingest-doc
description: Turn a non-private document the owner gives you (CV, project write-up, bio, talk abstract, public profile) into brain_propose operations (entities, facts, aliases) using the brain's predicate vocabulary. Use when the owner pastes or points to a document and asks to add it to the brain or "teach the brain" from it. If the document is private (health, family, relationships, finances, personal journals), stop and send them to local `brain ingest` instead.
---

# Ingest a non-private document

## 0. Privacy gate (always first)

Before extracting anything, decide whether the document is private. It is private if it is mostly
about:

- health, therapy, medication;
- family, partner, friends, or other people's personal lives;
- finances, legal matters, identity documents;
- personal journals, letters, or chat exports with people close to the owner.

If it is private, or you are unsure, stop and say:

> This looks private, so it shouldn't go through a hosted model. Ingest it locally instead:
> `brain ingest <path>`. The local model extracts it and nothing leaves your machine.

Don't summarise or extract from a private document even if it has already been pasted. A CV
with a short "interests" section is fine. Skip any personal lines that do appear in an otherwise
public document.

If the document is in the vault (`raw/docs/…`), you can't read it here. Ask the owner to run
`brain ingest` on it, or to paste it if it isn't private.

## 1. Read and plan

- Identify the document type, date and perspective (written by the owner? about the owner?).
- List the entities: orgs, projects, places, topics/skills, and only those people who appear in a
  professional role.
- For each entity, `brain_search` its name and aliases to find an existing id. Reuse ids. Only
  `create_entity` when nothing matches. Suggest `add_alias` when the document uses a new spelling
  ("NW Labs" → `orgs/northwind-labs`).

## 2. Extract facts

Map statements to predicates from the v0.4 vocabulary in `AGENTS.md`. Typical CV mapping:

| document says | operation |
|---|---|
| "Data Engineer, Northwind Labs, Feb 2023 – present" | `works_at` o=`[[orgs/northwind-labs]]` from=2023-02; `role` o="Data Engineer" from=2023-02 |
| earlier job with dates | `works_at` with `from` and `to` |
| "Led the Atlas Migration to the new warehouse" | `worked_on` o=`[[projects/atlas-migration]]`; on the project: `owner` / `goal` |
| "MSc Computer Science, Westbridge University, 2018 – 2020" | `studied_at` o=`[[orgs/westbridge-university]]` + `degree` with `extra: {"at": "[[orgs/westbridge-university]]"}`, both with `from`/`to` |
| skills list | one `has_skill` per skill, `o` = topic page if it exists, else literal |
| "Fluent in Spanish" | `speaks` o="Spanish" |
| certifications | `certification` |

Rules:

- `kind: observed` for everything taken from a document. Never `stated`, and never `inferred`
  unless you say so explicitly and give evidence.
- Valid time (`from` / `to`) comes from the document, at the precision it gives (`2023`,
  `2023-02`). For "present", omit `to`.
- `text` is a plain sentence in the third person: "Worked at Northwind Labs as data engineer".
- One fact per operation. Don't put two facts in one line. Skip marketing fluff ("passionate
  about data") unless the owner wants it as `interest`.
- If the document contradicts current memory (e.g. a different start date), don't overwrite.
  Propose `supersede_fact` only when the document is clearly newer and more authoritative, and say
  why in the rationale. Otherwise flag the conflict to the owner.
- Facts about other people (Sam Okafor's job title) go on their page with `subject: people/sam-okafor`
  only if they're professional and stated in the document.

## 3. Propose

- One `brain_propose` call per document, or per section for long documents (max ~30 operations
  per call). Order: `create_entity`, then `add_alias`, then `add_fact` / `supersede_fact`.
- The rationale names the document and its date: `Ingested CV (Alex Rivera, version 2026-09) pasted in this session.`
- Show the owner a compact table of what you proposed (entity · predicate · object · valid time)
  and anything you deliberately skipped, with the reason.
- Remind them: approve with `brain review`. This transcript, including the document text, is
  captured into `raw/external/<provider>/` as the provenance.

## Don't

- Don't write the document into the vault yourself, or try to copy it into `raw/docs/`.
- Don't invent predicates. Use `note` sparingly for things that matter but fit nothing.
- Don't extract from documents that belong to someone else, or third-party personal data
  beyond a professional role.
