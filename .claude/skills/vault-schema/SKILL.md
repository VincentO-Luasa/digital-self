---
name: vault-schema
description: Conventions of the brain's markdown vault — layout, page frontmatter, bi-temporal fact lines (p::/o::/from::/to::/rec::/src::/kind::, ^f- block ids), episodes, journal, reviews, predicates, privacy flags. Use whenever writing or changing code that parses, renders, validates or migrates vault files, when creating test fixtures, or when editing vault pages by hand.
---

# Vault schema

The canonical spec is `docs/vault-schema.md` — read it fully before touching vault formats.
Interview regions and slots are in `config/life-map.yaml`.

Checklist when changing anything vault-related:

1. Spec first: edit `docs/vault-schema.md`, bump `Vault schema (vX.Y)`, add a line to
   `docs/decisions.md` explaining why.
2. Parser and renderer must round-trip: `render(parse(page)) == page` for well-formed pages.
   Add a fixture under `tests/fixtures/vault/` (fictional persona Alex Rivera only) and a
   round-trip test.
3. Never delete facts: supersession moves a fact to `## History` with `to::` and `by::`.
4. Contradiction checks are scoped to the same `subject` + `p` (+ `o` for multi-valued predicates).
5. Every fact needs `src::`, `rec::`, `kind::`, and a unique `^f-xxxxxx` block id.
6. New predicates go in the spec's vocabulary first; the LLM may not invent predicates.
7. If the change affects existing vaults, write a migration in `brain lint --fix` and say so in
   the decision log.
8. Privacy: regions marked `privacy: private` in the life map produce `privacy: private` pages;
   private content never goes into MCP responses or unconfirmed handoff packets.

For Obsidian syntax details (wikilinks, block ids, properties, callouts) use the
`obsidian-markdown` skill.
