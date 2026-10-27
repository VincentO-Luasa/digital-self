---
name: lint
description: How `brain lint` cleans the knowledge graph — duplicate entities (names, embeddings, reranker, typed decision, exclusive "which one?"), doubtful page types (demote/retype), facts on the wrong subject (refile), missing structural edges (part_of, a duty's job, located_in), code-only consistency checks, questions for the owner, incremental runs — plus the writer ops it relies on (schema v0.5), its prompts and fixtures, and `brain eval lint`. Use when changing src/brain/lint/, src/brain/prompts/lint_*.md, the `lint` / `eval lint` CLI commands, the demote/retype/refile/merge code in brain.vault.writer, or tests/lint/ and tests/eval/lint/.
---

# Graph cleaning (`brain lint`)

Spec: `docs/vault-schema.md` ("Graph cleaning operations (v0.5)"), `docs/architecture.md`
(Flow 3, lint row), decision-log entry "brain lint: graph cleaning".

## Flow (`src/brain/lint/`)

`run_lint(vault, llm, LintOptions)` reads a `Graph` snapshot and runs the checks in order
(`CHECKS`); nothing is written. `apply_lint(vault, report)` commits uncommitted hand edits
apart, adds questions (`brain.interview.questions.add_question`, origin `lint`, `doubt::` = the
finding key, schema v0.6, so an interview can settle it; never the same doubt twice: the key
before `:` in `reason`), applies **one change set per finding** (origin
`lint`; every lint op is reviewed, so each repair is its own review) and records the revision in
`.brain/lint.json`. The next run checks only pages changed since (git) and their neighbours;
`--all` / `--since REV` override.

| module | check | model | act | otherwise |
|---|---|---|---|---|
| `duplicates.py` | same-type pairs by name signals (`names.py`: core words, acronym, typo, contains, first name, shared first word) or profile embeddings ≥ 0.75; reranker orders a page's candidates (≤ 3) | choice same/different; several look-alikes → one exclusive "which one, or none?" | `merge_entities` at p ≥ 0.9 | 0.5–0.9 → question |
| `kinds.py` | pages used as values (`has_skill`, `interest`, `uses`…) or with nothing that fits their type | choice person/org/project/place/event/topic/tool/activity (current type never shown) | value kinds' mass ≥ 0.9 → refile facts that used it as a page, `demote_entity`; other type ≥ 0.9 → `retype_entity` | question (also: a page with facts of its own is never demoted) |
| `subjects.py` | owner's `goal`/`deadline`/`status`/`uses` whose text or source sentences name a project | choice owner / project… / unclear | `refile_fact` subject at 0.8 | question (also when the project already has that value) |
| `edges.py` | project without `part_of` (no source naming it: asked directly); undated duty without `at::` (several jobs) or `at::` contradicted by its source; org without `located_in` whose source sentences name a known place | choices over the owner's jobs (+ unclear / side project), places (+ another place / not stated) | inferred `part_of` / `located_in` with `evidence::`, `refile_fact` with `at` at 0.8 | question; a confident side project adds nothing |
| `consistency.py` | code only: two current single-valued values, two open jobs, end before start, orphans (no facts, no links, no episode), literal objects naming a page, parse warnings, inferred facts without evidence, AGENTS.md predicate list vs `models.py` | — | `refile_fact` o → link | question / report |

`Context.busy` holds pages an op of this run removes or renames (merged away, demoted,
retyped): no other check touches them in the same run. Fallback decisions (runtime without
`/v1/decisions`) never act.

## Rules worth keeping

- Precision first: a wrong merge corrupts two pages. Acting thresholds: 0.9 for anything that
  removes or renames a page, 0.8 for re-filing and edges; 0.5 up to them → a question for the
  owner (the oracle); below 0.5 nothing.
- Questions describe the evidence, never the current filing (shown a proposal, the model agrees).
- Escape options first ("unclear", "none of them", "another place", "a side project"), and never
  a forced choice: offer employers without a page too.
- Owner-facing questions name at most two employers, ranked by evidence (`likely_jobs`:
  sentences naming both, a linking fact, overlapping dates at 0.5 only, the model's pick); one
  when it stands out, none ("Which of your jobs is X part of…?") when nothing points anywhere.
  Never "really a project"; no claim the owner listed anything (ingest made the page).
- An interview settles `part_of`, `type`, `merge` and `job` doubts from the owner's answer
  (`brain.interview.settle`, the interview skill); keep finding keys stable (`<kind> <ids>`). With a listed city first and the true one
  absent, the model picked the listed one at 0.98 (decision log).
- A type change needs a fact or link that says so (`kinds._support`); a verdict on a bare name
  is a report line only (made-up place names read as people's at p ≥ 0.9).
- Evidence is excerpts: sentences of the facts' sources naming the thing (`Graph.excerpt`,
  `subjects.related_sentences`); a list item carries the line introducing it (`graph.units`).
- General methods only: no rules for one vault's phrasing. New name rules go in `names.py` with a
  test case; judge every change on `brain eval lint`.

## Writer ops (schema v0.5, `brain.vault.writer`)

`demote_entity` (links → literal, page removed; refused with facts or notes of its own),
`retype_entity` (moved + repointed), `refile_fact` (subject / p / literal o → link / `at`; same
statement, id, dates; refused if it would contradict). A merge folds restatements, retires
self-references, queues clashes of single-valued facts as a new review, repoints pending reviews.
Inferred facts need `evidence::` (two from the big model).

## Prompts and fixtures

- Inputs: `lint_same_entity`, `lint_which_entity`, `lint_entity_kind`, `lint_fact_subject`,
  `lint_part_of`, `lint_duty_job`, `lint_located_in` (first comment line carries the version);
  the questions are constants next to the code. Bump the version on any change.
- Replay fixtures: `tests/fixtures/lint/llm/` (decisions on the eval's dirty vault, seed 1,
  embeddings and rerank off), replayed by request (`tests/lint/linttools.Replay`). Re-record after
  changing a prompt, a question, the persona or the noise:
  `BRAIN_RECORD_FIXTURES=1 uv run pytest -m llm tests/lint/test_lint_live.py -s`.

## Evaluation

`uv run brain eval lint [--seed N ...] [--no-cache] [--json out.json] [--keep]`: a clean
fictional vault (`tests/fixtures/vault` + `tests/fixtures/lint/eval-vault`) linted as is (any op is
wrong), and per seed a dirty copy (`lint_eval.inject`: 12 noise items, variants by seed, plus
near misses that must stay untouched). Every lint review is approved and the vault verified.
Report: fixes, **wrong changes (must be 0)**, questions, expected questions, ops, decisions,
seconds. Run `--no-cache` on two seeds before and after a change and diff the JSON.

## Checklist

- [ ] `uv run pytest tests/lint tests/eval/lint`, `uv run pytest -m llm tests/lint`, ruff clean.
- [ ] `brain eval lint --no-cache --seed 1 --seed 2`: wrong 0 on clean and dirty runs; read the
      questions and notes.
- [ ] No new auto-apply path: lint ops stay reviewed.
- [ ] A decision-log entry for any non-obvious choice.
