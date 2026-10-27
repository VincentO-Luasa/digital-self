---
name: memory
description: How the memory pipeline turns a finished chat or interview transcript into episodes, facts, journal lines and intentions — segmentation, entity resolution, predict-calibrate extraction, owner corrections, private answers, idempotency — plus how to version its prompts, re-record fixtures and run `brain eval memory`. Use when changing anything under src/brain/memory/, src/brain/prompts/memory_*.md, the `chat`/`memory` CLI commands, or tests/memory/ and tests/eval/memory/.
---

# Memory pipeline

Entry point: `brain.memory.process_session(vault, llm, transcript, *, channel, regions_hint=None,
today=None, db=None, refresh=True, core_max_tokens=1500, private_turns=None, turns=None) ->
SessionReport`.
The interview calls exactly this; keep the positional arguments and add keyword-only ones with
defaults. `brain chat` runs it when the session closes (`--no-memory` skips);
`brain memory process [TRANSCRIPT|--pending]` runs it on saved transcripts.

## Flow (src/brain/memory/)

1. **Render** (`conversation.py`): `[n] OWNER: …` in full, `[n] ASSISTANT: …` clipped to 60
   words, interview markers (`[interview] region:: … · topic:: … · about:: [[id]]`,
   `[interview] privacy:: private …`, `[interview] skip …`) as `[n] NOTE: …`; other system turns
   are dropped. Facts come only from the owner; NOTE lines are hints, never facts.
2. **Gate** (`segment.gate`): one yes/no typed decision on the owner's lines ("does the owner say
   anything about their own life…?"). p < 0.15 → no episode, a `no episode` log line (the session
   is never segmented again). No answer or an error → segmented.
3. **Segment** (`segment.segment`, prompt `memory_segment`, schema `Segmentation` with regions as an
   enum of the life map): episodes with turn ranges, topic, regions, mentions, gist, summary and
   importance labels. Code computes importance 1–10 (`BASE 2` + emotion 0/1/3 + goal 0/1/2 +
   change 0/1/3 + future 1 + remember 3), repairs ranges (clamped, sorted, disjoint), drops
   episodes without an owner turn and unknown regions, caps gists at 22 words.
4. **Resolve** (`resolve.py`) each episode's mentions: exact alias → page; else candidates by
   words and by instructed embeddings (cosine ≥ 0.55, top 3; never merged on embeddings alone) →
   one typed choice "who or what is X?" + "someone else"; p ≥ 0.8 picks it and adds the name as
   an alias. Unresolved mentions may become new pages at extraction.
5. **Write episodes** `pending` (`add_episode`, one commit, source = the transcript). Marker regions
   and `about::` pages join the episode's own; an episode holding a private turn (`private_turns`
   or the marker) is `privacy: private`.
6. **Extract** per episode (`extract.py`, prompt `memory_extract`, schema `MemoryExtraction`):
   predict-calibrate. The prompt shows KNOWN FACTS (current and ended facts of the mentioned
   pages and the owner's facts of the episode's regions or sharing words, ≤ 900 tokens) and the
   model labels each fact `new` / `confirms` / `contradicts` / `noop` with the known fact's id.
   `calibrate`: noop dropped; confirms checked (same predicate, similar value) → `confirm_fact`;
   contradicts without a value → end the known fact; met_at/last_seen about the owner dropped.
   Then ingest's machinery, unchanged: `check_part` (grounding, typed checks, story details),
   `build_changeset(origin="extract")`, `dedup.merge_paraphrases`. Declared entities are matched
   to resolved pages by name too (the model sometimes writes a bare id like `sam-okafor`), so
   grounding sees all their names. `ground_links` then drops a fact whose subject or object is a
   page the episode never names, whatever the predicate (ingest grounds page predicates only; a
   `met_at` once pointed at an unrelated city page). `plan_ops` compares each
   `add_fact` with memory by name: a restatement (current or ended) → `confirm_fact`; a clash
   (scoped like the writer, or the known fact the model named, or an ended fact with the same value
   and other dates) → `supersede_fact` carrying over unrestated dates and qualifiers.
7. **Owner corrections** (`decide_corrections`): for each supersession the writer would allow as a
   correction (`writer.correction_problem`: origin extract, about the owner, no relationship,
   nothing private, stated, same predicate), one typed choice "Does the owner tell that the stored
   fact is wrong or out of date?" (yes/no with descriptions) on the stored fact + the episode;
   p(yes) ≥ 0.9 → its fact id goes in `apply(..., corrections=…)`. The writer re-checks and applies
   it; everything else is queued.
8. **Apply** one change set per episode (source = the episode id, `created` = the episode start so
   journal lines carry the conversation day): facts, confirmations, supersessions, journal lines
   (≤ 2, importance of the episode), intentions (dated with `after::`, linked to resolved pages),
   and `update_episode` → `extracted`. A private episode marks every fact `privacy:: private`,
   new pages private, and adds no alias (`make_private`).
9. `refresh_generated` + `reindex` (unless `refresh=False`).

Idempotent: a session with episodes is not re-segmented; extracted episodes are skipped; a failed
episode stays `pending` and is resumed by the next run (`--pending`).

**Parts** (`turns=`: the turns of a session still running, read by the caller; the interview saves
each finished block this way, `refresh=False`): only the turns after the last
`extract | <transcript> | segmented turns A-B: …` line in log.md are segmented, and the part's
line is committed **with** its episodes (`apply(..., event=…)` in brain.vault.writer: one commit,
so a crash cannot leave episodes without their record or the reverse); a part with nothing
personal logs `segmented turns A-B: no episode (…)`. A tail without an owner turn waits for the
next part. Once a session has part lines, `segmented_through` is the end of the last one and
`is_processed` needs no owner turn after it and no pending episode; the run at close (no
`turns`) segments what is left. Sessions without part lines keep the old rule (episodes or a
"no episode" line = segmented as a whole). Only pending episodes ending within the given turns
are extracted. An interview transcript processed without `regions_hint` gets its frontmatter
`region` as the hint (`--pending`).

## Decisions and thresholds (see docs/decisions.md)

- Correction wording chosen on 20 labelled cases (`tests/fixtures/memory/corrections.yaml`):
  explicit ≥ 0.99, others ≤ 0.59. A yes/no predicate reached 0.9 on only 3 of 9 explicit cases.
  After changing the question or the cases: `BRAIN_RECORD_FIXTURES=1 uv run pytest -m llm
  tests/memory/test_memory_correction_calibration.py -s`; the offline test checks the recorded
  scores separate at `CORRECTION_P`.
- Name embeddings (instructed both sides): related 0.58–0.96, unrelated ≤ 0.48, two cities 0.73.

## Prompts and fixtures

- First comment line carries the version (`memory_segment vN`, `memory_extract vN`); bump on any
  change; keep the one-line minified JSON example last (output tokens are the cost).
- Replay fixture: `tests/fixtures/llm/memory/session/` (a fictional chat in `tests/memory/live.py`).
  Re-record after any prompt, schema or question change:
  `BRAIN_RECORD_FIXTURES=1 uv run pytest -m llm tests/memory/test_memory_live.py -s`, then
  `uv run pytest tests/memory`. The live and replay runs switch off embedding candidates and
  paraphrase merging (embeddings are not recorded). New fact ids are random: the replay masks them.
- The tests pin the life map (`tests/fixtures/memory/life-map.yaml`): the segmentation prompt
  lists the regions. Re-copy it from config/ and re-record when the life map changes.
- Read the recorded JSON before committing: fictional data only.

## Evaluation

`uv run brain eval memory [--only NAME] [--no-cache] [--json out.json] [--keep]`: each transcript
in `tests/eval/memory/sessions/` (gold inside, format in tests/eval/README.md) goes into a
fresh copy of the fixture vault. Metrics: episode count accuracy, region recall, fact recall and
precision, correct supersessions, queued recall, intention recall, **wrongly auto-applied (must be
0)**, duplicates, seconds. Run it twice with `--no-cache` (sampling noise is large) before and
after a change and diff the JSON files.

## Checklist

- [ ] `uv run pytest`, `uv run pytest -m llm tests/memory`, ruff clean.
- [ ] `brain eval memory --no-cache` twice: wrongly auto-applied 0 in both; read every
      `unverified` and `misses` entry.
- [ ] No new auto-apply path outside `writer.correction_problem`; anything private stays private.
- [ ] A decision-log entry for any non-obvious choice.
