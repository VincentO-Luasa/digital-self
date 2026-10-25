---
name: ingest
description: How `brain ingest` turns a file or web page into vault facts — readers (html/pdf/md/txt/web), chunking, the pinned extraction prompt, normalisation guards, review policy — plus how to add a reader, version the prompt, re-record fixtures and check quality. Use when changing anything under src/brain/ingest/, src/brain/prompts/ingest_extract.md, the `ingest` CLI command or tests/ingest/.
---

# Ingest

## Flow (src/brain/ingest/)

1. **Read**: `readers.load_document(path)` picks a reader by suffix and returns
   `Document(path, format, title, text)` with markdown text. A URL goes through
   `readers/web.py`: `fetch` (http/https only, browser User-Agent, 20 s timeout, 5 MB cap) →
   `vault.raw.save_doc_bytes` (raw copy, committed with the URL in the message) → read from the
   raw copy like any file. `--follow N` takes the first N same-origin links of the page
   (`same_origin_links`), each ingested as its own raw doc and change set.
2. **Split**: `planning.plan_chunks`. A document that fits one ~500-token chunk (`CHUNK_TOKENS`;
   the extraction reply is ~8x the excerpt) goes as is. A larger one is split by the model:
   - `chunking.split_blocks` cuts the markdown into ordered blocks (a heading, a paragraph, a
     list, a table, a fenced code block; a block over the cap is pre-split by paragraph/line).
   - The model sees an **outline**, not the text: `[n] # Heading` or `[n] (size tok) first 16
     words …`, plus a list of the heading-free stretches over the cap (`long_stretches`). Without
     that list the model never split inside a headed section, even with sizes shown or an
     "OVER CAP" flag. An outline over `WINDOW_TOKENS` (6k) is planned in windows sharing 8
     blocks; each window keeps the starts up to the middle of the overlaps.
   - One call per window: prompt `prompts/ingest_split.md`, schema `SplitPlan` (`reasoning`,
     then `parts: [{first_block, title}]`), temperature 0 (at 0.2 it sometimes kept a long
     stretch whole).
   - Code repairs the plan and notes each repair: out-of-range starts dropped (most out of
     range = unusable), sorted, deduplicated, a start at the window's first block added; a
     start moves back onto the heading(s) right before it and a headings-only part joins the
     next; a part over the cap is split at headings, then paragraphs (`pack_blocks`, no second
     call); a part under a quarter of the cap joins a neighbour when both fit.
   - A failed or unusable plan falls back to `chunk_markdown` (heading-bounded packing), or,
     for one window of several, to heading packing of that window.
   - Each part becomes a `Chunk` whose heading is the headings enclosing it plus the model's
     title (the extraction prompt's "Section:" line). `IngestReport.plan` (`PlanReport`) keeps
     the parts, notes, calls and seconds; the CLI prints it before extraction (`on_plan`).
   A chunk whose reply is truncated is still halved (twice at most).
3. **Extract**: `extract.extract_chunk`, one structured call per chunk. System prompt
   `prompts/ingest_extract.md` (byte-stable, cached), then the user message with the date,
   document title, section, known entities named in the chunk (cap 40) and the excerpt. Flat
   schema `Extraction`: names, not ids; predicates are an enum (`INGEST_PREDICATES`).
4. **Normalise**: `normalise.build_changeset` resolves names to ids, normalises dates, merges
   duplicates, applies the guards below, renders fact text and builds one
   `ChangeSet(origin="ingest")`. Everything it changed or dropped goes to `Normalised.notes`.
5. **Apply**: `vault.writer.apply` (policy decides what is queued for review), then
   `refresh_generated` (index.md, core.md) and `reindex`.

## Adding a reader

- Add `readers/<type>.py` with `read_<type>(data: bytes | str) -> (title, markdown)`. Keep
  headings as `#` lines: chunking and the prompt's "Section:" depend on them.
- Register its suffixes in `readers.SUFFIXES`, dispatch in `readers.read_bytes`; for the web,
  map its content type in `web.CONTENT_TYPES`.
- Drop boilerplate in the reader, not in the prompt (navigation, cookie banners, contact
  details): every input token costs prefill and invites junk facts.
- Test it on a fictional fixture in `tests/fixtures/docs/` (Alex Rivera persona only), no network
  (`httpx.MockTransport` for web code).

## Prompt versioning

- The prompt's first comment line carries its version (`ingest_extract vN`); bump it on any change.
- Keep it short. Measured: minified JSON (the closing line plus the one-line shape example)
  cut output from ~4k to ~2.5k tokens on the fixture page (decode is ~42 tok/s, so output tokens
  are almost all of the time). The model indents JSON unless that example is there.
- `text` may be `""`: code renders it from `normalise.TEMPLATES`, which saves output tokens.
- The split prompt (`ingest_split vN`) follows the same rules: version line, one-line JSON
  example last. Its fixture is `tests/fixtures/llm/ingest/split/` (the long fictional page
  `tests/fixtures/docs/alex-rivera-long.md`), recorded by `tests/ingest/test_ingest_split_live.py`.
  The homepage (654 tokens) is over one chunk, so its fixtures start with a plan call too.
- After any prompt, schema, outline or reader change, re-record and re-run:
  `BRAIN_RECORD_FIXTURES=1 uv run pytest -m llm tests/ingest/test_ingest_live.py tests/ingest/test_ingest_split_live.py -s`, then
  `uv run pytest tests/ingest tests/cli`. The replay test fails while the pinned request differs.
  Check the recorded JSON contains only fictional data before committing.

## Normalisation guards (normalise.py)

The model is ~3B active and will slip; code repairs or drops, and notes every repair.

- Types: `OBJECT_TYPES` (works_at/studied_at/member_of → org, lives_in/hometown/located_in →
  place, worked_on → project, relations → the owner) and `_subject_ok` (relations from a person,
  project predicates on a project). Repairs: works_at a project → worked_on; verb-led works_at
  ("Works on…", "Experiments with…") → responsibility; works_at/member_of a team → team; uses on
  the owner → has_skill; an entity where a literal belongs → the literal. Otherwise dropped.
- Literal values, never pages: entities typed `tool` (tools, apps, products, datasets), names
  that look like teams or record ids (`literal_reason`). Facts about them are dropped.
- Zero counters ("Has 0 years of …") are dropped.
- Past tense without an end on lives_in/role/team/manager becomes a `note` (no guessed end).
- Text: a pronoun, the owner's name (self facts), no object, or over 160 chars → template.

## Quality checklist (after any change)

- [ ] `uv run pytest -q` and `uv run pytest -m llm` green; `uv run ruff check . && uv run ruff format --check .`
- [ ] Live test output: seconds and output tokens per chunk no worse than before; read every note.
- [ ] Split plan: each job/project stays in one part, long heading-free stretches are split by
      the model (no "split into" note), no part ends with a heading, every block is covered.
- [ ] Smoke run on a scratch vault, never the real one:
      `BRAIN_VAULT=<tmp> uv run brain init-vault && uv run brain ingest tests/fixtures/docs/alex-rivera.html && uv run brain review --list`
- [ ] No pages for tools, products, ids or teams; works_at points at orgs only.
- [ ] Duties are `responsibility` facts, not merged into the employer fact.
- [ ] Self fact text has no pronoun or owner name; numbers are real values, never 0 placeholders.
- [ ] One current `lives_in`; past places have an end or are notes; nothing queued but relations
      and genuine contradictions.
- [ ] A decision-log entry in `docs/decisions.md` for any non-obvious choice.
