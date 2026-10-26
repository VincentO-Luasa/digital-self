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
   - Each part becomes a `Chunk` whose heading is its section trail plus the model's title (the
     extraction prompt's "Section:" line). The trail is that of the part's **first heading**
     (`planning.section_trail`): a part often starts with a kicker line ("Career" above
     "## Career path") whose trail is still the previous section's, and that label once made a
     job read as part of the side project above it. A part without a heading continues its
     section; one that starts with real text of an earlier section keeps the shared trail only. `IngestReport.plan` (`PlanReport`) keeps
     the parts, notes, calls and seconds; the CLI prints it before extraction (`on_plan`).
   A chunk whose reply is truncated is still halved (twice at most).
3. **Brief** (`brief.make_brief`, one structured call per document, after the plan): prompt
   `prompts/ingest_brief.md`, schema `Brief` (`reasoning`, subject, kind, voice, `periods` =
   jobs/studies, `homes` = cities lived in, `names` = ≤ 12 glossary entries with type, aliases,
   parent and the subject's role), temperature 0. It reads the whole text up to
   `FULL_TEXT_TOKENS` (6k), else each part's heading and first ~150 tokens. Code then cleans it
   (`clean_brief`: descriptions are not aliases, the subject is not a name) and **grounds** it
   (`ground`): a date is kept only if a sentence or line names its city / organisation / title
   and gives that year, in any form `dates.years_in` reads (2021, Sep 21, Sept. '21, 09/2021,
   09/21, 2021-09, ’21; not "May 21, 2024"'s day). `brief.units`: wrapped prose is joined into
   sentences, but a line ending with a date (no full stop) or starting with a date range is a
   CV/timeline line, never joined to its neighbours, and a record line (`_record`) is one unit,
   never cut into sentences (a CSV row "Org,Title,"Did X. Did Y.",Jan 2024," kept its date); a home needs a sentence naming the city with a residence word (lives, moved,
   based…), else it is dropped (the model takes an employer's city, or a job's start, for a
   move). Every drop is a `brief:` note. A failed call → `brief=None`, ingest goes on.
   Where it is used is `pipeline.BRIEF_USES` (default `{"normalise"}`). Measured (two extraction
   samples each; docs/decisions.md): in **normalisation** +4 required, +14 optional, +3
   precision, +9 pages; as a block in every **extraction** prompt it lost against that (required
   −4 to −7, precision −2 to −6, ~+60% time: longer replies, one part truncated and halved; with
   dated periods the extractor copied the brief's rounded years over the excerpt's months, so
   `render` shows periods undated); the brief's header as **checks** context changed nothing.
   `render(brief)` + `extract.build_messages(brief=…)` (first in the user message: a cached
   prefix across parts) and `header` stay as switches. Cost 12–17 s per document (~1.1–1.6k
   prompt, ~410–600 output tokens), ~+25% on an uncached ingest.
4. **Gate** (`decisions.gate`, only when there are 2+ parts; a document's only part is never
   skipped): one yes/no decision on a compact view of the part (headings + first 24 words of each
   block): "does it say anything about its author or the person it describes?". Under
   `SKIP_BELOW` (0.15) the part is skipped (`IngestReport.skipped`, CLI line, a note with p).
   Measured: CV/hobbies/opinion ≥ 0.99, contact block ~0.6, cookie notice / tutorial / footer
   0.03–0.12; ~0.6–0.9 s per part against ~40 s of extraction.
5. **Extract**: `extract.extract_chunk`, one structured call per chunk. System prompt
   `prompts/ingest_extract.md` (byte-stable, cached), then the user message with the date,
   document title, section, known entities named in the chunk (cap 40) and the excerpt. Flat
   schema `Extraction`: names, not ids; predicates are an enum (`INGEST_PREDICATES`).
6. **Check** (`decisions.check_part`, after each part's extraction):
   - **Grounding** (code): a fact whose subject (unless the owner) or page-type object
     (`OBJECT_TYPES` predicates) the part never names (name, alias or a capitalised 4+ letter
     word of it, line breaks ignored) is dropped. A weak part (tutorial + contact block) once
     restated every KNOWN ENTITY as works_at / worked_on / friend_of.
   - Then typed decisions on what code cannot verify, shown the evidence (the sentences / list
     lines naming the items, under their heading), never the extractor's answer. Each set is
     **screened** ("which of these is a team, tool, topic or activity rather than a person, org,
     project, place or event? / none of them") in groups of ≤ `SCREEN_GROUP` (6); items at
     ≥ `SCREEN_FLAG` (0.15), items the part never names, and sets of ≤ `DIRECT_UP_TO` (2) get
     their own question:
     - new entities → kind (person / org / team / project / place / event / tool / topic /
       activity); team, tool, topic, activity → a literal value (`entity_kinds` → normaliser
       literals); another page kind → the entity type.
     - works_at organisations → relation (employer / school / member / other): studied_at,
       member_of, or dropped.
     - has_skill values (screened for hobbies/interests) and interest/habit/likes values
       (screened for work skills) → "a skill, a hobby, a subject of interest, or none of these?";
       skill → has_skill, hobby/interest listed as a skill → interest, other → kept.
     - `speaks` values outside `LANGUAGES` → "is X a language?"; a confident no drops the fact.
     - **story details** (`_check_stories`, before the skill routing): the owner's
       responsibility / has_skill / interest / habit / likes facts (not languages) are screened as
       statements (the fact's text, else "predicate: value"; `SCREEN_STORY`, no "not named"
       shortcut: statements are paraphrases) and a flagged one is asked "What does this
       statement about the person record?" job / trait / event / lesson; p(event) + p(lesson)
       ≥ 0.8 drops the fact (`STORY_ACTION`: "drop", "note" or "off"). A write-up's "Came out of
       the experiment less sceptical about agents", "Had best holiday in Lisbon" go; CV
       achievements ("Redesigned the booking flow; drop-off fell…") stay. ~1.2 s per part.
       Goals, values and team facts are not asked. Wording chosen on 29 labelled statements
       (docs/decisions.md).
   A change needs `confidence ≥ ACT_ABOVE` (0.8); otherwise the extractor's choice stays and a
   note says so. Every screen and answer is a `check:` note (first in `Normalised.notes`).
   Why screens: the runtime answers questions one at a time and re-reads the input for each
   (~0.35 s + ~1.6 ms/token; a shared input is not reused on this hybrid model), so a question
   per item cost 8–14 s per CV part; screens bring it to ~2–6 s.
   `context=` on `gate`/`check_part` prepends shared context (the brief's `header`, when
   `checks` is in `BRIEF_USES`; measured no gain, off by default).
7. **Normalise**: `normalise.build_changeset` resolves names to ids, normalises dates, merges
   duplicates, applies the guards below, renders fact text and builds one
   `ChangeSet(origin="ingest")`. Everything it changed or dropped goes to `Normalised.notes`.
8. **Apply**: `vault.writer.apply` (policy decides what is queued for review), then
   `refresh_generated` (index.md, core.md) and `reindex`.

## Adding a reader

- Add `readers/<type>.py` with `read_<type>(data: bytes | str) -> (title, markdown)`. Keep
  headings as `#` lines: chunking and the prompt's "Section:" depend on them.
- Register its suffixes in `readers.SUFFIXES`, dispatch in `readers.read_bytes`; for the web,
  map its content type in `web.CONTENT_TYPES`.
- Drop boilerplate in the reader, not in the prompt (navigation, cookie banners, contact
  details): every input token costs prefill and invites junk facts. What the reader cannot
  recognise structurally is left to the gate.
- Rows (`html.join_rows`): an innermost repeated block (div / li / article with a sibling of
  the same tag and classes) holding 2–8 short text cells (≤ 80 words, ≤ 1 heading, no h1)
  becomes one list item, cells joined by " · " ("- Sep 21 — Feb 23 · Northwind Labs · Data
  Engineer. Built …"); `<dl>` term + definitions likewise. Read cell by cell, a grid timeline
  puts each date on the line above the name it belongs to. A series where any cell is a write-up
  (> `CELL_LONG` 30 words: project cards) is left alone; sections, tables and lone blocks are
  never rows.
- Keep fixes structural, never per site: the HTML reader puts a space between adjacent inline
  elements whose texts would touch (`<span>02</span><span>Lab</span>` read "02Lab", which the
  model took for a name) and drops decorative ordinals (a bare 1–2 digit element right before a
  heading, first in a heading, or first in a short line followed by another element when
  zero-padded or above a heading). Counters (`data-target`) and counts in sentences stay.
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
- The brief prompt (`ingest_brief vN`) follows the same rules. Its unit tests script a `Brief`
  (`conftest.brief`, `homepage_brief`); a scripted ingest needs one right after the plan (or
  first, for a one-part document).
- Recorded fixtures (`tests/fixtures/llm/ingest/`): the homepage as numbered exchanges in call
  order (the plan, the brief, then per part the gate, the extraction and the check decisions);
  `portfolio/` the same for `alex-rivera-portfolio.html`; `gate/` the gate on the fictional
  parts in `conftest.GATE_PARTS`; `split/` the long page's plan. `FakeLLM.from_fixtures`
  replays structured and decide exchanges in order; the replay tests compare the requests.
- After any prompt, schema, outline, reader or decision-question change, re-record and re-run:
  `BRAIN_RECORD_FIXTURES=1 uv run pytest -m llm tests/ingest/test_ingest_live.py tests/ingest/test_ingest_split_live.py tests/ingest/test_ingest_gate_live.py -s`, then
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
- From the brief (`build_changeset(brief=…)`, every change a `brief:` note):
  - names: an extracted entity named like a brief name or alias, or a brief name followed by
    lowercase words or a version ("Atlas Migration project", "Atlas 2.0"; `brief.lookup`), and of
    the same type, becomes that entity: one page across parts. The brief's page-type names (and
    the parents it gives) are planned before facts are resolved, so a city given as a value
    links to its page; unused ones are dropped silently.
  - structure (`_brief_structure`): an entity a fact uses gets `part_of` / `located_in` its
    brief parent (`STRUCTURE`: project → org/project, org/event/place → place; never a side
    project) unless a fact already says; a parent the brief does not list is typed from the
    child (`PARENT_TYPE`); a located_in/part_of stated the wrong way round is turned; an org with
    role "member" gets `member_of` (dated from a fact naming it) when no fact links the owner to
    it (not one of the document's team values); a person's `knows` takes the brief's friend /
    family / colleague, and `friend_of` becomes `family_of` when the brief says family.
  - homes (`_brief_homes`): a lives_in for a city the owner later left gets the brief's end; a
    missing home is added when its period is known; never the hometown, never an end on the
    last home.
  - stints: `_merge` keeps two dated facts with the same object apart when there is more than a
    month between them and the brief shows two periods at that organisation (an internship
    years before the job); without the brief they merge as before.

## Quality checklist (after any change)

- [ ] `uv run pytest -q` and `uv run pytest -m llm` green; `uv run ruff check . && uv run ruff format --check .`
- [ ] Live test output: seconds and output tokens per chunk no worse than before; read every note.
- [ ] Run `brain eval ingest` before and after any ingest change; keep the JSON of both runs
      (`--json before.json` / `after.json`) and diff misses and extras (tests/eval/README.md).
- [ ] Split plan: each job/project stays in one part, long heading-free stretches are split by
      the model (no "split into" note), no part ends with a heading, every block is covered.
- [ ] Smoke run on a scratch vault, never the real one:
      `BRAIN_VAULT=<tmp> uv run brain init-vault && uv run brain ingest tests/fixtures/docs/alex-rivera.html && uv run brain review --list`
      and the same with `alex-rivera-portfolio.html` (gate skips the tutorial; checks turn the
      team and topic into values; `portfolio_problems` in tests/ingest/conftest.py).
- [ ] Brief: read every `brief:` note (dropped dates and homes, merged names, added structure);
      the facts it adds name only entities the parts use, memberships and homes it states.
- [ ] Gate and checks: read every `gate:`/`check:` note; decision seconds per part (CLI chunk
      line) stay a small share of the extraction time.
- [ ] No pages for tools, products, ids or teams; works_at points at orgs only.
- [ ] Duties are `responsibility` facts, not merged into the employer fact.
- [ ] Self fact text has no pronoun or owner name; numbers are real values, never 0 placeholders.
- [ ] One current `lives_in`; past places have an end or are notes; nothing queued but relations
      and genuine contradictions.
- [ ] A decision-log entry in `docs/decisions.md` for any non-obvious choice.
