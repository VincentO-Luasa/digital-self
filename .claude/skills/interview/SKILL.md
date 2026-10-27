---
name: interview
description: How `brain interview` works — life-map coverage from the facts table, staleness, region choice and privacy, questions.md (generated gaps + items other components add, lint doubts with `doubt::`), the plan (lint doubts grouped per entity, budgeted, interleaved), the one-question-at-a-time loop (decide labels → follow-up / next; answer linking; asking back; grounded phrasing and provenance checks with a template fallback; soft time limit), skip/private/stop, settling lint doubts from answers, and the hand-off to the memory pipeline. Use when changing src/brain/interview/, src/brain/prompts/interview_*.md, config/life-map.yaml, questions.md handling or tests/interview/.
---

# Interview mode

Spec: `docs/architecture.md` (Flow 2), `docs/vault-schema.md` (questions.md, privacy),
`config/life-map.yaml`, decision log entry "Interview mode: private regions only on request".

## Modules (`src/brain/interview/`)

| module | does | model? |
|---|---|---|
| `coverage.py` | `coverage_report(db, lifemap, today)`: per region/slot status `filled·partial·stale·empty`, per-entity regions per entity, unplaced people | no |
| `questions.py` | questions.md: `gap_items`, `write_questions`, `refresh_questions`, `add_question` (API for other components; `doubt=` the lint key), `resolve_question`, `doubt_key` | no |
| `doubts.py` | `Doubt` (a lint key parsed), `EntityInfo` / `entity_infos` (title, one line, sources, employers ranked by evidence), `combined_hint`, `merge_hint` | no |
| `planner.py` | `choose_region` (privacy rule, boosts), `resolve_region`, `build_plan` (doubt blocks per entity, `MAX_DOUBT_BLOCKS`, interleaving), `context_lines` | no |
| `session.py` | `InterviewSession`: `opening()`, `reply(text)`, `close(process, db=)`; answer linking (`link`, `status`), asking back (`_context`), ending on content (`is_dry`, `_dry_end`), soft time limit, block boundaries for the saver | decide + structured |
| `saver.py` | saving as it goes: `Gate` (busy while the interview handles an answer; `wait(grace, latest)`, `interrupt_event`, `release` at close), `Yielding` (the worker's model client: every call waits for `SAVE_GRACE` s of owner idleness, long ones only up to `SAVE_LONG_LATEST` s into a pause, and structured calls are streamed and cancelled when the owner answers, then repeated, at most `MAX_INTERRUPTS` (5) times), `BlockSaver` (one worker thread, jobs in order; `drain(cancel=True)` returns the jobs never started) | via the pipeline |
| `settle.py` | `Settler.settle_step`: maps answers onto a doubt's options (typed choice, "does not say" first) → apply (p ≥ 0.9), review (≥ 0.5) or leave open | decide |
| `cli.py` | `interview(region, --minutes, --answers, --coverage, --plan)`, `coverage_table`, `coverage_summary` | |

## Rules worth keeping

- Coverage reads only the index (`facts`, `entities`): run `reindex` first. Present slots
  (`stale_days`) count current facts valid today; timeless slots count history too.
  "Confirmed" = max(`rec::`, dates in `src::` such as `ep-YYYYMMDD-…`).
- Never choose a private region unless the owner named it. Private answers get a
  `[interview] privacy:: private` system turn before them in the transcript.
- Interview answers become facts **only** through `brain.memory.process_session(...,
  channel="interview", regions_hint=[region])`: per finished block during the interview
  (`turns=` the turns so far, `refresh=False`) and for the rest at close. The interview never
  writes pages.
- questions.md: coverage items (`origin:: coverage`) are rebuilt; everything else is kept. Field
  values never contain parentheses (the core card and the parser strip `(key:: value)` groups).
- Prompts: `interview_system` (+ core card) is the cached prefix; history follows; the PLAN
  block (`interview_turn`) goes on the latest user message only. History keeps plain answers.
- Every generated message passes `invented_claim()` ("you mentioned/listed X" must be backed by the
  owner's answers), `ungrounded()` (names/numbers must appear in what the model was given) and
  `one_question()`; otherwise `fallback()` uses the hint.
- Per answer: `classify` (answered / partial / declined = nothing to add / refused = would rather
  not say, counted only at p ≥ 0.9 / unclear / asked back), then one `link` decision before the
  next step or a generated follow-up: "already answered?" (p ≥ 0.6 skips) and, only after a
  refusal and for questions sharing a word with a refused one, "asks again for what was
  refused?" (p ≥ 0.35 skips). The link input is an append-only window of the session (cached
  prefix) plus older exchanges naming the step's entities after it. Each decide question costs
  ~0.5 s uncached; keep candidates few.
- Asking back gets the step's `brief` (one-liners + sources, from `EntityInfo`) or its known
  facts, then the same question; a second ask-back moves on. No model call.
- Lint doubts: at most `MAX_DOUBT_BLOCKS` (4) entities, only in the first `DOUBT_SHARE` (40%) of
  the time; pages with a type or merge doubt get no coverage steps that session. The part_of
  question names at most two employers with evidence (`lint.edges.likely_jobs/named_jobs`); the
  settle choice offers every employer.
- Settling (`settle.py`, at close, before the memory pipeline): only from non-private answers
  given under the step (or the exchanges naming its entity when linking skipped it). Kind
  choices describe what the owner says (`OWNER_KINDS`): with lint's descriptions "stream
  processing", called a project by the owner, read as a topic. p ≥ 0.9 applies (origin `user`,
  source = the transcript), 0.5–0.9 queues a review, below or "does not say" leaves the item open.
- Time is soft: at `minutes`, between blocks, "We've covered … Keep going, or stop here?"; a clear
  yes adds `minutes × EXTEND`, anything else stops.
- Ending on content: an answer is dry (`is_dry`) when labelled declined / refused / unclear, a
  `skip`, or partial and made only of `FILLER` words ("not much really", "same as before"); a
  dry partial gets no follow-up. `MAX_DECLINES` (3) dry in a row, or `DRY_WINDOW` (4 of the last
  5), end the session ("I think I've got what I need on …"; "Let's leave … there" when they were
  declines or skips); `DRY_BLOCK` (2) dry in a row inside an entity block skip the rest of the
  block. `Reply.next_region` (least covered non-private other region) on end / content /
  declined; the CLI prints "Next time: `brain interview <region>`". Measured on scripted
  owners: a talkative one is never stopped, a terse one after 6–7 answers (docs/decisions.md).
- Saving as it goes: at a block boundary (`_block(item)` changes; a step without a block is its
  own block) with ≥ `SAVE_MIN_ANSWERS` (3) answers since the last save, the session reads its
  transcript (it is the file's only writer), builds the settle inputs of the finished steps and
  submits a `Job`. The worker settles those doubts, then runs the pipeline on the turns so far.
  It is the only vault writer while the interview runs. `close()` releases the gate, lets the
  job under way finish at full speed, drops the queued ones and one that never got a pause for
  its first call (`Cancelled`), then settles their doubts and processes everything not saved in
  one run, and merges the reports (`merge_reports`: episodes by id, the close's seconds). A pipeline without a `turns` parameter (old fakes) disables saving. `abandon()` on
  an error lets the running job finish; `--pending` does the rest. Inline saving
  (`background=False`, tests) uses the unwrapped client: the gate is busy on that same thread.
- The card's "To ask" lines are removed from the interview's profile (they pulled follow-ups
  towards plan questions).

## Testing

- `uv run pytest tests/interview` (fake model, fake pipeline, fictional vault +
  `tests/fixtures/interview/wiki`).
- Replay: `test_interview_replay.py` checks the recorded requests still match. After changing
  a prompt or the plan, re-record (delete the old files first; `SeqRecorder` numbers them in
  call order):
  `rm tests/fixtures/interview/llm/*.json; BRAIN_RECORD_FIXTURES=1 uv run pytest -m llm tests/interview/test_interview_live.py`.
- Lint doubts: `test_interview_doubts.py` (scripted `ScriptedLLM`: classification from
  `decisions`, linking from `link=`, settling from `decider=`), scenario in `doubts_scenario.py`
  (fictional vault: stream processing, ingestion, fraud alerts / transaction monitoring).
- End to end against the model, with a scripted owner: `uv run pytest -m llm
  tests/interview/test_interview_doubts_live.py -s` (`BRAIN_INTERVIEW_SAMPLES=3`,
  `BRAIN_INTERVIEW_MEMORY=1` to run the memory pipeline too). Targets: repeats 0, declined
  re-asked 0, wrong changes 0 (asserted); read the transcripts.
- By hand: `brain interview --coverage`, `brain interview projects --plan`,
  `brain interview current_job --answers answers.txt` (one answer per line; EOF closes).
- Ending and saving: `test_interview_stopping.py` (scripted labels: talkative not stopped, terse
  stopped, dry block, window, skips), `test_interview_saving.py` (the real pipeline with a second
  `FakeLLM`: block saved during the interview, close does only the rest, crash → `--pending`,
  failed part retried at close, private answer in a saved block, background thread, gate).
  Live: `uv run pytest -m llm tests/interview/test_interview_saving_live.py -s`
  (`BRAIN_SAVING_SAMPLES`, `BRAIN_SAVING_MODES=close,fast,typing`, `BRAIN_SAVING_TYPING=10,20`):
  per-turn latency, close time, terse vs talkative stops. Long: ~50 min for the default.
- Judge first questions on the fictional vault: grounded (cites only known facts), one thing,
  natural. Keep live runs short: the model server is shared and serialises requests.
