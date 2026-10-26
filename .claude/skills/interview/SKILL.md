---
name: interview
description: How `brain interview` works — life-map coverage from the facts table, staleness, region choice and privacy, questions.md (generated gaps + items other components add), the plan, the one-question-at-a-time loop (decide labels → follow-up / next; grounded phrasing with a template fallback), skip/private/stop, and the hand-off to the memory pipeline. Use when changing src/brain/interview/, src/brain/prompts/interview_*.md, config/life-map.yaml, questions.md handling or tests/interview/.
---

# Interview mode

Spec: `docs/architecture.md` (Flow 2), `docs/vault-schema.md` (questions.md, privacy),
`config/life-map.yaml`, decision log entry "Interview mode: private regions only on request".

## Modules (`src/brain/interview/`)

| module | does | model? |
|---|---|---|
| `coverage.py` | `coverage_report(db, lifemap, today)`: per region/slot status `filled·partial·stale·empty`, per-entity regions per entity, unplaced people | no |
| `questions.py` | questions.md: `gap_items`, `write_questions`, `refresh_questions`, `add_question` (API for other components), `resolve_question` | no |
| `planner.py` | `choose_region` (privacy rule, boosts), `resolve_region`, `build_plan`, `context_lines` | no |
| `session.py` | `InterviewSession`: `opening()`, `reply(text)`, `close(process, db=)` | decide + structured |
| `cli.py` | `interview(region, --minutes, --answers, --coverage, --plan)`, `coverage_table`, `coverage_summary` | |

## Rules worth keeping

- Coverage reads only the index (`facts`, `entities`): run `reindex` first. Present slots
  (`stale_days`) count current facts valid today; timeless slots count history too.
  "Confirmed" = max(`rec::`, dates in `src::` such as `ep-YYYYMMDD-…`).
- Never choose a private region unless the owner named it. Private answers get a
  `[interview] privacy:: private` system turn before them in the transcript.
- Interview answers become facts **only** through `brain.memory.process_session(...,
  channel="interview", regions_hint=[region])` at close. The interview never writes pages.
- questions.md: coverage items (`origin:: coverage`) are rebuilt; everything else is kept. Field
  values never contain parentheses (the core card and the parser strip `(key:: value)` groups).
- Prompts: `interview_system` (+ core card) is the cached prefix; history follows; the PLAN
  block (`interview_turn`) goes on the latest user message only. History keeps plain answers.
- Every generated message passes `ungrounded()` (names/numbers must appear in what the model
  was given) and `one_question()`; otherwise `fallback()` uses the hint.

## Testing

- `uv run pytest tests/interview` (fake model, fake pipeline, fictional vault +
  `tests/fixtures/interview/wiki`).
- Replay: `test_interview_replay.py` checks the recorded requests still match. After changing
  a prompt or the plan, re-record:
  `BRAIN_RECORD_FIXTURES=1 uv run pytest -m llm tests/interview/test_interview_live.py`.
- By hand: `brain interview --coverage`, `brain interview projects --plan`,
  `brain interview current_job --answers answers.txt` (one answer per line; EOF closes).
- Judge first questions on the fictional vault: grounded (cites only known facts), one thing,
  natural. Keep live runs short: the model server is shared and serialises requests.
