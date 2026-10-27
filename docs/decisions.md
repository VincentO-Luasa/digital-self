# Decision log

Short entries, newest last. Format: date — decision — why — alternatives considered.

## 2026-10-25 — Markdown vault is the source of truth; SQLite is a derived index
Why: at personal scale (thousands of notes) file + grep/FTS agents match graph-DB memory on
LoCoMo (Letta); markdown is readable, diffable and editable by the owner, which matters for a
self-model holding "flaws"; Karpathy's LLM-Wiki pattern; any index can be rebuilt after a model or
schema change. Alternatives: Graphiti on FalkorDB, Cognee, Mem0 (dropped graph memory in v2),
Hindsight. Kept as later options as *derived* indexes if multi-hop/temporal evals fail.

## 2026-10-25 — Borrow graph-memory semantics into markdown
Bi-temporal fact lines (`from/to` + `rec`), supersede-never-delete, mandatory `src::` provenance,
`aliases:` for entity resolution, stated/observed/inferred split. Why: these are what the memory
literature shows matters (Zep/Graphiti, StateMemBench, LongMemEval), independent of storage.

## 2026-10-25 — Local model performs all memory writes; the big model only proposes
Why: the big model's seat may be organisation-administered (the author's is a Claude Enterprise
seat, whose transcripts are retained and reachable via the Compliance API); automated nightly use
of a seat conflicts with "ordinary use" terms; Qwen3.6-35B-A3B is close to API models on extraction benchmarks (Hindsight retain board).

## 2026-10-25 — Runtime: oMLX, model Qwen3.6-35B-A3B (MLX 4-bit), non-thinking
Why: oMLX serves chat + embeddings + rerank in one process with enforced JSON schema and a
persistent prefix cache; MoE with ~3B active params is the only class fast enough on a base M4
(~45 tok/s vs ~7 tok/s for dense 27B). Fallbacks: LM Studio (runtime), Gemma 4 26B-A4B (model).
The client is OpenAI-compatible only, so swapping is configuration.

## 2026-10-25 — Escalation by rules + retrieval coverage + categorical answer labels
Why: small models' verbalised confidence is unreliable; learned routers need labelled data we do
not have yet. The routing log collects that data for a later learned router.

## 2026-10-25 — Public repo, private vault
Why: the owner wants to share how the brain is built. Code/docs/fictional fixtures only in the
repo; vault outside the repo with a local-only git; pre-commit hook with path rules, a
vault-generated denylist, and gitleaks.

## 2026-10-25 — Interview mode with topic-chunked episodes
Why: populating the graph should be a conversation, not an autobiography. Coverage over a life map
(regions × slots) picks what to ask; each conversation is segmented into topic episodes so
extraction is small, focused and reviewable.

## 2026-10-25 — Human-memory-inspired retrieval (proposed)
Schema card always in context (impression + current situation + marking facts), a 4-slot
working memory, cue-driven spreading activation (Personalized PageRank) combined with ACT-R
base-level activation, similarity and importance, gist-first packing, and gated proactive
surfacing. No LLM call in the hot path. Why: lets a ~3B-active model work from a small, ranked,
pre-digested context. HippoRAG 2, Synapse and EM-LLM report gains from the same mechanisms.
Weights are priors to be fitted on the personal eval. Details in `docs/retrieval.md`.

## 2026-10-25 — Big model is pluggable (Claude in the author's setup)
The local model escalates to "the big model", chosen by an `[escalation]` provider in the config:
`claude-code` (Claude Code in the vault workspace, the author's choice), `clipboard` (any chat UI,
reply brought back with `brain capture`), or `openai-compatible` (OpenAI, Moonshot/Kimi,
OpenRouter, a self-hosted server). Big-model transcripts live in `raw/external/<provider>/` and
their episodes use `channel: external` + `provider:`. The vault's runtime instructions are
`AGENTS.md`; `CLAUDE.md` only imports it and adds Claude-Code-specific notes.
Why: the owner wants the project usable with OpenAI, Kimi K3 and other big models, not only
Claude. The integration points are already cross-vendor conventions: MCP is an open protocol many
clients speak, `AGENTS.md` is read by many coding agents, and skills use the open Agent Skills
`SKILL.md` format. Privacy rules (confirmed packets, no private content over MCP) apply to every
provider, since any hosted provider may retain transcripts. Earlier entries' "Claude" wording was
generalised to "the big model".
Alternatives: Claude-only (simpler, but ties a public project to one vendor); a multi-provider SDK
or agent framework (conflicts with the explicit-control-flow rule; the OpenAI-compatible API
already covers most providers).

## 2026-10-25 — Ingest: flat extraction schema, resolution and policy in code
One structured call per heading-bounded chunk (~1.2k tokens). The model fills a flat schema
(`reasoning`, `owner_names`, `entities[{name,type,id,aliases,summary}]`,
`facts[{subject,predicate,object,text,start,end}]`) with names, not ids; the predicate is an enum
over a document subset of the vocabulary (no traits, flaws, health). Code resolves names to ids
(known aliases → entities named earlier in the document → new slug ids), normalises dates,
merges duplicates (union of their periods), turns `(self, owner|relation, X)` around to
`(X, p, self)`, keeps tools/skills/languages as literals, drops entities no fact uses, derives
new pages' regions (hence privacy) from the facts that link them, and adds the owner's names as
aliases of `self/profile`. One `ChangeSet(origin="ingest")` per document; the writer's policy
decides what is queued. The prompt lists only known entities named in the chunk (cap 40) so it
scales with the vault. Why: a ~3B-active model is reliable at "list the facts", not at producing
a discriminated union of operations with ids; everything it returns is a proposal checked in code.
Alternatives: emitting `Operation` JSON directly (likely more invalid ids and wrong directions;
the first live runs already showed wrong directions); sending the whole alias index (does not scale).

## 2026-10-25 — Documents the owner wrote yield `stated` facts
`brain ingest` treats a document as the owner's own (CV, homepage) and records `kind: stated`;
`--observed` marks documents written by others. Why: a first-person homepage is the owner
speaking, and `stated` facts with no contradiction auto-apply. The schema's "observed (in a
document)" now means documents by third parties.

## 2026-10-25 — Phase 1 core.md and index.md are deterministic
`core.md` (L0 card) is built in code from current self facts valid today plus the entities tied
to the owner by a current fact, ranked by a salience prior (predicate weight, stated > observed >
inferred, newest first on ties) and cut at `core_max_tokens`. No timestamps, so the file is
byte-stable and the chat prefix cache survives until a contributing fact changes. Both files
are regenerated and committed after ingest and review approvals, and include private pages
(they are local; the big model's instructions forbid reading them). `build_core(...,
include_private=False)` exists for the MCP profile tool. Phase 4 consolidation replaces the card
builder (impression, marking facts, open loops).

## 2026-10-25 — Raw capture: exclusive files, turn-by-turn transcripts
`raw/docs/YYYY-MM-DD-<slug>.<ext>` and `raw/conversations/YYYY/MM/conv-YYYYMMDD-HHMM.md` are
created with exclusive opens (a suffix `-2` on collision), never overwritten; re-ingesting an
identical document reuses its copy. Transcript turns are `## <n> · <role> · <timestamp>` blocks
appended and fsynced one by one (lines that look like turn headings are escaped), committed at
the end of the session. Why: raw/ is episodic ground truth; a crash must lose nothing, and
Phase 2 segmentation needs stable turn numbers for episode `turns: [a, b]`.

## 2026-10-25 — Chat prompt: memory in the latest user message, history trimmed in blocks
System prompt + core card form the cached prefix. History holds the plain user text (never the
memory block) and drops the oldest 4 exchanges once it exceeds 8. The per-turn block (today's
date + search hits packed gist-first under `retrieval.budget_tokens`, skipping facts already in
the card) is prepended to the latest user message. Token budgets use a chars/3.5 estimate (no
tokenizer dependency; errs on the safe side).

## 2026-10-25 — core.md is a friend's profile card, not a ranked fact dump
The L0 card is now laid out the way people hold a close friend in mind: an identity line, then
Who they are (values, strengths, flaws, habits; "Not known yet: …" for missing aspects), Now
(active projects, responsibilities, goals, recent important journal lines, facts new in the last
90 days), Life story (one line per lifetime period: job + roles + city, studies + degree), People
who matter (≤6, inner circle first), Interests and skills (comma lists with "+n more") and Open
loops (deadlines within 30 days, undone intentions, top `questions.md` items). Still deterministic
and byte-stable; near-duplicates (token Jaccard ≥ 0.75) merge and pool their sources; each
section has a share of a ~600-token target and spare budget flows to the others; `core_max_tokens`
stays the hard cap. Why: the flat builder gave one line per skill, tool and side project (~60
lines on a real homepage; 64 lines / ~600 tokens on a 68-fact synthetic vault, ~1k tokens at 128
facts), which costs prefill and dilutes a small model's attention; the new card is 20 lines /
~440 tokens on both. Grounded in McAdams (1995) three levels, Conway & Pleydell-Pearce (2000)
lifetime periods and working self, trait summaries stored apart from episodes (Park 1986; Klein,
Loftus & Kihlstrom), and Dunbar's inner layers (docs/retrieval.md, "L0: schema card").
Alternatives: an LLM-written profile now (non-deterministic, breaks byte stability; kept for the
Phase 4 impression paragraph); a fixed per-predicate line cap without sections (still a list).

## 2026-10-25 — Ingest prompt v3: minified JSON, code-rendered text, a "tool" bucket
The extraction prompt ends with a one-line JSON shape example and asks for minified output;
`text` may be `""` (code renders it from per-predicate templates); entities lose `summary`; the
entity type `topic` is replaced by `tool` (tools, apps, products, websites, datasets, record ids),
which the normaliser keeps as literal values. Why: decode (~42 tok/s) is almost all of ingest time,
and the model pretty-printed its JSON (an instruction alone did not stop it, even placed last;
the example did). On the fictional homepage (2 chunks) output went from 4010 to ~2.5k tokens and
from 104 s to ~63 s, with recall kept (values, team and every duty extracted). Without a bucket
for tools the model labelled them `project`; with `tool` it files them there and no page is made.
Alternatives: a custom grammar forbidding whitespace (oMLX supports it, but it is runtime-specific
and `brain.llm` stays on the OpenAI-compatible API); shorter key names (smaller gain, less
readable to the model).

## 2026-10-25 — Ingest normalisation guards: repair by type, never guess an end
`normalise.py` checks every fact against a predicate → subject/object type table and repairs what
it can (works_at a project → worked_on, a verb-led works_at → responsibility, works_at a team →
team, uses on the owner → has_skill, an entity where a literal belongs → the literal) and drops the
rest (relations not from a person to the owner, facts about tools, works_at a place, zero counts).
Fact text with a pronoun, the owner's name, no object or over 160 chars is replaced by a template.
A past-tense lives_in/role/team/manager without an end becomes a `note` rather than getting the
current value's start as its end. Every repair is a `Normalised.notes` line. Why: a ~3B-active
model slips even with a good prompt (seen on a real homepage); the checks are cheap and keep
wrong pages and stale "current" facts out of the vault. A guessed end would invent an overlap
("Seville until 2024" next to "Lisbon 2019–2024"); the prompt asks the model for an end when
the text gives one. Alternatives: dropping every suspicious fact (loses duties and teams); asking
the model to self-check (doubles the calls).

## 2026-10-25 — Web pages: fetch to raw, then read like a file
`brain ingest <url>` fetches with httpx (http/https only, browser User-Agent, 20 s timeout, 5 MB
cap), saves the bytes as `raw/docs/YYYY-MM-DD-<host-path>.<ext>` and reads that copy with the
normal readers; the raw-copy commit message records the source URL. `--follow N` ingests the
first N same-origin links on the page (navigation included: on a personal site it leads to the
CV and about pages), each as its own raw doc and change set; a linked page that fails is skipped
and reported. The HTML reader reads animated counters from `data-target`-style attributes,
drops navigation, footers, forms, dialogs, hidden elements and cookie/consent banners, and puts
`<meta name="description">` and schema.org `Person` JSON-LD (contact fields left out) in a short
header. Why: raw/ stays the ground truth for any source, re-ingest stays idempotent (identical
bytes reuse the copy), and the vault keeps where each page came from without a schema change.
Alternatives: a frontmatter sidecar for the URL (a new raw format); headless-browser rendering
(heavy; counters and JSON-LD cover the cases seen so far).

## 2026-10-25 — Ingest: the model plans where to split a long document
A document over one extraction chunk (~500 tokens) is cut into blocks (headings, paragraphs,
lists, tables, code); the model gets an outline (one line per block: number, size, first 16
words) and returns `parts: [{first_block, title}]` (`prompts/ingest_split.md`, temperature 0).
Code repairs the plan (sort, dedupe, drop out-of-range starts, start at the first block, keep
headings with their text), splits a part over the cap at headings/paragraphs without another
call, joins parts under a quarter of the cap to a neighbour, and falls back to the old
heading-bounded packing if the call fails. Each part is one extraction call whose "Section:" is
the part's title. Outlines over 6k tokens are planned in overlapping windows. Why: the owner
asked that long documents be cut where the topic changes (a job, a project, a period of life)
rather than wherever 500 tokens run out; the outline keeps the call cheap (fictional 1.9k-token
page: one call, ~10 s, vs ~2.5 min of extraction) and the model never reproduces the text.
Findings: the model follows headings well but never split a long heading-free section, even
with section sizes or an "OVER CAP" flag in the outline; listing those stretches explicitly in
the user message ("blocks 24 to 33 (645 tok): split into 2 or more parts") made it split them
at subject changes; at temperature 0.2 it sometimes still did not, at 0 it did. Known cost: a
part loses the context of its neighbours (on the homepage the projects part, cut off from the
job history, yields a role taken from the page title, which goes to review as a
contradiction). Alternatives: model-free greedy packing (cuts mid-topic); asking for topic
boundaries and packing them up to the cap in code (fewer calls, but the model's parts are no
longer the chunks); sending the full text (too slow on a 50-page PDF).

## 2026-10-25 — Typed decisions: `LLM.decide` over `/v1/decisions`, structured fallback
`brain.llm` gains `decide(text, questions) -> list[Decision]` (`brain/llm/decide.py`): choice and
yes/no (`predicate`) questions answered with a probability per option and a confidence, via
`POST /v1/decisions` (OpenAI Decisions-compatible; oMLX serves it for the chat model by reading
the option letters' next-token probabilities, one token per question). A refusal (the letters
hold under half the mass) is `Decision.refusal`; FakeLLM's unscripted questions are refusals, so
code must treat "no answer" as "keep what you have". A server without the route (404/405/501 that
is not an API error such as an unknown model) gets one `structured` call with an enum per
question, answers marked `fallback` with confidence 1.0; the client remembers it. Recorder,
FakeLLM replay and StructuredCache cover `decide` (refusals are not cached). Why: closed questions
cost one forward pass, cannot come back malformed, and give a number to threshold; rule 6 keeps
the runtime swappable. Measured on the base M4 (oMLX 0.7.2, Qwen3.6-35B-A3B): the server answers
questions one after the other and re-reads the input for each (~0.35 s + ~1.6 ms per input
token); on this hybrid (GatedDeltaNet) model only the full prompt is cached, so questions sharing
one input do not reuse it, and concurrent requests are serialised. Cost therefore scales with
questions × input tokens; callers keep inputs short and questions few.

## 2026-10-25 — Ingest gate: skip parts that say nothing about a person
Before extracting a part of a multi-part document, one yes/no decision on a compact view of it
(headings and the first 24 words of each block): "Does this excerpt say anything about its author
or the person it describes (work, education, projects, people, places, interests, values or
habits)?". Below `SKIP_BELOW = 0.15` the part is skipped and reported with its probability. A
document's only part is never skipped; a failed or refused decision extracts. Measured on
fictional parts: CV 0.995, hobbies 0.996, opinion post 0.992, contact block 0.59, footer 0.12,
tutorial 0.06, cookie notice 0.05 (phrasings naming "the owner" scored a third-person hobbies
part 0.80 and a contact block 0.56, less separation). The compact view kept the separation
(tutorial part 0.03; a tutorial with one personal line at the end 0.91) at half the tokens of a
long part. Cost ~0.6–0.9 s per part against ~40 s of extraction for a skipped one. Why 0.15 and
not the research's 0.05: positives sit at ≥ 0.99, negatives spread 0.01–0.12, the contact block
(~0.6) must stay. Alternatives: reader rules per kind of boilerplate (the reader already drops
what it can recognise structurally; the rest varies per site); asking on the full part (2× the
tokens, same verdicts).

## 2026-10-25 — Ingest checks: ground, screen, then ask; evidence, never the proposal
After each part's extraction: (1) **grounding** in code: a fact whose subject (unless the owner)
or page-type object the part never names is dropped. A part that passed the gate weakly (a
tutorial with a contact block, p 0.85) restated every entity of its KNOWN ENTITIES list
(works_at the university, worked_on "Bouldering", friend_of the manager) and created seven junk
pages on a vault that already held the homepage. (2) **typed decisions** on what code cannot
verify: new entities (person / org / team / project / place / event / tool / topic / activity;
team, tool, topic and activity become literal values through the normaliser's literal path),
works_at organisations (employer / school / member / other → studied_at, member_of or dropped),
skill vs leisure values (skill / hobby / interest / other → has_skill or interest), and `speaks`
values that are not a known language. Questions show the sentences or list lines naming the
items, under their heading, never the extractor's answer (shown a proposal, the model agrees
with it). A change needs confidence ≥ `ACT_ABOVE = 0.8`; below it the extractor stands and a
note records the answer. Cost decides the shape: each question costs ~0.5–0.8 s (see the decide
entry), so asking about every item of a CV part took 8–14 s; each set is instead screened by one
choice question ("which of these is a team, a tool, a topic or an activity rather than a person,
organisation, project, place or event? / none of them") in groups of at most 6, and only items
at or above `SCREEN_FLAG = 0.15`, items the part never names, and sets of one or two get their
own question. Findings: a re-screen after removing the flagged item mostly flagged ordinary
items (0.86–0.92), so one round; one screen of 11 names missed two of three suspects, so groups
of 6 and a low flag; "none of them" rarely wins a 2-item screen (the past employer next to the
current one at 0.99), so small sets are asked directly; wording the project option "including
the person's own app or library" stopped own libraries reading as products; "skill or hobby?"
sent a work-related interest to has_skill (0.89), a third option "a subject of interest" fixed
it (0.96) and a fourth "none of these" stopped an own library listed as an interest becoming a
skill (0.88 → other 0.81); without an "activity" kind, hobbies typed as projects got 0.39–0.44
for tool/topic, with it 0.83–0.99. Known gaps: the ~2 s per part target is met by the gate alone,
checks add ~2–6 s per part (a fraction of the 20–60 s extraction); grounding can drop a fact that
names its entity only by pronoun across parts. Alternatives: one question per item (8–14 s);
the reranker as a classifier (judges relatedness, not kind); more regexes for teams/tools/hobbies
(per-site rules).

## 2026-10-25 — Section labels from the part's own headings; inline HTML kept apart
A part's "Section:" label is its first heading's trail plus the model's title. A part that
started with a kicker line ("Career" above "## Career path") took that line's trail, which is the
previous section's ("Side projects > A reading tutor > Research Engineer at …"), and the
extraction attributed the side project to the job. A part with no heading continues its
section; one starting with real text of another section keeps only the shared trail. The HTML
reader puts a space between adjacent inline elements whose texts would touch ("02Data lab")
and drops decorative ordinals by structure (a bare 1–2 digit element before a heading, first in a
heading, or first in a short line followed by another element when zero-padded or above a
heading), not by class names: the fix must hold for any site. Counters and counts in sentences
("7+ years", "12 talks") stay.

## 2026-10-25 — A document brief, used by normalisation only
Parts are extracted one at a time, so no part knows that its project belongs to the job two
sections up, that "Atlas" and "the Atlas Migration project" are one thing, or that the intro's
city was left years ago. One structured call (`ingest_brief`, temperature 0) now reads the whole
document (≤ 6k tokens, else each part's heading and first lines) after the split plan and returns
a brief: subject, kind, voice, jobs/studies, homes, and ≤ 12 glossary names with type, aliases,
parent and the subject's role. Code grounds it before use: a date is kept only if a sentence
names its city / organisation and gives that year, a home only if a sentence names the city with
a residence word (the model took a job's start for a move, and every employer's city for a home),
descriptions are not aliases. Normalisation then: merges variant names into one entity (brief
name or alias, or the name plus lowercase words or a version, same type only); adds `part_of` /
`located_in` from parents for entities the facts use (never for a side project; unlisted parents
typed from the child) and turns a stated-backwards one around; adds `member_of` for a "member"
organisation no fact links (not one of the document's teams); sharpens `knows` (and friend →
family); ends a past city where the next home starts and adds a home the parts missed (never the
hometown, never an end on the last home); keeps two stints at one employer apart when the brief
shows both and the dates leave a gap. Every change is a `brief:` note.
Measured with `brain eval ingest`, two extraction samples per variant (the eval's run-to-run
spread is larger than assumed: two baseline samples differ by 5 precision points, and a single
document's optional recall by 40 points; samples differ by the model's sampling at 0.2):

| variant (sample 1 / sample 2) | required | optional | precision | dates | pages | s/run |
|---|---|---|---|---|---|---|
| baseline (HEAD 3f246a3) | 0.959 / 0.945 | 0.640 / 0.649 | 0.775 / 0.724 | 0.889 / 0.871 | 0.782 / 0.795 | 350–460 |
| brief → normalisation (kept) | 1.000 / 0.986 | 0.784 / 0.793 | 0.812 / 0.755 | 0.897 / 0.881 | 0.872 / 0.885 | ~440 |
| + brief block in every extraction prompt (timeline, names) | 0.945 / 0.918 | 0.766 / 0.757 | 0.744 / 0.740 | 0.935 / 0.903 | 0.859 / 0.756 | ~720 |
| + names-only block in extraction | 0.945 / 0.986 | 0.775 / 0.784 | 0.756 / 0.774 | 0.853 / 0.776 | 0.833 / 0.821 | 450–740 |
| + brief header as checks context | 1.000 / 0.986 | 0.757 / 0.784 | 0.806 / 0.754 | 0.896 / 0.881 | 0.859 / 0.872 | — |

The normalisation-only rows share their extractions with the baseline rows (same cache), so the
deltas are paired: +4 required, +14 optional, +3 precision, +1 dates, +9 pages on both samples;
"optional fact not extracted" 28 → 18 and 25 → 13, "entity missing" 17 → 10 and 16 → 9 (project
`part_of` its employer in every document, `member_of` a network stated as a duty, a past city
with its end, the internship kept apart from the later job at the same employer, cities as
pages). Queued +1 to +2 (relations and past homes). In the extraction prompt the brief lost: the
extractor copied the brief's rounded years over the excerpt's months (dates 0.75 with dated
periods, so periods are shown undated), restated or lost facts (required −4 to −7 against
normalisation-only, precision −2 to −6) and wrote longer replies (a notes document truncated
and was halved into six parts); its prefill is ~200–300 tokens per part (cached after the first
part). The checks' context changed nothing. So `pipeline.BRIEF_USES = {"normalise"}`; the prompt
block and the context stay as switches for a stronger model. Cost: 12–17 s per document
(~1.1–1.6k prompt, ~410–600 output tokens; v2 dropped tools from the glossary and capped it at 12,
−10% output, same scores), about +25% on an uncached ingest. Known gaps: the brief's types
are not checked (a band typed "team" gets no `member_of`); a home's residence word is matched per
sentence, so "lived in Seville and worked in Madrid" keeps Madrid as a home.

## 2026-10-25 — `member_of` is multi-valued (schema v0.2)
A person belongs to several clubs, networks and groups at once; as a single-valued predicate a
second membership was treated as a contradiction and dropped or queued. `located_in` and
`part_of` stay single-valued.

## 2026-10-26 — Timeline rows as lines, résumé dates in grounding, story details dropped
A real homepage and résumé showed three generic failures; each fix is structural, measured on
the eval set extended with two documents of a second fictional persona (a homepage whose
experience/education timelines are CSS-grid rows with the date cell before the name and
CV-style "Sep 21 — Feb 23" dates; a first-person experiment write-up with a few facts in a long
story). Two extraction samples per variant (separate caches); post-extraction fixes are paired
on the baseline's cached extractions, the reader fix re-extracts only the HTML whose text
changed.
1. **Rows** (`html.join_rows`): an innermost repeated block holding 2–8 short cells becomes one
   list item ("- 2019 — 2020 · School B · MSc in X, double degree with School A"); `<dl>` pairs
   likewise. Cells read one per line put each date above the name it belongs to, and the model
   shifted dates by one row on the real page. A series where any cell is a write-up (> 30 words:
   project cards) is left alone: a first version also rewrote cards (label moved into the
   heading, or one line per card) and the card homepage lost 9 points of pages and 13 of
   precision on one sample, gained 3 on the other.
2. **Résumé dates** (`dates.years_in`, `brief.units`): grounding read only four-digit years, so
   "Sep 21 – July 22" lost every period date; it now reads month + two-digit year, MM/YYYY,
   MM/YY, YYYY-MM and ’YY, keeps CV lines (ending with a date, or starting with a date range)
   apart from wrapped prose, and keeps a record line whole (a CSV row's date was cut off from
   its organisation by a sentence split once lines stopped being joined). An end "now" is still
   not checked. Extraction already turned "Sep 21" into 2021-09 (verified on the new homepage),
   so `normalise_date` is unchanged.
3. **Story details** (`decisions._check_stories`): the owner's duty / skill / interest / habit /
   likes statements are screened, then asked "what does this statement record?" (job / trait /
   event / lesson); p(event or lesson) ≥ 0.8 drops the fact. Wording chosen on 29 labelled
   statements from the eval documents (13 story details, 16 facts), at 0.8: job/trait/event/
   lesson caught 9 and lost no fact; duty/trait/story caught 8 and lost a CV achievement;
   "still true a year later?" 5 (one fact lost); "would it belong in a profile?" 3; "fact or
   story detail?" kept everything (measured on the eval: 0 drops). Drop rather than `note`: a
   note is still an unverified fact in the wiki (with the first wording, `note` scored precision
   0.785 / 0.741 against `drop`'s 0.791 / 0.744 and the baseline's 0.785 / 0.745), and the raw document stays in the vault for Phase 2 episodes. Languages are not
   asked (a has_skill "Dutch" with a story-like text was dropped before normalisation could
   route it to `speaks`). Hike mountains listed as interests are not caught (named places read
   as interests; open).

| variant (sample 1 / sample 2) | required | optional | precision | dates | pages | queued |
|---|---|---|---|---|---|---|
| baseline (HEAD 5607008), 8 docs | 1.000 / 0.989 | 0.802 / 0.817 | 0.785 / 0.745 | 0.904 / 0.890 | 0.889 / 0.900 | 21 / 20 |
| + résumé dates only | same | same | same | same | same | same |
| + story check only (paired) | 1.000 / 0.989 | 0.802 / 0.817 | 0.794 / 0.750 | same | same | same |
| + rows, cards rewritten too (dropped) | 1.000 / 0.977 | 0.778 / 0.825 | 0.779 / 0.745 | 0.904 / 0.889 | 0.878 / 0.889 | 20 / 22 |
| + rows, write-up series left alone | 1.000 / 0.989 | 0.802 / 0.817 | 0.787 / 0.744 | same | same | same |
| all three (kept) | 1.000 / 0.989 | 0.802 / 0.817 | 0.795 / 0.749 | 0.904 / 0.890 | 0.889 / 0.900 | 21 / 20 |

Per target document: the write-up's precision 0.471 → 0.533 and 0.571 → 0.615 (story check;
year-in-review notes 0.581 → 0.600, 0.655 → 0.679); no gold fact lost anywhere. The timeline
homepage scored required 1.0 and dates 0.92 already at baseline (the shift did not reproduce
with this model on this page), facts unchanged with rows (precision +1.3 / −0.8), but the brief
now keeps all 11 period dates the grounding dropped before (rows + résumé dates; neither alone).
Cost: rows none (73 / 93 s vs 73 / 74 s per sample on that page: sampling); story check
~1.2 s per part (33 questions over 20 parts, ~3 s per document, a few % of an ingest); dates
none. Known gaps: one line per row hides a row heading from the outline (rows are small);
the story check sees paraphrased statements, so its evidence is the whole part when the value
is not named.

## 2026-10-26 — A duty filed under a known employer keeps the employer
The "works_at that describes the work" guard turned `works_at Fabrikam: "Did demand forecasting"`
into a responsibility, dropping the employer link; when that was the employer's only fact, its
page was dropped too (an internship lost on the timeline eval homepage, after the typed check had
confirmed the employer). Now a `works_at` whose object is an organisation page keeps the link,
with its dates, and the duty is added as a separate responsibility; only a `works_at` with no
page behind it becomes a responsibility. Paired re-score on both cached samples: the six older
documents are unchanged; the timeline homepage keeps the internship (precision +2 points on both
samples, dates 0.92 → 1.00 on one).

## 2026-10-26 — Degrees linked to their school (`at::`, schema v0.3); paraphrased facts merged
Two generic failures seen on a real homepage.
1. **A degree shown next to the wrong school.** Extraction gives `degree` facts and separate
   `studied_at` facts; the card paired them by overlapping dates, so a timeline's preparatory
   school (2017–2019) took the engineering degree that started in 2019, and a double degree's
   school took both degrees. Options weighed: (a) link the degree to its school when one row or
   sentence states both, (b) pair on a shared source part/line at card time, (c) copy dates
   between a degree and its row's `studied_at`. (b) is not possible from markdown: facts keep the
   document, not the line, as source. Chosen: (a) + (c). Normalisation (`_link_degrees`,
   `education.degree_school`, given each part's text) finds the unit naming most of the
   degree's words, takes the document's schools named there (or in the short line above it: a
   heading or "Org, City"), and of several the one whose `studied_at` has the degree's dates,
   else the nearest before the degree in a record/timeline row ("2019 — 2020 · School B · MSc X,
   double degree with School A") and the nearest after it in prose ("a master's in X at School
   B"). The link is a fact qualifier `(at:: [[orgs/…]])` like `level::`/`rel::` (schema v0.3; a
   merge repoints it; `brain reindex` reads it from markdown). When the school's `studied_at` in
   that part is undated and the degree is dated (or the reverse), the undated one takes the
   other's dates. The card pairs by the link; an undated school takes its degrees' period; an
   unlinked degree (older vaults, hand edits) goes with the one school whose period equals its
   own, else the one where it began that overlaps it by more than a year (a start-only or
   one-year degree: the school whose period contains it); a tie or a shared boundary year is no
   evidence and the degree gets its own line.
2. **The same duty in other words** ("Fine-tunes graph neural networks for solubility
   prediction" / "Responsible for fine-tuning graph neural networks to predict solubility"),
   also skills and interests ("deep-learning" / "Deep learning"). `dedup` runs after
   normalisation on new facts of multi-valued literal predicates with the same subject and
   predicate, within the change set and against current vault facts: equal words after folding
   → same (no model); else Qwen3-Embedding cosine with an instruction on both sides ≥ 0.90 →
   same; 0.75–0.90 → a typed decision ("same thing in other words, or one is a more detailed
   version / different things, even if related"), p(same) ≥ 0.8 → same. Two dated facts with a
   gap are never merged (one duty at two jobs). The more specific statement (more words, then
   earlier start) is kept with both periods; a restatement of a vault fact takes that fact's
   value, so the writer's restatement rule applies. Embeddings down → only equal words merge
   (a note). Calibration on 85 labelled fictional pairs (`tests/fixtures/dedup/pairs.yaml`:
   40 paraphrases, 45 related-but-different duties/skills/interests/habits/goals; scores
   recorded in `tests/fixtures/llm/dedup/pair-scores.json`):

| representation (closest different pair) | paraphrases above it |
|---|---|
| rendered fact text (0.896) | 0.50 |
| value only (0.875) | 0.57 |
| own text, else value (0.875) | 0.60 |
| own text, else value, instructed both sides (0.850) — kept | 0.72 |

   Decision wordings on all 85 pairs: a yes/no "same thing, one can be dropped?" merged 12/40
   at p ≥ 0.8 (0 wrong), the two-option choice 28/40 (0 wrong; different pairs ≤ 0.62, ≤ 0.11 in
   the asked band). Kept rule: 29/40 paraphrases merged, 0 of 45 different pairs (in-sample;
   margins 0.05 on the cosine, ≥ 0.69 on p in the asked band). Prefer missing a duplicate: a
   wrong merge loses a fact.

Measured with `brain eval ingest`, now 9 documents: a Theo "about" page was added whose intro,
job list and "Now" restate the same duties, skills and hobbies, with degrees in prose (one gold
fact each). The scorer gained `at` on gold degrees and `link_accuracy` (JSON only; matching is
unchanged). Paired on cached extractions (two samples, separate caches), HEAD vs each fix:

| variant (sample 1 / sample 2) | required | optional | precision | dates | pages | links | queued |
|---|---|---|---|---|---|---|---|
| HEAD 4d891c5 | 1.000 / 0.990 | 0.801 / 0.809 | 0.791 / 0.748 | 0.916 / 0.904 | 0.878 / 0.898 | 0 / 0 (of 9) | 21 / 20 |
| degree links | same | same | same | same | same | 1.000 / 1.000 | same |
| dedup | same | same | 0.799 / 0.759 | same | same | 0 / 0 | same |
| both (kept) | 1.000 / 0.990 | 0.801 / 0.809 | 0.799 / 0.759 | 0.916 / 0.904 | 0.878 / 0.898 | 1.000 / 1.000 | 21 / 20 |

Per document: the "about" page's precision 0.833 → 0.926 and 0.788 → 0.867 (3 merges each);
the timeline homepage 0.750 → 0.794 on sample 2 (two restated duties); the CV 0.870 → 0.867 on
sample 2 ("accessibility" merged into "Accessibility (WCAG)": both had matched a gold line);
others unchanged. All 9 merges across both samples are right; no gold fact lost. The timeline
homepage's card at HEAD read "2016–2019 · MSc in Machine Learning, Engineering degree in applied
mathematics, Halvorsen …" and "2014–2016 · Engineering degree …, Lycée Arbelet"; with the links
each degree sits with its own school. No date was copied on the eval (the model dated every
school), so dates are unchanged. Cost: degree links none; dedup one embedding call per document
plus ~0.8 s per asked pair (+0.2–0.3 s per document with cached decisions, ~2.6 s on the about
page uncached), against ~60–130 s of extraction. Known gaps: "Manages DS team" / "Manages team
at Fernhill" (cosine 0.65) and "Boulders twice a week" / "Climbs at the bouldering gym" stay
apart; a degree whose line names no school (a markdown CV with the school as a heading two
lines up) gets no link; `at::` is not a retrieval graph edge yet.

## 2026-10-26 — Interview mode: private regions only on request; code steers, the model phrases
Decision: `brain interview` never picks a region the life map marks `privacy: private` (family,
friends, traits, health) on its own. It interviews about one only when the owner names it
(`brain interview relationships.friends`, its label, or a unique last part such as `friends`);
naming a parent (`relationships`) picks among its non-private children. In a private region every
answer is marked private, and anywhere the owner can say `private` (or `private: <answer>`) to
mark the answers of the current step. The transcript carries `[interview] privacy:: private`
system turns before those answers (and `process_session` gets `private_turns` once it accepts
them), so nothing from them reaches the big model.
Why: an unprompted "tell me about your health" is intrusive, and the owner should decide when
private topics enter memory. Coverage of private regions is still computed and listed in
`questions.md` (`privacy:: private`), so the owner sees the gaps.
Also: region choice is `argmax(weight × (1 − coverage) + boost)`, the boost counting only items
other components or the owner added to `questions.md` (0.1 × priority/5 each, capped at 0.3: a
nudge, never an override); coverage items are already in the coverage term. Slot coverage gives
partial credit to multi slots (`n/min`) and none to stale facts; timeless slots (no
`stale_days`) count history, since a school that ended is still known; "confirmed" is the latest
of `rec::` and the dates in `src::`. Per answer, `LLM.decide` labels it (answered / partial /
declined / unclear, plus "opens a new thread?") and code chooses follow-up, clarify or next step
(≤2 follow-ups per step, 3 declines in a row end the session); a structured call phrases the
question, and code rejects a message naming anything (capitalised word, number) absent from the
facts it was given, the profile or the conversation, falling back to the slot's hint.
Alternatives: let the model choose the next action and write the question in one call (fewer
calls, but no typed signal to test and tune, and harder to keep to one question); ask private
regions at low priority (rejected: still unprompted).

## 2026-10-26 — Memory pipeline: episodes, predict-calibrate, owner corrections (schema v0.4)
`brain.memory.process_session` turns a finished transcript into episodes and memory (the
`memory` skill has the flow). Choices:
1. **Reuse ingest's machinery** rather than a second extractor: the episode's candidate facts are
   handed to `ingest.decisions.check_part` (grounding, typed checks, story details),
   `normalise.build_changeset` (now with `origin=`) and `dedup.merge_paraphrases` unchanged; ingest
   tests and its eval are untouched. Memory adds only what conversations need: KNOWN FACTS in the
   prompt with a per-fact label (predict-calibrate, Nemori), comparison with memory by name
   (`plan_ops`), grounding of objects of any predicate (`ground_links`), and owner corrections.
2. **Small talk** is decided by one typed yes/no on the owner's lines (as ingest's gate), not by
   the segmenter alone: no segmentation call (~6–10 s) for a "hey, tell me a joke" chat, and a
   `no episode` log line so it is never asked again (a marker in markdown, not in `.brain/`).
3. **Importance** is computed in code from categorical labels (emotion none/mild/strong, goal
   none/related/central, life change none/minor/major, a date ahead, "remember this"): 2 + 0/1/3
   + 0/1/2 + 0/1/3 + 1 + 3, clamped to 1–10. Small models rate numbers badly; labels are stable.
4. **`confirms` changes nothing but provenance.** A restated fact gets the episode added to its
   `src::` (`confirm_fact`), never a new `rec::`, a new fact or new dates. More sources raise its
   salience on the card; "last confirmed" for interview staleness can be read from it later.
5. **Owner corrections auto-apply** (schema v0.4, Reviews): an *explicit* correction of the
   owner's own fact, decided by a typed question at p ≥ 0.9 and allowed by
   `writer.correction_problem` (origin extract; about `self`; no relationship; nothing private;
   stated, same predicate). Implicit contradictions, other people's or projects' facts,
   relationships and private material stay in the review queue; each change is one vault commit.
   Question chosen on 20 labelled fictional cases (`tests/fixtures/memory/corrections.yaml`):

   | wording | explicit (9): min p | others (11): max p | explicit ≥ 0.9 | others ≥ 0.9 |
   |---|---|---|---|---|
   | yes/no "explicitly says … wrong, changed or no longer true?" | 0.78 | 0.56 | 3 | 0 |
   | yes/no "explicitly correct or update … (value or dates)?" | 0.35 | 0.56 | 5 | 0 |
   | four-way wrong / changed / same / other | 0.996 | 0.92 | 9 | 1 |
   | three-way corrects / restates / something else | 1.000 | 0.84 | 9 | 0 |
   | **two-way yes/no choice with descriptions (kept)** | **0.994** | **0.59** | **9** | **0** |

6. **An ended fact can be corrected** ("I left in March, not January"): a History fact without
   `by::` may be superseded by a replacement; it stays in History with `by::`. Without this the
   corrected job would have been a second, contradicting fact.
7. **Private answers** (interview `[interview] privacy:: private` markers, or `private_turns`):
   per episode, conservatively: the whole episode is `privacy: private` (its summary may quote
   the answer), its facts get `privacy:: private` (a fact-level flag the index and the card's
   big-model variant honour, so a private answer about the owner can live on a normal page
   without leaking), new pages are private, no alias is added, and no correction from it is
   auto-applied. Interview markers are shown to the model as `NOTE` lines (never facts) and
   their `region::`/`about::` join the episode's regions and entities.
8. **Resolution** proposes candidates by words and by name embeddings (instructed both sides:
   related 0.58–0.96, unrelated people/orgs ≤ 0.48, two cities 0.73), but only a typed choice
   (p ≥ 0.8) links a mention to a page; embeddings alone never merge.

Tried and dropped: a yes/no correction predicate (too few explicit corrections reach 0.9); the
model's "contradicts" label alone as the trigger (it mislabels; the writer's scoped check and the
decision decide); per-turn privacy inside an episode (the summary would leak the answer).

Measured with `brain eval memory` (9 fictional transcripts, Alex Rivera; two `--no-cache`
samples):

| sample | episode count | regions | fact recall | precision | corrections auto-applied | queued | intentions | wrongly auto-applied | duplicates | s total |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 9/9 | 11/11 | 5/6 | 9/12 | 2/2 | 0/1 | 2/2 | **0** | 0 | 175 |
| 2 | 9/9 | 11/11 | 6/6 | 10/12 | 2/2 | 0/1 | 2/2 | **0** | 0 | 189 |

Per session 15–25 s for a 3–5-turn chat (segmentation 6–8 s, ~1 s for small talk that the gate
stops), ~40 s for the three-topic chat. Two earlier samples, before grounding of link objects
and the membership-text fix, had the same counts but 1 of 4 corrections lost (a "Left Fabrikam
in March 2023" `works_at` turned into a duty by ingest's verb-led guard). Unverified extras are
mostly reasonable but outside the gold (a planned project "uses Rust", a goal "transfer to the
Lisbon office"); systematic misses: "moving in January" dated to the past January (5 of 6 runs over all
samples), the implicit contradiction never extracted as `lives_in` (so nothing queued: the safe
side), a self `goal` linking to the project page.

## 2026-10-26 — `brain lint`: graph cleaning by reviewed operations and owner questions (schema v0.5)
`brain lint` (`brain.lint`, the `lint` skill has the flow) finds five kinds of dirt and proposes
ordinary operations, one review per finding, so the owner approves or rejects each repair alone
(a review is all-or-nothing). Nothing it proposes auto-applies; every doubt between 0.5 and the
acting threshold becomes a `questions.md` item (origin `lint`, never asked twice: the key before
`:` in its reason), because the owner is the oracle.
1. **Duplicates**: same-type pairs proposed by name signals in code (same words without generic
   ones such as "University of" / "École", acronym, typo by Damerau-Levenshtein, one name inside
   the other, shared first name or first word) or by profile embeddings (cosine ≥ 0.75; on the
   eval vaults duplicates scored 0.50–0.95, acronyms lowest, different pages ≤ 0.65), ordered by
   the reranker (≤ 3 per page); never two people with different surnames, never two pages linked
   by a fact. A typed choice same/different on both profiles and the source sentences naming them;
   p(same) ≥ 0.9 → `merge_entities` (the older page is kept). A page with several look-alikes gets
   one exclusive choice "which one, or none?" (a pairwise rule blocked a right merge when a
   weaker look-alike, Atlas Reporting, reached 0.53; "Sam" next to two Sams now answers "none",
   0.96–0.98, and nothing is asked).
2. **Doubtful types**: only pages used as values (`has_skill`, `interest`, `uses`…) or with
   nothing that fits their type are asked about (person / org / project / place / event / topic /
   tool / activity; the current type is never shown). Demotion uses the value kinds' summed mass
   ("Data Contracts": topic 0.89 + tool 0.09) at ≥ 0.9 and only for a page without facts of its
   own; any type change needs a fact or link that says so (`works_at` X makes X an organisation).
   A synthetic 100-page vault showed why: made-up place names linked by `related_to` were
   retyped as people at 0.91–1.00 from the name alone. Without support and without a source
   sentence a verdict is a report line, not a question.
3. **Facts on the wrong subject**: the owner's `goal` / `deadline` / `status` / `uses` whose text
   or source sentences name a project; "whose goal is this?" at ≥ 0.8 → `refile_fact`.
4. **Missing edges**: project → job (`part_of`), undated duty → job (`at::`, new in v0.5 for
   `responsibility` and `role`; the card no longer lists a duty of an ended job as current),
   organisation → place (`located_in`, only places the vault knows that the org's source sentences
   name). Inferred, with `evidence::` (sources, the `works_at` fact, a quote).
5. **Consistency** in code: two current single-valued values or two open-ended jobs (asked),
   end before start, orphans (no facts *and* no links: with "≤ 1 link" every employer page ingest
   makes was reported), literal objects naming exactly one page of a fitting type (`refile_fact`
   with the link), parse warnings, inferred facts without evidence, and the AGENTS.md predicate
   list and version against `models.py` (closes the Phase 0 open item; a test keeps the template
   in sync).

Writer (schema v0.5): `demote_entity`, `retype_entity`, `refile_fact` (always reviewed); a merge
folds restatements into one fact (sources joined), retires self-references, queues clashing
single-valued facts as a new review and repoints pending reviews; inferred facts need
`evidence::` (two from the big model). `brain lint` commits uncommitted hand edits apart first.
Incremental by default: pages changed since the last lint (`.brain/lint.json`) and their
neighbours; `--all`, `--since REV`.

**Option order matters for one-token decisions.** With the listed cities first and "another
place / not stated" last, "Brightwater Studio, a Stockholm design agency, I moved to Rotterdam…"
gave Rotterdam 0.98 (Stockholm was not a place the vault knew); with the escape options first,
"another place" 0.91 while every true location stayed ≥ 0.97. The same order turned a project no
source names from "side project" 0.91 into "unclear" 0.53, and a client job from 2021 from "side
project" 0.86 into "unclear" 0.48; clear cases kept 0.95–1.00. Every lint choice now lists
"unclear" / "none" / "another place" first, and employers without a page are options too (acting
needs a page). The part_of wording was chosen on eight fictional projects ("According to the
text, where does the owner do this project?": Tidepool 0.99 vs 0.71 for "Is this project part
of the owner's work at one of their jobs?").

Other things tried and dropped: duty evidence as "sentences naming a job" (the employer header
lines filled the budget, "unclear" 0.58; now sentences sharing the duty's words, a list item
carrying the line above it); jobs offered only with an org link (a literal "Fabrikam Analytics"
could not be chosen; a literal naming exactly one org page now counts as that page); a
different-people guard on the last word only ("Marchetti, Theo" was never compared); marking
both sides of a merge busy (a literal "Northwind Labs" could not be linked to the kept page).

Measured with `brain eval lint` (`src/brain/eval/lint_eval.py`): the fixture vault + a richer
fictional persona (20 pages), and per seed a dirty copy with 12 noise items (variants by seed:
org typo, "University of" prefix, acronym, accent / "Last, First", a project's other name, a
topic page, a project goal on the owner, a missing `part_of`, a duty under the wrong job or
none, literal employers, two current cities, an org that lost its location) next to near misses
that must stay untouched (three Sams, Atlas Migration / Atlas Reporting, Northwind Labs /
Northwind Traders, a side project, personal goals, a university whose location no source
states). Every lint review is approved, then the vault is checked. Seeds 1–3 were used while
building; 4–6 only at the end (same variant pool). `--no-cache`:

| sample | runs | noise fixed | wrong changes | wrong on the clean vault | questions (expected matched) | ops | decisions | s |
|---|---|---|---|---|---|---|---|---|
| 1 | clean + seeds 1–3 | 36/36 | **0** | **0** (0 questions) | 3 (3/3) | 36 | 55 | 80 |
| 2 | clean + seeds 4–6 | 36/36 | **0** | **0** (0 questions) | 3 (3/3) | 36 | 54 | 73 |

Acting decisions ranged 0.92–1.00 (the typo merges lowest, 0.92–0.95). Six vaults produced by
`brain eval ingest` on the Maya and Theo documents (real ingest output, other personas): two
`located_in` proposals (both supported by the text), no wrong change after the order fix, and
real dirt asked about (three current jobs on one homepage). Cost: a full run is ~0.5 s per page
on the base M4 (a synthetic 100-page vault: 51 decisions, 100 texts embedded, 18 reranks, 56 s;
200 pages: 96 decisions, 112 s; no change proposed on either); incremental runs check only what
changed. Known gaps: duplicates across page types; one persona in the eval; a doubt answered in
an interview is settled only through what the memory pipeline makes of the answer.

## 2026-10-26 — Memory extraction in windows; questions ticked only once their answers are in memory

A twelve-minute interview on one topic became a single 80-turn episode (~1,600 words), and its
extraction reply hit `max_tokens` (3072): the whole episode was lost until a retry. Raising the
limit only moves the wall, and one call over a long span also extracts less per exchange. Now an
episode is extracted in windows of whole exchanges (`WINDOW_WORDS` = 450, cut only after an owner
turn, so a question stays with its answer); every window sees the same known facts and episode
summary, and the windows' extractions are merged (entities once per name) before calibration. A
window whose reply is still cut off is halved at an exchange boundary, as ingest halves a chunk;
a single exchange that is cut off fails as before. Short episodes are one window, so their
behaviour and recorded fixtures are unchanged. Re-run on the failed transcript: one episode, 75 s.

The interview used to tick its questions.md items whenever the pipeline returned, even when an
episode had failed; those answers were then neither in memory nor asked again. Items are now
ticked only when every episode was extracted; otherwise they stay open and the CLI points to
`brain memory process --pending`.

## 2026-10-27 — Interviews group, link and settle lint doubts (schema v0.6)

A twelve-minute "Projects" interview after `brain lint` was almost all taxonomy, asked doubts
about one page several turns apart, asked things the owner had just answered (or refused), named
only the current employer, answered "can you tell me more?" with "which part?", said "You listed
X as a project" about a page ingest made, and settled nothing: answers reached the graph only
through extraction, which never demotes, merges or adds `part_of`. Now (`brain.interview`):

- **questions.md `doubt::` (v0.6)**: lint stores the finding key on its item (older items: read
  from `reason::`). The plan groups an entity's doubts into one block and asks its `type` and
  `part_of` doubts as one question; at most 4 entities per session, interleaved one for one with
  coverage steps, and only in the first 40% of the time. Pages with a type or merge doubt get no
  coverage questions that session (they may disappear).
- **Employers in questions**: at most two, ranked by evidence (`lint.edges.likely_jobs`: source
  sentences naming both, a linking fact, the model's pick; overlapping dates count half and never
  name a job alone: a weekend project overlaps the job of its time). One that stands out is named
  alone; with no evidence the question is open ("Which of your jobs is X part of, or is it
  something you do on your own?"). Every employer stays in the typed choice that maps the answer.
- **Answer linking**: before each planned step and each generated follow-up, one `LLM.decide`
  call: "already answered?" (open / answered, p ≥ 0.6 skips) and, only after a refusal and for
  questions sharing a word with a refused one, a predicate "asks again for what was refused, or the
  same kind?" (p ≥ 0.35 skips). Tried and dropped: "declined" as a third option of one choice
  (0.7–0.77 for plain questions after "nothing more on that", which skipped a merge doubt); the
  predicate on every question (drifted to ≥ 0.5 for unrelated ones over a long session).
  "Refused" is a new answer label, counted only at p ≥ 0.9 ("I'd rather not say" 1.00, "nothing
  more on that for now" 0.70). The link input is an append-only window of the session (its prefix
  stays cached: an identical input costs 0.08 s against ~1.2 s cold) with older exchanges naming
  the step's entities after it; each decide question costs ~0.5 s uncached, so one candidate per
  call. Follow-ups see "already said" and "declined" lines in the PLAN block.
- **Asking back**: answer label `asked`; the reply is code-built from what the vault has (title,
  first sentence, source name: "from your document “homepage”") and repeats the question, once.
- **Provenance**: prompts say where a fact came from as given, or "I have X filed as…"; a code
  check (`invented_claim`) rejects "you mentioned/listed/noted X" unless most of X's words are in
  the owner's answers (the model wrote "You noted Atlas is in testing" on a first question in 2 of
  3 runs before the confirm-mode wording changed). The card's "To ask" lines are dropped from the
  interview prompt: the model asked a plan question as a follow-up from them.
- **Settling** (`brain.interview.settle`, at close, before the memory pipeline): a typed choice per
  doubt on the step's own non-private exchanges, "does not say" first; the kind choices describe
  what the owner says (with lint's descriptions, "stream processing", which the owner called a
  project, was project 0.53 / topic 0.41; with owner-oriented ones 0.97–1.00, and "it's not a
  project, it's what I work on" never above 0.31 for project). **The owner said it, so p ≥ 0.9
  applies the change** (origin `user`, source = the transcript; `part_of` is `stated`), like an
  owner correction: an explicit statement in answer to exactly that question, a closed option set,
  one change set per doubt in `log.md` and git. 0.5–0.9 or a fallback decision queues the same
  operation as a review (`part_of` then `inferred`, evidence = transcript + the owner's words);
  "does not say" or < 0.5 leaves the item open. A project placed at a job (part_of ≥ 0.5) settles
  its type doubt when "project" was the best answer. Private answers never settle anything.
- **Soft time limit**: at `--minutes` the current entity block is finished, then "We've covered
  … Keep going, or stop here?"; a clear yes adds half the time, anything else stops.

Measured with `tests/interview/test_interview_doubts_live.py` (fictional vault: the lint eval
persona + a homepage listing Stream processing, Ingestion, Fraud alerts and Transaction monitoring
without employers; four lint doubts; a scripted owner who answers stream processing with "a
project at Fabrikam … ingestion is not a project", asks back on the duplicates, refuses a client's
name), local model, five samples (the last runs after the fixes):

| | repeated / already-answered questions | refused info asked again | ask-back answered with the vault's descriptions | doubts settled right | wrong graph changes |
|---|---|---|---|---|---|
| first run, 2 samples | 0, 1 | 0, 0 | yes, yes | 3/4, 3/4 | 0, 0 |
| final, 3 samples | 0, 0, 0 (one flagged by the scripted owner was a new question, "what was the scraper's purpose?": its check was too broad) | 0, 0, 0 | yes ×3 | 4/4 ×3 | 0, 0, 0 |
| final + memory pipeline, 2 samples | 0, 0 | 0, 0 | yes, yes | 4/4, 4/4 | 0, 0 (after extraction: one `part_of`, neither removed page recreated, no review left) |

Latency on the base M4: a reply that was 5.9 s (classify + compose) on the old code is 7.4 s cold /
2.4–2.8 s warm on the recorded check; in the long scripted sessions a turn averages 6.4–8.5 s
(classify 1.15 s, linking 0.9–1.5 s, compose 4.9–5.2 s; a dropped follow-up costs a second
compose, max ~16 s).
