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
previous section's ("Side projects > A voice companion > Research Engineer at …"), and the
extraction attributed the side project to the job. A part with no heading continues its
section; one starting with real text of another section keeps only the shared trail. The HTML
reader puts a space between adjacent inline elements whose texts would touch ("02Claude lab")
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
