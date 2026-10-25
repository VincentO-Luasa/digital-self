# Retrieval: how the brain remembers (v0.1, proposed)

Goal: make a small local model (~3B active parameters, ~400 tok/s prefill) converse as if it
*knows* its owner. The approach copies what people do when they talk to someone they know well:

1. **They already have an impression of the person.** Temperament, how they act, what matters to
   them. Nobody looks this up; it is simply present.
2. **They keep in mind the things that marked them.** A few salient facts and events about the
   person are always close at hand.
3. **As the conversation moves, cues bring memories back.** The current topic and the people
   mentioned call up related past conversations ("this reminds me of when…"). Memories they have
   recalled recently or often come back more easily, and recalling a memory strengthens it.
4. **They hold only a few things in mind at once** and work from the gist, reaching for exact
   detail only when it is needed.

Each of these maps onto a well-studied mechanism, and recent LLM-memory systems have measured
gains from implementing them. Graph spreading beats pure vector search on associative questions
(HippoRAG 2), activation dynamics and decay matter (Synapse ablations), and moving LLM work out of
the hot path into offline consolidation saves large numbers of tokens and calls (LightMem, Mastra).
Sources are at the end.

## The mechanisms we use

| human mechanism | in the brain |
|---|---|
| Person schema / impression formation (McAdams; Conway; Park; Klein, Loftus & Kihlstrom) | **L0 schema card**: identity, traits and values, current chapter, life periods, people, open loops; always in the prompt |
| Flashbulb / emotional salience | importance score sets a **floor under decay**; top items become "marking facts" |
| Working memory ~4 chunks (Cowan) | **L1 buffer**: 1 topic + 3 entity/episode slots + 1 conversation-state line |
| Cue-dependent retrieval, encoding specificity (Tulving) | cues = entities, topic, time, wording of the current turn; episodes are indexed by the same cues |
| Temporal context model (Howard & Kahana) | running context vector `c_t = norm(e_turn + 0.5·c_{t-1})` instead of the last message alone |
| Spreading activation (Collins & Loftus; ACT-R) | **Personalized PageRank** over the entity/fact/episode/link graph, seeded by the cue entities |
| Fan effect (ACT-R) | a cue linked to everything carries little signal: seed weight ∝ 1/ln(1+degree); `self` ≈ 0 |
| Base-level activation, power-law forgetting (Anderson & Schooler) | `B = ln Σ_j t_j^-0.5` over access times: recent *and* frequent use → easier recall |
| Priming | items that just left working memory get a small bonus for 5 turns |
| Event segmentation (Zacks; EM-LLM) | a topic shift closes the current episode and triggers fresh retrieval; retrieved episodes pull in their neighbours in time |
| Gist vs verbatim (fuzzy-trace theory) | every memory has a ≤30-token gist, a 3–5 line summary, and verbatim text fetched on demand only |
| Reminding (Schank) / proactive recall | **L4 surfacing**: at most one "possibly related memory", gated by a high threshold |
| Prospective memory | `intentions`: "remind me about X when Y comes up", triggered by cue match |
| Reconsolidation | using a memory adds an access (reinforcement); a contradiction creates a new fact version |
| Retrieval-induced forgetting | used only at **ranking** time (diversity / lateral inhibition), never to demote stored memories |
| Complementary learning systems (hippocampus/neocortex) | episodes are written fast during the day; nightly "sleep" distils them into facts and rebuilds L0 |

## Prompt layout (~1.8–2k tokens, ordered for prefix caching)

```
[system ~250]  [L0 schema card ~300–600, byte-stable until a contributing fact changes]  ← cached prefix
[conversation history, trimmed to recent turns]
[L1 working memory ~150]  [L3 retrieved memories ≤1,000]  [L4 surfaced memory ≤80]
[user turn]
```

The stable part is prefilled once and cached. The per-turn memory sits just before the user turn,
where small models pay the most attention. To keep this cheap every turn, the brain **prefetches**
(`docs/brain-map.md` #1). After each answer it predicts the next memory block and warms the cache
while you read. On submit it only appends a ≤200-token "late" block of items the prediction
missed. History is trimmed 4 turns at a time so the cached prefix survives. The model only ever reasons over gists and fact lines.
Detail costs an explicit `[recall:<id>]` round-trip.

### L0: schema card (`core.md`, ~300–600 tokens, deterministic, no LLM call)

The card is what a friend has "at hand" before you say a word. It is built in code from the vault
(`brain.vault.generated.build_core`), contains no timestamps, and stays byte-stable until a fact
it shows changes, so the cached prefix survives. Layout, in this order:

```
Name: Alex Rivera · Senior data engineer at Northwind Labs (Data platform team) · lives in Porto · speaks English, Spanish

## Who they are            values, strengths, flaws, habits, fears, health; or "Not known yet: …"
## Now                     active projects (≤3, with goal/status), responsibilities, goals,
                           "On their mind" (recent important journal lines), "New lately" (<90 days)
## Life story              one line per lifetime period, newest first:
                           "2019–2023 · Analytics engineer, Fabrikam Analytics (Lisbon)"; undated last
## People who matter       ≤6, inner circle first: partner, family, close friends, manager, colleagues
## Interests and skills    comma lists: "Skills: Python (expert), Airflow, SQL (+16 more)"
## Open loops              deadlines in the next 30 days, undone intentions, top questions.md items
```

Rules:
- **Group, don't list.** Multi-valued predicates become one comma line per kind, capped, with
  "(+n more)" so the model knows to look further. A job, its roles (each role goes to the job it
  overlaps most) and the city at the time become one period line (Conway's lifetime periods).
- **Salience** inside a list: predicate weight, then stated > observed > inferred, skill level,
  a bonus per extra independent source, newest first on ties.
- **Dedupe:** items whose normalised tokens overlap ≥75% (Jaccard; case, punctuation, plural,
  stop words ignored) are one item and pool their sources; repeated `works_at` spans of the same
  org merge. The threshold is conservative ("Spark" and "Apache Spark" stay two): aliases on
  topic pages are the place to merge names.
- **Shared budget:** each section gets a share of `min(core_max_tokens, 600)`; what a section
  leaves unused goes to the others (Now, Who, People, Open loops, Life story, Interests). List
  rows shrink before they are dropped, and empty sections are left out. `core_max_tokens` is a
  hard cap; the identity line always survives.
- **Gaps are shown:** when values, strengths, flaws or habits are missing, "Not known yet" names
  them, so the local model can ask and interview mode can fill them.
- **Privacy:** `include_private=False` (the variant for the big model) drops private pages, facts
  in private life-map regions (traits, health, family, friends) and journal lines.

Phase 4 consolidation adds an **impression** paragraph (~80 tokens: temperament, communication
style, how they react to stress and praise) written from the observation journal. It **changes
slowly**: a line changes only with ≥3 supporting episodes over ≥2 weeks, and the diff is shown to
the owner (a self-schema should not swing with one bad day). Each version is kept and the diff is
appended to the journal.

#### How people keep a profile of a friend

- **Three levels of knowing a person** (McAdams 1995): dispositional traits; characteristic
  adaptations (goals, values, concerns, current projects); and the life story. → "Who they are",
  "Now" and "Life story".
- **Self-memory system** (Conway & Pleydell-Pearce 2000): autobiographical knowledge is organised
  as lifetime periods ("when I worked at Y, lived in X") → general events → specific episodes,
  and a "working self" holds current goals that steer recall. → periods in the card, episodes in
  retrieval, "Now" as the working self.
- **Trait summaries are stored apart from episodes** (Park 1986; Klein, Loftus, Trafton &
  Fuhrman 1992; Klein, Loftus & Kihlstrom 1996): as we get to know someone, impressions move from
  concrete behaviours to abstract trait summaries that are retrieved without recalling episodes.
  → the card carries the summary; episodes stay in L3.
- **Central traits dominate impressions** (Asch 1946), and **warmth and competence** are the two
  main dimensions of social judgement (Fiske, Cuddy & Glick 2007). → values, strengths and flaws
  come first, before skills.
- **A handful of intimates** (Dunbar's ~5 / ~15 layers; Zhou, Sornette, Hill & Dunbar 2005):
  people track a support clique closely. → ≤6 people, inner circle first.
- **What a friend has at hand:** name and what they do, where they live, who matters to them,
  what they're going through (current chapter, open loops: "how did the go-live go?"), what they
  like, and what to avoid. → identity line, Now, People, Interests, Open loops.

### L1: working memory (updated every turn, no LLM call)
- 4 slots: {topic} + up to 3 {entity | episode}, each with a gist and an activation `a`.
- Every turn: `a ← 0.6·a`. Re-cued slots reset to 1. A new candidate replaces the weakest slot
  if its score is higher. Evicted items go into a **primed** set for 5 turns.
- A one-line **conversation state** ("discussing X; user wants Y; open question Z"), emitted by the
  model as a hidden `<state>` line at the end of each answer. This costs no extra call.
- **Topic boundary:** if `cos(e_turn, topic_centroid) < 0.5` for 2 turns, close the topic, stub an
  episode, and reset the context vector.

### Cues (per turn, target < 30 ms)
1. Alias trie (Aho-Corasick) over every entity name and alias → entity ids.
2. Embed the turn (Qwen3-Embedding-0.6B) → update `c_t`.
3. FTS5 BM25 over the turn's content words.
4. Time expressions → an "as-of" date for bi-temporal filtering ("when I lived in Lisbon", "last spring").
5. Intention triggers.

### L3: activation scoring
Candidates (~100–150) = PPR top-50 ∪ vector top-50 (on `c_t`) ∪ FTS top-30, over fact lines,
episode gists and journal observations.

- **PPR:** damping 0.5 (≈2 hops; HippoRAG default). Seeds: cue entities (confidence /
  ln(1+degree)), L1 entities ×0.5, primed ×0.25, `self` ≤0.05. Edges: wikilinks, fact↔subject/object,
  episode↔entity, episode↔next episode (0.5, temporal contiguity), observation↔entity, plus
  learned co-use weights.
- **Base level:** `B_i = ln Σ_j max(Δt_j, 1h)^-0.5` (Δt in days). Being used in an answer counts as
  a full access; being retrieved but not used counts 0.05 (only real recall strengthens memory).

```
score = 0.40·sim(c_t, i) + 0.25·PPR_i + 0.15·B_i + 0.15·Imp_i + 0.05·primed_i
B_i   = max(B_i, 0.8·Imp_i)        # salience floor: marking memories don't fade
score *= validity_i                 # 1 if valid at the as-of date, 0.3 if superseded (unless the cue is past-tense)
```

All components are normalised to [0,1]. The weights are **priors to be tuned** on the personal eval.

- **Packing:** drop near-duplicates (cos > 0.9 *and* same entities *and* same date window, so two
  similar trips stay two memories) and anything already in L0/L1, then fill
  greedily by score per token up to 1,000 tokens. Fact lines and gists go first; summaries only
  for the top 2 episodes; verbatim never by default.
- **Feeling of knowing:** retrieval returns two signals for the router: *familiarity* (cue
  entities known, with high base-level activation) and *recollection* (top score above a calibrated
  gate, or a fact for the asked predicate found). Familiar-but-not-recalled is "tip of the
  tongue": search deeper, then ask the owner. Personal gaps never go to the big model.

### L4: surfacing ("this reminds me of…")
An episode is surfaced only when **all** of these hold: its score is above the 98th percentile of
historical candidates; it beats the runner-up by ≥0.1; it is not from the last 2 conversations and
was not surfaced in the last 14 days; the turn is not a pure task command (or we are at a topic
boundary); and there has been at most one surfacing per ~6 turns. It is injected as
`Possibly related (mention only if useful): <gist, date>`. Whether you engage with it moves the
threshold up or down. Intentions bypass the gate, because you asked for them.

### Reinforcement and decay
- After each answer: record which memory ids the model cited (with gist↔answer similarity as a
  fallback). Append use timestamps, and strengthen co-use edges (+0.1, capped at 1). This is
  Hebbian learning on the graph. Edges also fade: ×0.97 at each consolidation, pruned below 0.05.
- Decay needs no job: `B` falls with time. Nightly, items whose `B` stayed below the 5th
  percentile for 90 days *and* whose importance is low get archived. They are still searchable,
  never deleted.

## Why this helps a small model

- **No LLM call in retrieval.** Trie, embeddings, sparse PPR and SQLite take about 50 ms. The
  model's effort goes into answering.
- **Small, ranked, deduplicated context.** About 1k tokens of gists instead of pages of raw notes.
  Small models degrade quickly with long, noisy context.
- **Pre-digested knowledge.** The hard thinking (impression, reflections, summaries) happens
  offline in consolidation, where latency doesn't matter and outputs can be reviewed.
- **Explicit "I don't know".** The activation gate gives a calibrated abstention signal that the
  model's own confidence cannot.

## Evaluation (Phase 5)

A personal eval set of 200–300 items, generated from the vault and checked by hand. It covers
single facts, "as of" temporal questions, multi-hop (person → org → project), temperament and
preference questions, unanswerable questions, and **proactive turns** where a gold episode should
surface. Replay is "time-travel": freeze the vault at date T and replay conversations held after T.

- **Metrics:** gold-item recall@budget (no LLM needed), answer accuracy, surfacing precision and
  rate, tokens, latency.
- **Ablations:** without L0 / L1 / PPR / B / importance / salience floor / diversity; decay
  `d ∈ {0.3, 0.5, 0.7}`; damping `∈ {0.3, 0.5, 0.85}`; context vector vs last turn; budgets 1k/2k/4k;
  hand-set weights vs fitted weights.

## Build order

- **Phase 1** (read-only chat): cues (trie + embedding + FTS), similarity + BM25 ranking, L0 from
  `core.md`, packing.
- **Phase 2:** access log + base-level activation, episodes with gists, event boundaries.
- **Phase 3/4:** PPR graph, L1 working memory, surfacing, intentions, co-use edges, nightly
  normalisation.
- **Phase 5:** eval harness, ablations, weight fitting.

## Sources

HippoRAG 2 <https://arxiv.org/abs/2502.14802> · EM-LLM <https://arxiv.org/abs/2407.09450> ·
Synapse <https://arxiv.org/abs/2601.02744> · Spreading activation for KG-RAG <https://arxiv.org/abs/2512.15922> ·
Generative Agents <https://arxiv.org/abs/2304.03442> · MemoryBank <https://arxiv.org/abs/2305.10250> ·
A-MEM <https://arxiv.org/abs/2502.12110> · CoALA <https://arxiv.org/abs/2309.02427> ·
MemGPT <https://arxiv.org/abs/2310.08560> · Nemori <https://arxiv.org/abs/2508.03341> ·
LightMem <https://arxiv.org/abs/2510.18866> · Mastra OM <https://mastra.ai/research/observational-memory> ·
ProMemAssist <https://arxiv.org/abs/2507.21378> · MapDia <https://arxiv.org/abs/2503.05150> ·
Cognitive science: Collins & Loftus 1975; Anderson & Schooler 1991; Tulving & Thomson 1973;
Howard & Kahana 2002; Cowan 2001; Zacks et al. 2007; Reyna & Brainerd (fuzzy-trace theory);
Nader et al. 2000; Anderson, Bjork & Bjork 1994; McClelland, McNaughton & O'Reilly 1995;
Einstein & McDaniel 2005; Schank 1982.
Person knowledge (L0): McAdams 1995 (J. Personality 63:365–396); Conway & Pleydell-Pearce 2000
(Psych. Review 107:261–288); Asch 1946; Park 1986 (JPSP 51:907); Klein, Loftus, Trafton & Fuhrman
1992 (JPSP 63:739–753); Klein, Loftus & Kihlstrom 1996; Fiske, Cuddy & Glick 2007 (TiCS 11:77–83);
Zhou, Sornette, Hill & Dunbar 2005 (Proc. R. Soc. B 272:439–444).
