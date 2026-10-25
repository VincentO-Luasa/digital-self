# Brain map: how brain systems shape the pipeline

The brain is a digital *self*, so its design borrows from how a real brain divides the work. This
page maps brain systems to components. It records what the neuroscience changed in the design,
and how much evidence stands behind each borrowing.

Evidence tags: **[established]** solid in neuroscience · **[debated]** influential but contested ·
**[analogy]** useful metaphor only. Brain analogies *suggest* designs. Every item below ships
behind a flag and is checked by the Phase 5 ablations.

## The map

| brain system | what it does | in the brain (component) | status |
|---|---|---|---|
| Hippocampus (index theory) [established] | binds episodes, points to cortical content | episodes + `src::` provenance + PPR graph | have |
| Dentate gyrus / CA3 [established] | pattern **separation** at storage, **completion** from partial cues | entity resolution at write time; alias trie + spreading activation at read time | improve |
| Neocortex [established] | slow, structured semantic knowledge | wiki pages, bi-temporal facts | have |
| Sleep replay [established]; what gets replayed [debated] | selective transfer to cortex | **opportunistic, prioritised consolidation** | improve |
| mPFC schema-congruent fast consolidation (Tse 2007) [established in rodents] | what fits existing knowledge integrates fast | **predict-calibrate extraction**: only new or contradicting info is written | new |
| Prefrontal cortex + gating (PBWM) [debated] | holds goals, gates what enters working memory | L1 buffer with a protected goal slot | improve |
| Anterior cingulate: conflict monitoring, expected value of control [established] | decides when to invest more effort | router with **three effort tiers** + a cheap conflict monitor | improve |
| Metamemory: feeling of knowing [established] | "I know that I know it" vs "never heard of it" | **two-signal gate** (familiarity × recollection) | new |
| System 1 / System 2 [framing] | fast intuition vs slow deliberation | local fast → local thinking → big model | improve |
| Salience network [established] | detects what matters, switches internal ↔ task mode | surfacing gate; foreground/background scheduler | improve |
| Amygdala tagging, behavioural tagging [established] | emotional events consolidate more, and boost related weak memories retroactively | **composite importance**, revised later | improve |
| Default mode network: self-reference [established] | self-schema, autobiography | L0 schema card, `wiki/self/` | have; make it slow-changing |
| Default mode network: simulation [debated] | rehearsal, making connections | idle-time reflection (low priority) | new |
| Global workspace / thalamus [theory] | limited stage where modules compete; the winner is broadcast | prompt assembled from a **blackboard** by one selector | new |
| Predictive processing [established for event boundaries] | anticipates input; surprise drives encoding | **prefetch between turns**; surprise feeds importance | new |
| Basal ganglia, dopamine prediction error [established] | learning from reward | router learned from feedback; co-use edges | improve |
| Cerebellum, procedural memory [analogy] | skills that get cheap with practice | interaction preferences, workflow templates | new |
| Reconsolidation [established in rodents, mixed in humans] | recalled memories become editable | confirm stale facts when they come up | improve |
| Active forgetting, synaptic downscaling [forgetting: established; SHY: debated] | generalise, drop noise | decay; nightly downscaling of co-use edges | improve |
| Theory of mind network [established] | models what *others* believe and know | person models + "who knows what" | new |
| Event segmentation [established] | chunks experience at prediction errors | topic boundaries → episodes | have |
| Prospective memory [established] | remember to do something later | `intentions.md` | have |

## What it changed (ranked by benefit / cost on a laptop)

1. **Prefetch between turns (predictive processing), Phase 1.** After answering, predict the next
   retrieval from the conversation context and send a prefill-only request, so the prompt cache
   warms while you read. On submit, add only a small "late" block of new items. History is
   trimmed in chunks so the cache survives. Target time to first token: about 3.7 s → under 1 s.
2. **Predict-calibrate extraction (schema-congruent fast path), Phase 2.** The extractor sees
   what the brain already knows about each entity and outputs only `new`, `contradicts(^f-id)`,
   `confirms(^f-id)` or `noop`. Congruent updates auto-apply; surprises go to review. The result
   is fewer duplicates and explicit supersession.
3. **Feeling-of-knowing gate, Phases 1 and 3.** Two cheap signals choose the action:

   | | recollection high | recollection low |
   |---|---|---|
   | **familiar cue** | answer locally | "tip of the tongue": search deeper; then **ask the owner** and add to `questions.md` |
   | **unfamiliar cue** | answer | personal → "who's X?" (a chance to learn); world knowledge → big model |

   **Correction:** gaps in personal knowledge never go to the big model, because it doesn't know
   your life either.
4. **Opportunistic, prioritised "sleep", Phase 4.** A laptop is asleep or on battery at night, so
   consolidation runs when the machine is idle (≥10 min) *and* on mains power, or when
   "consolidation pressure" (summed importance of unprocessed episodes) gets high. The queue is
   ordered by need × gain (Mattar & Daw), grouped by target page, and checkpointed after every
   item.
5. **Composite, revisable importance, Phase 2.** Categorical labels (emotion, goal link,
   future-dated, explicit "remember this") plus a surprise count, combined in code. This replaces
   a 1–10 number from the model. A strong episode also boosts related episodes from the past 14
   days (behavioural tagging), and actual use is blended in nightly.
6. **Three effort tiers, Phase 3.** T1 local fast → **T2 local thinking mode** (private
   multi-step reasoning stays on the machine) → T3 big model. A conflict monitor in code
   escalates the effort tier: contradictory current facts, a cited superseded fact, or cited facts
   that disagree.
7. **Briefing cards, Phase 4.** During sleep, write ≤120-token cards for tomorrow's likely topics:
   active projects, open loops, upcoming events. Each card cites fact ids and is invalidated when
   any of them changes. Multi-hop synthesis moves offline (sleep-time compute, Lin et al. 2025).
8. **Workspace blackboard + scheduler, Phase 1 contract, Phase 4 scheduler.** Every memory
   source posts entries (content, salience, token cost). One selector fills the prompt budget and
   logs what it showed. Chat preempts background model work.
9. **Confirm stale facts in passing, Phases 2–3.** If an answer uses a fact past its slot's
   `stale_days`, it adds one short check ("Is Atlas still due in March?").
10. **Person models and a learned router, Phases 4–5.** Inferred, private `## Model` sections for
    people you talk about often, plus `told` / `knows_about` facts ("did I tell Sam?"). Answers
    to "how would Sam react?" are labelled as speculation. The router learns from your feedback
    with a contextual bandit.

## Corrections to the earlier design

- The per-turn memory block broke the prompt cache every turn → fixed by #1.
- "Coverage gap → escalate" was wrong for personal facts → #3.
- Importance was one frozen 1–10 number from the model, contradicting our own rule against
  self-rated scores → #5.
- "Nightly" assumed a desktop → #4.
- **L0 must change slowly**: an impression line changes only with ≥3 supporting episodes across
  ≥2 weeks, and the diff is shown to the owner for confirmation (complementary learning systems:
  slow, interleaved learning).
- **Co-use edges must also fade**: multiply by 0.97 nightly and prune below 0.05.
- **Near-duplicate removal is scoped** to the same entities and date window, so two Lisbon trips
  stay two memories (pattern separation).
- **"Retrieved but not used" counts 0–0.1**, not 0.25. Only actual recall strengthens memory.
- **Surprise is a signal**: a turn that contradicts a known fact raises importance and triggers
  fresh retrieval.

## Sources

Tse et al. 2007 · van Kesteren et al. 2012 (SLIMM) · Kumaran, Hassabis & McClelland 2016 ·
Mattar & Daw 2018 <https://pmc.ncbi.nlm.nih.gov/articles/PMC6203620> · Dunsmoor et al. 2015 ·
Shenhav et al. 2013 · Koriat & Levy-Sadot 2001 · Menon & Uddin 2010 · Symons & Johnson 1997 ·
Richards & Frankland 2017 · Tononi & Cirelli (SHY) · Nader et al. 2000 ·
Sleep-time Compute <https://arxiv.org/abs/2504.13171> · Nemori <https://arxiv.org/abs/2508.03341> ·
LightMem <https://arxiv.org/abs/2510.18866> · HippoRAG <https://arxiv.org/abs/2405.14831> ·
EM-LLM <https://arxiv.org/abs/2407.09450> · SwiftSage <https://arxiv.org/abs/2305.17390> ·
PipeRAG <https://arxiv.org/abs/2403.05676> · Agent Workflow Memory <https://arxiv.org/abs/2409.07429> ·
Generative agent simulations of 1,000 people <https://arxiv.org/abs/2411.10109> ·
PILOT router <https://arxiv.org/abs/2508.21141> · Blackboard LLM systems <https://arxiv.org/abs/2507.01701>
