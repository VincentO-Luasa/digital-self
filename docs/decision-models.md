# Decision models: where they fit in the brain (v0.1, proposed)

Can "decision models" make the brain's judgement calls (split points, topic boundaries, entity
types, predicate repair, duplicates, routing, importance) faster or more accurate? This page
answers that question. Measurements were taken on 2026-10-25 on the base M4 (32 GB) with
oMLX 0.7.2 and Qwen3.6-35B-A3B-4bit, using fictional inputs (Alex Rivera persona). Sources are at
the end.

## In plain terms

A chat model *writes* an answer token by token. A **decision model** *picks* one: you give it a
state (some text) and typed questions with a fixed answer set, and it returns a probability for
every option and writes no text. There are three question types:

| type | question | answer |
|---|---|---|
| yes/no (`predicate`, Jev calls it `noul`) | "Does block 7 start a new topic?" | P(yes) |
| choice (2–255 options) | "What is *Atlas Migration*: tool, project, org, team, place?" | the pick, a probability per option, confidence |
| score (2–10 ordered levels) | "How important is this episode, 1–5?" | expected level and a distribution |

Why this suits the brain: the answer can never be malformed, it costs one forward pass instead of
a generated JSON reply, and it comes with **a number we can threshold**. A high probability means
code acts. A middle one sends the item to the full LLM or to the owner. A low one means code
rejects. Classical trained classifiers (trees, boosting, fine-tuned encoders) are not pursued,
because they need a labelling and training pipeline the owner does not want to maintain.

## Jev (TypeSafe AI)

**What is public.** TypeSafe AI released Jev (version 1.13) on 15 Sep 2026. It takes a `state`
(text or JSON, text only) and named questions of type choice, score or noul. It answers all of them
in parallel in one pass and never generates text. TypeSafe says it uses a new non-autoregressive
architecture trained with "RLCD" (reinforcement learning for calibrated decisions). Context is 64k
tokens, input costs $0.042 per million tokens and output is free. The stated latency is 70–500 ms
per call from the US West Coast.

**Verified vs claimed.**
- *Verified by third parties:* the API shape, the question types, the price and the context
  limits.
- *Vendor claims only:* "193.6× faster, 444.6× cheaper" comes from TypeSafe's own four-workflow
  eval, and the homepage demo itself works out to about 75× and 171×. Calibration is claimed but
  not independently measured. There are no public benchmark scores. "Zero hallucinations" covers
  only the answer's *shape*. In TypeSafe's words, Jev "guarantees the shape of its answers, not
  that every decision is correct."
- *Documented limits:* dates are read as text, it does no arithmetic, irrelevant state distracts
  it, multi-hop questions and negations are weak, it is not deterministic and English is its main
  language.

**Usable for us?** No. Jev is hosted-only: early-access waitlist, no weights, no on-prem option.
Every one of our decision points reads personal content (document text, facts, chat turns), and
personal content must not leave the machine. There is no meaningful set of *non-personal*
decisions in the pipeline to send it, and its 70–500 ms round-trip is no faster than the local
option below. The same holds for **OpenAI's Decisions API** (`/v1/decisions`, public beta in
Oct 2026, same three question types).

**Open-weight alternatives** have appeared since: Cloudflare **Clef** (27B, 9B "flash", Apache 2.0),
**Decision 3.0** from vLLM Semantic Router (`d3` 27B down to `d3-lite` 0.85B and `d3-nano` 2.2B,
Apache 2.0), **Laya** (ModernBERT 421M; weak zero-shot, its authors recommend fine-tuning) and
community fine-tunes (OpenJev, Kev).

## The local equivalent is already installed

**oMLX 0.7.2 serves decisions locally.** Our server exposes two endpoints:

- **`POST /v1/decisions`** (OpenAI Decisions-compatible). It works with **any chat model**,
  including ours. For each question it lists the options under letters, appends `Answer:`, runs
  *one* token in non-thinking mode and reads the next-token probabilities of the option letters,
  renormalised over the options. If the letters together hold under 50 % of the probability mass,
  it returns a `refusal` ("the model is not answering").
- **`POST /v1/systemone`** (Jev-compatible). It serves dedicated decision models (Clef, OpenJev,
  d3) if their weights are placed in the model directory.

**Logprobs check.** `/v1/chat/completions` accepts `logprobs`/`top_logprobs` but returns
`logprobs: null`. A 1-token chat answer (about 110 prompt tokens) takes 0.55–1.0 s cold and
0.13 s when the prompt is cached, but gives no probabilities. `/v1/decisions` is the way to get
them.

**Measured decisions with the local chat model:**

| decision | result | time |
|---|---|---|
| entity type, 5 names in one passage | project / org / tool / team / place, all correct, confidence 0.94–1.0 | 3.7 s cold, 2.7 s warm (≈0.55 s/question) |
| predicate from a CV line, neutral wording (3 lines) | responsibility 0.94, works_at 1.0, worked_on 0.99, all correct | ≈0.6 s each |
| same predicate check, input framed as "works at: Atlas Migration" | **works_at 0.999, wrong** | 0.66 s |
| "same section?" for two block pairs | 0.89 (same job history) / 0.004 (hobbies), both correct | 0.7 s |
| manager change, dated later | supersedes 0.72, correct with moderate confidence | 0.7 s |
| "lives in Porto" vs "based in Porto, Portugal" | adds 0.60, same 0.37: borderline, should be "same" | 0.8 s |
| needs extraction? (CV / cookie footer / generic tutorial / opinion post) | 0.995 / 0.005 / 0.005 / 0.84 | 0.7–0.8 s |
| new topic in chat? (follow-up / "by the way…") | 0.002 / 0.85 | 0.65 s |
| 16 independent yes/no questions | sequential 0.66 s each, 8 concurrent 0.27 s each | 10.5 s / 4.3 s |

**The reranker as a judge.** `/v1/rerank` takes a custom `instruction` and answers in 0.13–0.16 s
per batch. It judges *relatedness* well: it scored the migration block 0.99 and the hobby block
0.0 against a work block. It cannot tell *same* from *contradicts*: "lives in Porto" scored
0.96–1.0 against "based in Porto", "lived in Lisbon 2019–2023" and "lives in Lisbon" under both
instructions. Under the default instruction it also missed a same-section pair (0.03). Use it to
find candidates, never to judge them.

**Small open decision models: not now.**
- `d3-nano` (2.2B) is 4.5 GB in BF16. Our memory guard leaves about 3 GB, so it would need
  quantising.
- Our chat model is an MoE with about 3B *active* parameters, so a 2B dense model is not much
  faster per token. The gain would be trained calibration, not speed.
- Laya, GLiClass and GLiNER-style encoders (50–420M, fast on CPU/MPS) are zero-shot with generic
  labels, and Laya needs fine-tuning. NLI models (DeBERTa-MNLI) ignore validity time, so
  "lived in Lisbon" vs "lives in Porto" looks like a contradiction.

Revisit a small decision model in Phase 5, scored against the chat model on the personal eval.

## Calibration: how far to trust the numbers

- **Instruction tuning and RLHF make token probabilities overconfident** (Tian et al. 2023;
  Leng et al. 2025). A 0.99 is not a 99 % guarantee.
- **Framing anchors the model.** Showing the extractor's proposed predicate produced a
  confident wrong answer. Questions must describe the *evidence* (the source sentence, the two
  facts) and never the answer under review.
- **Letters are renormalised over the options given.** Every closed set needs an explicit
  `other / unknown` option, or the model is forced to pick something.
- **Use three bands per question type:** `p ≥ hi` → code acts, `lo < p < hi` → fall back (a full
  structured LLM call with reasoning, or the review queue), `p ≤ lo` → reject. Start strict
  (hi 0.95, lo 0.05). Log every decision with its probability and the eventual outcome (review
  accept/reject, "should have escalated"). After about 100 outcomes per type, check a reliability
  table and Brier score, then fit **one temperature per question type**. That is a single number,
  not a training pipeline.

## The decision points

Every call goes through one `brain.llm.decide()` (rule 6) with Pydantic question and answer
types and recorded fixtures. On a runtime without `/v1/decisions` it falls back to a one-token
enum JSON call that gives a label without a probability, treated as mid-band.

**1. Document split points.** *Keep the planning call.*
- Measured on a 20-block outline: one structured plan call took **2.6 s** (378 tokens in, 61 out)
  and returned clean heading-aligned starts.
- The same outline asked as 19 per-gap yes/no questions took **18.3 s** (7,885 input tokens; the
  shared prefix did not visibly cache), and the per-gap probabilities were noisy (0.7–0.8 inside
  one section).
- Splitting is under 1 % of ingest time. No change; at most, use one yes/no question to confirm a
  single ambiguous gap in heading-less text. Phase: none.

**2. Topic boundaries in chat.**
- Questions: `new_topic` (yes/no) on the last 2–3 turns, with an optional `topic_label` choice
  over the open L1 topics.
- It runs after the answer is shown (alongside prefetch), so it adds no latency; it took 0.65 s
  per turn.
- Today's `cos < 0.5 for 2 turns` rule stays as a free pre-filter; the decision settles cases in
  the ambiguous band (cos 0.4–0.65).
- Gain: fewer split or merged episodes, which feed everything downstream. Effort: small.
  Phase 2.

**3. Entity typing.**
- Question: choice over {person, org, team, project, tool/product, place, other} for a *new*
  name, given its sentence.
- Measured: 5/5 correct, about 0.55 s per name, asked only for names not already in the vault.
- Replaces the growing regex for "looks like a team / id / tool". `p ≥ hi` sets the type, the
  middle band goes to review. Effort: small. Phase 2 (with entity resolution).

**4. Predicate repair.**
- Question: choice over the confusable group (works_at / responsibility / worked_on / has_skill
  / other), built from the **source sentence**, not the extracted fact. About 0.6 s each.
- Asked only for facts the normalisation guards already flag (a few per document), so it costs
  seconds per ingest.
- A confident answer applies the repair. Otherwise the fact is dropped or queued as today.
  Effort: small. Phase 2 (ingest, now).

**5. Duplicates and contradictions.**
- Structured facts: code first (same subject, single-valued predicate, overlapping validity →
  conflict).
- Free text (notes, preferences, journal lines): embeddings and the reranker find candidates,
  then a choice question {same, adds, contradicts, supersedes, unrelated} judges each pair.
  About 0.7 s per pair.
- The Porto example shows why the middle band must go to review rather than auto-merge.
  Effort: medium. Phase 2 (predict-calibrate) and Phase 4 (lint).

**6. Escalation routing.**
- Personal gaps stay in code (feeling-of-knowing gate; they never go to the big model).
- A choice question over the pre-route categories (personal recall, advice from my facts, small
  talk, coding, maths, current events, high-stakes, long writing) takes about 0.6 s, so it runs
  only when the kNN pre-route's margin is low.
- A post-answer yes/no "is the draft fully supported by the cited facts?" needs about 1k tokens
  of prefill (about 2–3 s), so it runs only when the answer label is `partial`.
- Learned routers such as RouteLLM need large preference datasets, and their best routers call
  OpenAI embeddings, so they are not an option. Effort: medium. Phase 3.

**7. Importance.**
- Questions: a score question (1–5 rubric from routine to life-changing) plus yes/no questions
  (emotional? future-dated? tied to a goal? asked to remember?) on each episode gist at session
  end.
- The probabilities replace hard labels in the composite importance, which gives a graded floor
  under decay. A few seconds per session, off the critical path. Accuracy is unverified until
  the Phase 5 eval. Effort: small. Phase 2 (segmentation).

**8. Which chunks need extraction.**
- Question: `has_facts` (yes/no) per chunk before extraction. Measured 0.7–0.8 s against roughly
  45 s of generation per chunk.
- It separated the CV (0.995) from the cookie footer (0.005), and it also scored a generic
  tutorial 0.005, which still hints at interests. So skip only below 0.05 and record the skipped
  chunk's heading in the change-set notes.
- First, log how many chunks keep zero facts today; the gain equals that share of ingest time.
  Effort: small. Phase 2 (ingest, now).

## Top 3 by value / effort

1. **Extraction gate (#8).** It is the only change that cuts ingest time, about 45 s for every
   empty chunk skipped, at under 1 s per chunk. It needs one call, one threshold and a log.
2. **Typed checks for predicate repair and entity typing (#3, #4).** They replace hand-written
   regex with probability-gated repairs. Each costs about 0.6 s and is asked only on flagged
   items, so the main gain is accuracy.
3. **Topic boundaries in chat (#2).** Better episodes at no latency cost: about 0.65 s in the
   background per turn.

Where the LLM stays: extraction itself (generation), the split plan, consolidation and anything
that needs reasoning across many facts. Decision questions are for small, local, well-posed
choices.

## Sources

- Jev: [Eden AI](https://www.edenai.co/post/jev-a-new-kind-of-ai-model-built-for-decisions-not-conversation) ·
  [DataNorth](https://datanorth.ai/blog/jev-what-it-is-and-why-this-new-model-matters) ·
  [Thoughtworks](https://www.thoughtworks.com/insights/blog/generative-ai/jev-decision-models-what-need-know-why) ·
  [Wavect review](https://wavect.io/es/blog/jev-ai-decision-model-review.md)
- OpenAI Decisions API: [promptfoo provider docs](https://www.promptfoo.dev/docs/providers/openai-decisions/) ·
  [public beta news](https://letsdatascience.com/news/openai-opens-decisions-api-public-beta-b69a1e40)
- Open decision models: [Cloudflare Clef](https://flaviocopes.com/clef/) ·
  [Decision 3.0 d3](https://huggingface.co/vllm-sr/d3), [d3-nano recipe](https://recipes.vllm.ai/vllm-sr/d3-nano) ·
  [Laya](https://flowtivity.ai/blog/laya-open-source-jev-alternative/) ·
  [OpenJev](https://huggingface.co/AlexWortega/openjev)
- Zero-shot encoders: [GLiNER](https://arxiv.org/abs/2311.08526) ·
  [GLiClass](https://huggingface.co/knowledgator/gliclass-base-v3.0) ·
  [DeBERTa-v3 NLI](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)
- Reranker as yes/no judge: [Qwen3-Reranker (NeMo docs)](https://docs.nvidia.com/nemo/automodel/latest/nemo-automodel/nemo_automodel/components/models/qwen3_reranker/model)
- Calibration: [Tian et al. 2023, "Just Ask for Calibration"](https://arxiv.org/abs/2305.14975) ·
  [Leng et al. 2025, reward calibration in RLHF](https://proceedings.iclr.cc/paper_files/paper/2025/file/29fb6e1456b3d8b57ede5c45aa2c6537-Paper-Conference.pdf)
- Routing: [RouteLLM](https://arxiv.org/abs/2406.18665) and its [README](https://github.com/lm-sys/RouteLLM) (OpenAI embeddings required) ·
  [Hybrid LLM](https://arxiv.org/abs/2404.14618)
- Chunking context: [Is semantic chunking worth the cost?](https://arxiv.org/abs/2410.13070) ·
  [LumberChunker](https://arxiv.org/abs/2406.17526)
- Runtime: oMLX 0.7.2 source (`omlx/api/decisions.py`, `/v1/decisions`, `/v1/systemone`), read locally.
