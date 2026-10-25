---
name: brain-answer
description: Answer a question the local brain escalated to Claude ("→ Ask Claude" handoff packet), or any question whose answer depends on what the brain remembers about its owner. Covers reading the packet, pulling memory through the brain MCP tools, answering with cited fact/episode ids kept apart from general knowledge, proposing new durable facts, and suggesting follow-up intentions. Use at the start of every escalated session and whenever the owner asks for advice, a plan, or a decision grounded in their own situation.
---

# Answer an escalated question

Work through these steps in order. Skip a step only when it clearly doesn't apply.

## 1. Read the packet

- Pull out: the **question**, the **reason** for escalation (`needs:` reasoning / world_knowledge /
  fresh_info / expertise, or a category such as high-stakes or long-form), the **shared memory**
  lines with their ids, and the **local draft**.
- Items marked `[private, confirmed]` may be used in this conversation. Never look up more private
  content. MCP won't return it, and you must not try to get it another way.
- Read the local draft as a hypothesis. Note what it got right, what it missed, and what it made up.
- No packet? Take the owner's first message as the question.

## 2. Load the impression

Call `brain_core_profile()` once. It gives you the owner's temperament, current situation, marking
facts and open loops. Let it set your tone and framing (e.g. if Alex Rivera tends to over-commit
under deadline pressure, plan with that in mind).

## 3. Targeted retrieval

Use 2–5 short `brain_search` calls rather than one broad one:

- one per key entity in the question (`"Atlas Migration"`, `"Sam Okafor"`, `"Northwind Labs"`);
- one for the topic or decision (`"go-live deadline"`, `"push back on scope"`);
- `as_of: "YYYY-MM-DD"` for questions about the past ("when I was on the data team…");
- `brain_get_page(id)` when one entity is central and you need its full Facts + History;
- `brain_recent_episodes(14)` when the question is about "lately", a mood, or something ongoing.

Stop when new searches bring back nothing new. Note what you expected to find and didn't. Those
are the gaps.

## 4. Think, then answer

Use the full depth the reason asked for (proper reasoning, real expertise, current information
via web search with **generic** queries that contain no personal details).

Structure the reply:

1. **Answer / recommendation** in 1–3 sentences.
2. **Why, given your situation.** Memory-grounded reasoning, every owner-specific claim cited:
   "Atlas go-live is 2026-11-15 [f-9w2mtd] and you're the cut-over lead [f-k3x9q2], so…"
3. **General knowledge.** Expertise, frameworks and outside facts, clearly labelled as not coming
   from memory.
4. **What I don't know.** Gaps that would change the answer, with one concrete question.
5. **Next step.** One concrete action.

Rules:
- Never state an inference about the owner as a memory fact. Write "I suspect…, because [ids]".
- Where memory and the local draft disagree, memory wins. Where the owner and memory disagree,
  the owner wins (then propose a supersession in step 5).
- Long-form deliverables (a plan, a letter, an outline) go in the reply. Never write files in the vault.

## 5. Propose what you learned

During the conversation, collect durable new information: decisions made, changed facts, new
people or projects, stated preferences. At a natural pause:

1. `brain_search` each candidate to check it isn't already known (or is known differently).
2. Build the operations (see the reference in `CLAUDE.md`):
   - a new fact → `add_fact` with `kind: stated` and the owner's words quoted in the rationale;
   - a changed fact → `supersede_fact` on the old id, with `to` set to when it stopped being true;
   - a new entity → `create_entity` first, then its facts;
   - a pattern you noticed → `add_fact` with `kind: inferred` and `evidence` (≥2 ids or quotes).
3. Call `brain_propose(operations, rationale)` once per topic. Keep the rationale short:
   what changed, and from which turn of this conversation.
4. Tell the owner in one line: "Proposed 2 updates (review r-…). Approve with `brain review`."

Example for Alex Rivera, who just said the go-live moved:

```json
[{"op": "supersede_fact", "fact_id": "f-9w2mtd", "to": "2026-10",
  "new": {"p": "deadline", "o": "2026-12-01", "text": "Atlas go-live moved to 2026-12-01", "kind": "stated"}}]
```

Rationale: `Alex: "leadership agreed to move go-live to December 1st".`

## 6. Suggest a follow-up intention

If the conversation leads to a future event or decision (a meeting, a deadline, "I'll try X for
two weeks"), offer to add one intention so the local brain brings it up later:

```json
{"op": "add_intention", "text": "Ask how the Atlas go-live went", "when": "[[projects/atlas-migration]]", "after": "2026-12-02"}
```

Propose it only if the owner agrees. One intention per conversation is usually enough.

## Don't

- Don't read vault files directly, or grep the folder, to "check" something.
- Don't pad the answer with every fact you found. Cite only what supports the answer.
- Don't propose transient feelings, hypotheticals, or facts about other people that the owner
  only guessed at.
