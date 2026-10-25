---
name: brain-reflect
description: Run a deep reflection session the owner explicitly asks for, looking for patterns in their own behaviour, habits, decisions or relationships with work (e.g. "what patterns do you see in how I handle deadlines?", "reflect on how my last three projects went"). Uses only non-private memory plus what the owner shares, and ends with inferred-fact proposals backed by evidence, flagged for review. Use only when the owner asks for reflection or pattern-finding about themselves, never on your own initiative.
---

# Reflection session

Reflection turns many episodes into a few honest, well-evidenced observations. It is the most
sensitive thing you do here: the output is about the owner's character, and it ends up in
their self-model.

## 1. Frame the question

- Restate the reflection question in one sentence and agree on its scope: which area (deadlines,
  feedback, collaboration), which period ("since joining Northwind Labs", "the last 6 months").
- Say plainly what you can and can't see: "I only see non-private memory. Your traits page, health
  and personal relationships are private. Share anything from there you'd like me to consider."

## 2. Gather evidence

- `brain_core_profile()` for the current impression.
- `brain_recent_episodes(days, region?)` across the period. For longer periods, call it with a
  larger `days` and with the relevant regions (`career.current_job`, `projects`, `self.routine`).
- 4–8 `brain_search` calls on the theme from different angles. Example for deadlines:
  `"deadline"`, `"slipped"`, `"estimate"`, `"crunch"`, `"pushed back"`, `"shipped early"`, plus
  each project entity (`"Atlas Migration"`), with `as_of` to sample different periods.
- `brain_get_page` on the 1–3 most relevant projects, to read their `## History` (what changed
  and when).

Keep a scratch list as you go: `id · date · one-line what happened`. Aim for 8+ items before
drawing any conclusion. With fewer than 4, say the memory is too thin and stop at step 3 with
questions.

## 3. Find patterns

For each candidate pattern:

- **Support:** at least 2 independent items, from different episodes and ideally different
  months.
- **Counter-evidence:** search for it on purpose ("met the deadline", "said no") and report it.
- **Conditions:** when does it happen and when not (e.g. "only when Sam Okafor isn't on the
  project", "only for migrations, not for analysis work")?
- **Alternative explanations:** workload, org changes, things outside the owner's control.

Rank by strength of evidence, not by how interesting the pattern is. Three solid patterns beat
seven speculative ones.

## 4. Share the reflection

Structure:

1. **Patterns**, each with: the observation (specific, behavioural, kindly worded), the
   evidence ids with dates, any counter-evidence, and the conditions.
2. **Strengths that show up alongside them.** Flaws rarely come without a matching strength.
3. **One or two experiments** the owner could try, if they want.
4. **Questions** that would confirm or refute the weakest pattern.

Tone: direct, warm and honest. Describe behaviour, not character ("you accepted the Atlas date
before scoping the backfill [f-9w2mtd], [ep-20260912-0930-career]" rather than "you're
disorganised"). Don't diagnose, and don't speculate about health or mental state.

Let the owner react before proposing anything. Their corrections are evidence too.

## 5. Propose inferred facts

After the discussion, propose only the patterns the owner didn't reject:

```json
{"op": "add_fact", "subject": "self", "p": "has_flaw",
 "o": "Commits to deadlines before scoping data backfills",
 "text": "Tends to commit to deadlines before scoping data backfills",
 "kind": "inferred",
 "evidence": ["f-9w2mtd", "ep-20260912-0930-career", "ep-20260503-1610-projects", "this conversation: owner agreed, 'that's fair for migrations'"]}
```

- Use `has_strength`, `has_flaw`, `habit`, `values` or `likes`/`dislikes`. Never invent a predicate.
- `kind` is always `inferred`, even when the owner agrees. They confirmed an inference, they
  didn't state it.
- Put every proposal from this session in a single `brain_propose` call, with a rationale that
  starts with `Reflection:` followed by the question.
- Tell the owner these are queued for review, will likely land on a private traits page, and
  that you can't see existing traits, so some may duplicate what they already approved.
- If the owner wants to try an experiment, offer an `add_intention` to check in on it.

## Don't

- Don't reflect without being asked, or widen the scope beyond the agreed question.
- Don't use private material the owner didn't share in this conversation.
- Don't propose a pattern supported only by this conversation.
