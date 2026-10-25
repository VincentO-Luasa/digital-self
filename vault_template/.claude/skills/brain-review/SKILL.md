---
name: brain-review
description: Help the owner judge pending memory reviews (proposed facts, supersessions, merges, inferred traits) before they approve or reject them with the `brain review` CLI. Use when the owner asks to go through, check, or discuss pending reviews or a specific review id. You only advise; you never apply, edit, or delete a review.
---

# Help judge pending reviews

Reviews live in `reviews/pending/` and contain the operations the pipeline wasn't allowed to
auto-apply. Only the owner applies them, with `brain review`. Your part is to be a careful second
opinion.

## 1. Get the reviews

You cannot read `reviews/` (and must not try). Ask the owner to paste the reviews they want help
with, as shown by `brain review`.

- Before they paste, remind them once: anything pasted reaches this hosted model's provider, so reviews
  that touch private pages (family, friends, health, traits) are best judged locally.
- Work through one review at a time, highest-impact first: merges, then supersessions, then
  inferred facts, then the rest.

## 2. Check each operation

Use `brain_search` and `brain_get_page` (non-private only) to see the memory it touches. Then apply
the checklist for its type:

**add_fact**
- Is the predicate the right one from the v0.1 vocabulary (see `AGENTS.md`)? `note` is a last resort.
- Does `o` link to an existing entity rather than create a near-duplicate? (`[[orgs/northwind-labs]]`
  rather than the literal "Northwind").
- Does `src` support it? Is `kind` honest? Something the owner said is `stated`. Something you or
  the pipeline concluded is `inferred`.
- Is the valid time plausible (`from` / `to`), and at the right precision?

**supersede_fact**
- Is it really a contradiction for the same `subject` + `p` (+ `o` for multi-valued predicates)?
  Two employers can both be current (`works_at` is multi). A new role usually supersedes the old one.
- Is the `to` date the end of the old fact's validity, not the date the brain learned of the change?
- Is the old fact kept in History rather than lost?

**merge_entities**
- Same real-world entity? Compare aliases, orgs, regions and the facts on both pages.
  "Sam Okafor" at Northwind Labs and "Sam O." in a climbing club may be different people.
- Which id should survive? Usually the one with more facts and links.
- When unsure, recommend `add_alias` or leaving them separate. A wrong merge is much harder to
  undo than a missed one.

**inferred facts (traits, flaws, patterns)**
- List the evidence. Are there at least two independent sources, from different episodes and
  ideally different months?
- Is it a pattern, or one bad week? Is a kinder, more precise wording available? ("Tends to accept
  deadlines before scoping them" beats "Bad at planning".)
- Is there counter-evidence in memory? Search for it explicitly.
- Remember these normally land on private pages. The owner may know context you cannot see.

**update_summary / add_alias / create_entity / add_intention**
- Does the summary contradict any current fact? Does it mention anything you can't cite?
- Is the alias ambiguous with another entity?
- Does the new entity already exist under another name (search its aliases)?
- Does the intention have a sensible cue (`when`) or date (`after`)?

## 3. Give a verdict

For each review give one of:

- **Approve.** One line explaining why.
- **Approve with edits.** Spell out the exact change (predicate, date, wording, target id) so the
  owner can make it in `brain review`.
- **Reject.** Say why (duplicate, wrong entity, weak evidence, not durable).
- **Ask yourself first.** A question only the owner can answer ("Did you leave the data team in
  March or in June?").

Format:

```
r-7k2p · supersede_fact f-9w2mtd (Atlas deadline) → Approve with edits
  `to` should be 2026-10 (when the date moved), not 2026-12.
```

## Rules

- Never run `brain review` yourself, and never edit or delete files in `reviews/`.
- If a pasted review reveals a fact memory should also hold elsewhere, propose it separately with
  `brain_propose`. Don't rewrite the review under discussion.
- Be candid about weak inferences, including unflattering ones that look well supported. Say why.
