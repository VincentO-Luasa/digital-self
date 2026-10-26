# Evaluation sets

Measure a pipeline across many kinds of input instead of tuning it on one document. Everything
here is fictional: the owner is **Maya Lindqvist**, a product designer (a persona separate from
the unit tests' Alex Rivera). Never add real documents or facts.

## Ingest (`brain eval ingest`)

`ingest/docs/` holds six documents about Maya, one per type: a homepage (html, with nav, cookie
banner, eyebrow labels, a projects grid, a footer), a CV (pdf), a first-person "about me" post
(md), a third-person conference speaker bio (html, ingested as `observed`), informal year-in-review
notes (md) and a professional-network data export (txt with CSV sections). They overlap and
sometimes disagree, like real life ("since 2021" vs "Mar 2021", a job ending in January in one and
February in another), and include traps: a project inside a job, a side project, a team inside an
org, a tool (Wayfinder SDK) next to a project (Waypoint), past cities vs the current one, a holiday
city, a friend's city, a hobby that sounds like a skill, other people's facts on the same page.

Three more documents use a second fictional persona, **Theo Marchetti** (a machine learning
engineer), for layouts and genres the Maya set lacks: a homepage whose experience and education
timelines are CSS-grid rows (the date cell before the organisation's heading, CV-style dates such
as "Sep 21 — Feb 23", two rows at one employer, project cards with a label above the heading),
a first-person write-up of an experiment where a few lasting facts sit inside a long story
of events, results, lessons and plans (strict gold: story details are not facts), and an "about"
page that restates the same duties, skills and hobbies in other words in its intro, job list and
"Now" section (one gold fact each: a restatement kept as a second fact is an extra), with
degrees in prose that name their school and a double degree's second school.

`ingest/gold/<doc stem>.yaml` lists what a careful human would extract from each document.

### What is measured

Each document is ingested into a fresh scratch vault (`init_vault`, then `ingest()` with the real
model, date fixed to 2026-10-25), then the wiki pages, current and history facts, and the
operations queued for review are compared with the gold:

| metric | meaning |
|---|---|
| required | recall of `required` gold facts (jobs, roles, dates, current city, education, languages) |
| optional | recall of `optional` gold facts (projects, teams, hobbies, people, skills) |
| precision | extracted facts that match a gold fact (any level, `acceptable` included) / all extracted facts. The others are **unverified**, listed in the JSON for a human to read |
| dates | matched facts with dates in the gold whose dates agree at the gold's precision |
| links | matched degree facts whose `at::` school is the gold's `at` (JSON `link_accuracy` only) |
| types | gold entities whose page has the expected type, of those found (and how many exist) |
| lit→page | gold `literals` (tools, teams, ids, topics) that became pages |
| queued | extracted facts waiting in a review instead of the wiki |
| s | seconds for the whole ingest |

Matching: subject and predicate must agree (the gold may list several predicates, or `alt` forms
such as the relation stated the other way); objects match on tokens after folding case, accents,
punctuation, stop words and light suffixes (80% of the gold's tokens present, and half of the
extracted value's tokens being the gold's, except for free-text predicates like
`responsibility`); entity objects compare through the page title and aliases. Each extracted fact
matches at most one gold fact. Misses come with a reason (`wrong predicate`, `wrong subject`,
`other value`, `subject has no page`, `not extracted`) and the run prints the most common errors.

### Running it

```bash
uv run brain eval ingest                       # all documents, extraction cache on
uv run brain eval ingest --docs cv --docs export
uv run brain eval ingest --json before.json    # full details: matches, misses, extras, notes
uv run brain eval ingest --no-cache            # re-run the model (after a prompt change)
uv run brain eval ingest --keep                # keep the scratch vaults to inspect them
```

The model server must be up. The extraction cache lives in `~/.cache/digital-self/eval-ingest`
(`$XDG_CACHE_HOME` if set): with it, a normalisation change is measured in seconds; a prompt,
schema or chunking change misses the cache by itself. Your vault is never read or written.

Run it before and after any ingest change and keep both JSON files; diff them (`misses`,
`unverified`, `date_problems`) rather than eyeballing the table alone. The model samples at
temperature 0.2, so a `--no-cache` run moves by a few points from run to run.

### Adding a document

1. Write it under `ingest/docs/` (fictional, about Maya, 400–2,500 tokens; any supported reader
   suffix). For a PDF, write the text under `ingest/sources/` and render it with
   `uv run python tests/eval/ingest/make_pdf.py <source.txt> <docs/name.pdf>`.
2. Add `ingest/gold/<stem>.yaml`:

```yaml
doc: maya-new-doc.md          # file name under docs/
kind: stated                  # stated (the owner's own) | observed (written by someone else)
owner: [Maya Lindqvist, Maya] # a page with one of these names is "a page for the owner"
entities:                     # every entity that should get a page, with its type
  - {name: Fernhill Health, type: org, aliases: [Fernhill]}
literals: [Figma, Design Systems team]   # names that must stay values, never pages
required:                     # s: self or a declared entity; p: predicate(s); o: value(s)
  - {s: self, p: works_at, o: Fernhill Health, from: 2021-03, current: true}
optional:
  - {s: Fjord, p: owner, o: self, alt: [{s: self, p: worked_on, o: Fjord}]}
  - {s: self, p: [habit, interest, likes], o: [urban sketching, sketching]}
acceptable:                   # fine if extracted, not expected: count as precise, not in recall
  - {s: Fernhill Health, p: located_in, o: Copenhagen}
```

   `from`/`to` are `YYYY`, `YYYY-MM` or `YYYY-MM-DD` and are checked at that precision;
   `current: true` means the fact must have no end; `at` (degrees) names the school the degree
   is from, checked against the fact's `at::` link (`link_accuracy`, not part of matching);
   `why` is a free comment.
3. `uv run pytest tests/eval` checks the gold parses, uses only vocabulary predicates and
   declared subjects, and that the document reads at a useful size.
