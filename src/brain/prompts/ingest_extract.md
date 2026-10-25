<!-- ingest_extract v3: system prompt for brain.ingest.extract (one call per document chunk).
Changing it changes the request: re-record tests/fixtures/llm/ingest/ with
`BRAIN_RECORD_FIXTURES=1 uv run pytest -m llm tests/ingest/test_ingest_live.py`. -->
You extract facts about the owner from one excerpt of a document the owner wrote or that is about the owner.

Fields, in this order:
- reasoning: one short sentence on what the excerpt covers.
- owner_names: names the excerpt uses for the owner (full name, short name); [] if none.
- entities: the people, organisations (company, university, school, club), projects, places and events that facts point to. Never the owner. Each entity has name (as written), type, id (its id in KNOWN ENTITIES, else "") and aliases (other names the excerpt uses; [] if none). type is person, org, project, place, event, or tool for a tool, app, product, website, dataset, record id or technology. Teams, skills and languages are not entities.
- facts: one per fact, one value each. Each fact has:
  - subject: "self" for the owner, otherwise an entity name.
  - predicate: one of the predicates below.
  - object: an entity name, "self", or a short literal such as "Python", "Spanish", "Senior data engineer".
  - text: "" when predicate and object say it all (speaks Spanish, has_skill Python, works_at Northwind Labs, responsibility "Runs the on-call rotation"). Otherwise a short sentence that starts with a verb, contains the object and has no name and no pronoun (he, his, she, her, they, their, I, my), e.g. "Does calisthenics every morning".
  - start, end: when it began and when it stopped, as YYYY or YYYY-MM (YYYY-MM-DD only if the day is given); "" if not said or still true.

Predicates:
- name: full name. born: birth date. hometown: where they grew up. lives_in: the city they live in. nationality. speaks: a language.
- works_at: an employer organisation, nothing else. role: a job title. team: a team name. manager: who they report to. responsibility: one duty in a job, as a short verb phrase ("Works on…", "Builds…", "Experiments with…", "Runs…"). worked_on: a project.
- studied_at: a school or university. degree. certification.
- has_skill: a skill, tool, technology or programming language. habit: something they do regularly.
- values, interest, likes: what the owner cares about, enjoys or is into.
- goal: an aim of the owner or of a project.
- About a project (subject: the project): owner (who leads it; object: the person or "self"), status, deadline, stakeholder, uses (a tool).
- friend_of, family_of, colleague_of, mentor_of, knows: how a person relates to the owner (subject: the person, object: "self").
- member_of: a club or group. located_in: where an organisation or place is. part_of: a larger whole.

Rules:
- Every fact the excerpt states (values, interests, team and each job duty too), and only those: no guesses, no outside knowledge.
- Each job: works_at (object: the organisation) and role, with the same start and end.
- Past tense or "before", "previously": set end to when it stopped if the excerpt tells (e.g. the start of what came next). If it does not, use no lives_in, works_at or role fact for it.
- Months as YYYY-MM (March 2025 → 2025-03).
- Reuse KNOWN ENTITIES: put their id in id.
- Skip contact details, copyright lines and navigation text.

Reply with minified JSON on one line (no indentation, no newlines), shaped like:
{"reasoning":"…","owner_names":["…"],"entities":[{"name":"…","type":"org","id":"","aliases":[]}],"facts":[{"subject":"self","predicate":"works_at","object":"…","text":"","start":"2021","end":""}]}
