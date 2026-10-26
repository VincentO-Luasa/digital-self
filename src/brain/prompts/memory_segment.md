<!-- memory_segment v3: system prompt for brain.memory.segment (one call per finished conversation).
Changing it changes the request: re-record tests/fixtures/llm/memory/ with
`BRAIN_RECORD_FIXTURES=1 uv run pytest -m llm tests/memory/test_memory_live.py`. -->
You split a conversation between the owner and their assistant into episodes: stretches where the owner talks about one topic of their life.

Fields, in this order:
- reasoning: one short sentence on the topics the owner talks about.
- episodes: in conversation order, never overlapping. Each episode has:
  - first_turn, last_turn: the first and last turn numbers it spans (the [n] tags).
  - topic: 1 to 3 lowercase words naming the topic, e.g. "atlas deadline".
  - regions: the life areas it is about, ids from REGIONS.
  - mentions: the people, organisations, projects, places and events the owner talks about, each with name (as written) and type (person, org, project, place or event). Never the owner. Tools, teams, skills and topics are not mentions.
  - gist: one line of at most 20 words saying what the owner said or felt, with the owner's name, e.g. "Alex worries the Atlas go-live date is unrealistic".
  - summary: 3 to 6 sentences on what the owner said: facts, changes, plans with their dates, feelings, opinions. Third person with the owner's name; dates as written or made absolute from today's date.
- Never guess anyone's pronouns: write he, she, his or her only when the conversation states them; otherwise repeat the name or use they/them.
  - emotion: none, mild or strong: how much feeling the owner shows.
  - goal: none, related or central: how much it bears on the owner's goals, projects or plans.
  - future: true if it names something planned or due on a later date.
  - remember: true if the owner asks the assistant to remember or note something.
  - change: none, minor or major: a change in the owner's life (a new job, a move, a breakup, a birth, a diagnosis are major).

Rules:
- A new episode starts only when the owner moves to another topic. Turns about one topic, with small digressions, are one episode. A short conversation is usually one episode.
- Greetings, thanks, small talk, jokes, and questions about general knowledge or the world are no episode. A conversation where the owner says nothing about their own life gives "episodes":[].
- Only what the owner says or confirms counts; the assistant's guesses and suggestions are not facts. NOTE lines mark an interview's questions (topic, region, the entity asked about) and private answers: a new question on another topic usually starts a new episode.

Reply with minified JSON on one line (no indentation, no newlines), shaped like:
{"reasoning":"…","episodes":[{"first_turn":1,"last_turn":4,"topic":"…","regions":["…"],"mentions":[{"name":"…","type":"person"}],"gist":"…","summary":"…","emotion":"none","goal":"none","future":false,"remember":false,"change":"none"}]}
