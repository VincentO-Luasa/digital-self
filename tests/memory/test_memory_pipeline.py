"""``process_session`` end to end with a scripted model: episodes, predict-calibrate, owner
corrections, review queue, idempotency, resume, reindex."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from memtools import (  # type: ignore[import-not-found]
    Decider,
    extraction,
    fake,
    segmentation,
    write_transcript,
)

from brain.index.db import open_index
from brain.index.reindex import reindex
from brain.llm.base import StructuredOutputError
from brain.memory import is_processed, pending_transcripts, process_session
from brain.memory.extract import MemoryExtraction, calibrate, known_facts
from brain.vault.store import Vault

AT = datetime(2026, 10, 27, 20, 30)
TODAY = date(2026, 10, 27)
TURNS = [
    ("user", "Update for you: I don't live in Porto any more. I moved to Braga in September."),
    ("assistant", "Got it. Anything else?"),
    ("user", "Sam just started a new job at Contoso. We're celebrating on Saturday."),
    ("assistant", "Congratulations to Sam!"),
]
SEGMENTS = segmentation(
    {
        "first_turn": 1,
        "last_turn": 2,
        "topic": "moved to braga",
        "regions": ["identity"],
        "mentions": [{"name": "Porto", "type": "place"}, {"name": "Braga", "type": "place"}],
        "gist": "Alex moved from Porto to Braga in September",
        "summary": "Alex no longer lives in Porto and moved to Braga in September 2026.",
        "change": "major",
    },
    {
        "first_turn": 3,
        "last_turn": 4,
        "topic": "sam new job",
        "regions": ["relationships.friends"],
        "mentions": [{"name": "Sam", "type": "person"}, {"name": "Contoso", "type": "org"}],
        "gist": "Sam started a new job at Contoso",
        "summary": "Sam started a job at Contoso. Alex and Sam celebrate on Saturday.",
        "emotion": "mild",
        "future": True,
    },
)
MOVE = extraction(
    [
        {
            "subject": "self",
            "predicate": "lives_in",
            "object": "Braga",
            "start": "2026-09",
            "label": "contradicts",
            "known": "f-d5e6f7",
        },
        {
            "subject": "self",
            "predicate": "speaks",
            "object": "English",
            "label": "confirms",
            "known": "f-g2h3i4",
        },
    ],
    entities=[{"name": "Braga", "type": "place"}],
    observations=["Alex finds Braga quieter and cheaper"],
)
SAM = extraction(
    [{"subject": "Sam", "predicate": "works_at", "object": "Contoso", "start": "2026-10"}],
    entities=[
        {"name": "Sam", "type": "person", "id": "people/sam-okafor"},
        {"name": "Contoso", "type": "org"},
    ],
    intentions=[
        {"text": "Ask how Sam's celebration went", "after": "2026-10-31", "about": ["Sam"]}
    ],
)


def run(vault: Vault, llm, transcript, **kw):
    return process_session(vault, llm, transcript, channel="chat", today=TODAY, **kw)


def test_session_becomes_episodes_facts_and_a_correction(mvault: Vault) -> None:
    transcript = write_transcript(mvault, TURNS, at=AT)
    decider = Decider()
    llm = fake([SEGMENTS, MOVE, SAM], decider)
    report = run(mvault, llm, transcript)

    assert report.status == "processed", report.notes
    episodes = [e for e in mvault.episodes() if e.session == transcript.session_id]
    assert [e.id for e in episodes] == [
        "ep-20261027-2030-moved-to-braga",
        "ep-20261027-2032-sam-new-job",
    ]
    assert all(e.status == "extracted" for e in episodes)
    assert episodes[0].importance == 5  # 2 + major change 3
    assert (
        episodes[1].entities[:1] == ["people/sam-okafor"] and "orgs/contoso" in episodes[1].entities
    )

    # the explicit change auto-applied as an owner correction
    _, old, in_history = mvault.find_fact("f-d5e6f7")
    assert in_history and old.valid_to == "2026-09"
    profile = mvault.page("self")
    braga = [f for f in profile.facts if f.p == "lives_in"]
    assert [(f.o, f.valid_from, f.src) for f in braga] == [
        ("[[places/braga]]", "2026-09", ["ep-20261027-2030-moved-to-braga"])
    ]
    # confirms adds the episode as a source, nothing else
    _, english, _ = mvault.find_fact("f-g2h3i4")
    assert english.src == ["raw/docs/cv.md", "ep-20261027-2030-moved-to-braga"]
    # a fact about a friend, a new org page, an intention, a journal line
    sam = mvault.page("people/sam-okafor")
    assert any(f.p == "works_at" and f.o == "[[orgs/contoso]]" for f in sam.facts)
    assert mvault.page("orgs/contoso") is not None
    assert any("celebration" in i.text and i.after == "2026-10-31" for i in mvault.intentions())
    assert any("Braga quieter" in o.text and o.importance == 5 for o in mvault.observations())
    assert report.total("corrected") == 1 and report.total("queued") == 0
    assert "1 owner correction" in report.summary()
    assert [q for q, _ in decider.asked].count("correction") == 1
    log = mvault.log_path.read_text()
    assert "- owner correction applied: ^f-d5e6f7" in log
    assert is_processed(mvault, transcript) and not pending_transcripts(mvault)
    # generated files and the index
    assert "Braga" in mvault.core_path.read_text()
    assert report.reindexed and report.reindexed.episodes >= 2


def test_rerun_does_nothing(mvault: Vault) -> None:
    transcript = write_transcript(mvault, TURNS, at=AT)
    run(mvault, fake([SEGMENTS, MOVE, SAM]), transcript)
    head = mvault.log_path.read_text()
    again = fake([])  # any structured call would fail
    report = run(mvault, again, transcript)
    assert report.status == "already processed"
    assert not [c for c in again.calls if c[0] in ("structured", "decide")]
    assert mvault.log_path.read_text() == head


def test_implicit_contradiction_goes_to_review(mvault: Vault) -> None:
    turns = [("user", "Here in Braga the coffee is half the price. Loving the slower pace.")]
    transcript = write_transcript(mvault, turns, at=AT)
    segments = segmentation(
        {"first_turn": 1, "last_turn": 1, "topic": "braga coffee", "regions": ["identity"]}
    )
    facts = extraction(
        [{"subject": "self", "predicate": "lives_in", "object": "Braga", "label": "new"}]
    )
    report = run(mvault, fake([segments, facts], Decider(correction=0.3)), transcript)
    [ep] = report.episodes
    assert ep.queued == 1 and ep.corrected == 0 and ep.review_id
    assert mvault.find_fact("f-d5e6f7")[2] is False  # Porto still current
    review = (mvault.reviews_pending_dir / f"{ep.review_id}.md").read_text()
    assert "supersede_fact" in review and "f-d5e6f7" in review


def test_a_friend_fact_change_is_never_auto_applied(mvault: Vault) -> None:
    turns = [("user", "Correction: I last saw Sam on the 25th, not the 19th.")]
    transcript = write_transcript(mvault, turns, at=AT)
    segments = segmentation({"first_turn": 1, "last_turn": 1, "topic": "sam"})
    facts = extraction(
        [
            {
                "subject": "Sam",
                "predicate": "last_seen",
                "object": "2026-10-25",
                "label": "contradicts",
                "known": "f-e5f6g7",
            }
        ],
        entities=[{"name": "Sam", "type": "person", "id": "people/sam-okafor"}],
    )
    decider = Decider()
    report = run(mvault, fake([segments, facts], decider), transcript)
    assert report.episodes[0].queued == 1 and report.episodes[0].corrected == 0
    assert "correction" not in [q for q, _ in decider.asked]  # not eligible: not even asked
    assert mvault.find_fact("f-e5f6g7")[2] is False


def test_restated_facts_are_not_duplicated(mvault: Vault) -> None:
    turns = [("user", "As you know I work at Northwind and I lived in Lisbon before Porto.")]
    transcript = write_transcript(mvault, turns, at=AT)
    segments = segmentation({"first_turn": 1, "last_turn": 1, "topic": "recap"})
    facts = extraction(
        [
            {"subject": "self", "predicate": "works_at", "object": "Northwind Labs"},
            {"subject": "self", "predicate": "lives_in", "object": "Lisbon", "end": "2024"},
        ],
        entities=[{"name": "Northwind Labs", "type": "org", "id": "orgs/northwind-labs"}],
    )
    before = sum(len(p.facts) for p in mvault.pages())
    report = run(mvault, fake([segments, facts]), transcript)
    assert report.episodes[0].facts_added == 0 and report.episodes[0].confirmed == 2
    assert sum(len(p.facts) for p in mvault.pages()) == before


def test_ending_without_a_new_value(mvault: Vault) -> None:
    turns = [("user", "Scratch Spark from my skills, I never really used it.")]
    transcript = write_transcript(mvault, turns, at=AT)
    segments = segmentation({"first_turn": 1, "last_turn": 1, "topic": "skills"})
    facts = extraction(
        [
            {
                "subject": "self",
                "predicate": "has_skill",
                "object": "",
                "label": "contradicts",
                "known": "f-s5p6k7",
            }
        ]
    )
    report = run(mvault, fake([segments, facts]), transcript)
    assert report.episodes[0].corrected == 1
    _, spark, in_history = mvault.find_fact("f-s5p6k7")
    assert in_history and spark.valid_to == TODAY.isoformat() and spark.by is None


def test_small_talk_writes_no_episode_and_is_not_asked_again(mvault: Vault) -> None:
    transcript = write_transcript(mvault, [("user", "hey, tell me a joke")], at=AT)
    report = run(mvault, fake([], Decider(personal=0.02)), transcript)
    assert report.status == "no episode"
    assert not [e for e in mvault.episodes() if e.session == transcript.session_id]
    assert "| no episode: gate: p(personal)=0.020" in mvault.log_path.read_text()
    again = fake([], Decider())
    assert run(mvault, again, transcript).status == "already processed"
    assert not again.calls


def test_failed_extraction_resumes_at_the_pending_episode(mvault: Vault) -> None:
    transcript = write_transcript(mvault, TURNS, at=AT)
    broken = StructuredOutputError("bad json", "{")
    report = run(mvault, fake([SEGMENTS, MOVE, broken]), transcript)
    assert report.episodes[1].error and report.status == "processed"
    statuses = {e.id: e.status for e in mvault.episodes() if e.session == transcript.session_id}
    assert list(statuses.values()) == ["extracted", "pending"]
    assert pending_transcripts(mvault)[0].session_id == transcript.session_id
    llm = fake([SAM])
    report = run(mvault, llm, transcript)
    assert [e.id for e in report.episodes] == ["ep-20261027-2032-sam-new-job"]
    assert is_processed(mvault, transcript)


def test_failed_segmentation_leaves_the_session_pending(mvault: Vault) -> None:
    transcript = write_transcript(mvault, TURNS, at=AT)
    report = run(mvault, fake([StructuredOutputError("bad", "")]), transcript)
    assert report.status == "failed" and "segmentation failed" in report.error
    assert not is_processed(mvault, transcript)


def test_reindex_from_markdown_alone_matches(mvault: Vault, tmp_path) -> None:
    transcript = write_transcript(mvault, TURNS, at=AT)
    llm = fake([SEGMENTS, MOVE, SAM])
    run(mvault, llm, transcript, refresh=False)
    db = open_index(tmp_path / "fresh.sqlite", embed_model=llm.embed_model, dim=llm.embed_dim)
    stats = reindex(Vault(mvault.root), db, llm)
    rows = db.execute("SELECT id, kind FROM items WHERE kind = 'episode'").fetchall()
    assert {r["id"] for r in rows} >= {"ep-20261027-2030-moved-to-braga"}
    assert stats.observations == len(mvault.observations())


def test_known_facts_show_the_regions_and_the_mentioned_entities(mvault: Vault) -> None:
    known = known_facts(
        mvault,
        ["people/sam-okafor"],
        regions=["identity"],
        text="I moved from Porto",
        lifemap=mvault.lifemap,
    )
    ids = [k.fact.id for k in known]
    assert "f-d5e6f7" in ids  # lives_in Porto (identity slot and a shared word)
    assert "f-l6s2b3" in ids  # lived in Lisbon: an ended fact, shown as such
    assert "f-y5z6a7" in ids  # Sam's facts
    assert "f-w5x6y7" not in ids  # Atlas: not mentioned


@pytest.mark.parametrize(
    ("label", "known", "obj", "kept", "confirms"),
    [
        ("noop", "", "x", 0, 0),
        ("confirms", "f-d5e6f7", "Porto", 0, 1),
        ("confirms", "f-d5e6f7", "Lisbon", 1, 0),  # not what the known fact says: new
        ("confirms", "f-nonexist", "Porto", 1, 0),
        ("new", "", "Braga", 1, 0),
    ],
)
def test_calibrate(mvault: Vault, label, known, obj, kept, confirms) -> None:
    raw = MemoryExtraction.model_validate(
        extraction(
            [
                {
                    "subject": "self",
                    "predicate": "lives_in",
                    "object": obj,
                    "label": label,
                    "known": known,
                }
            ]
        )
    )
    facts = known_facts(mvault, [], regions=["identity"], text="Porto", lifemap=mvault.lifemap)
    out = calibrate(raw, {k.fact.id: k for k in facts}, {}, [])
    assert (len(out.facts), len(out.confirms)) == (kept, confirms)


INTERVIEW = [
    (
        "system",
        "[interview] region:: projects · topic:: status · about:: [[projects/atlas-migration]] · mode:: ask",
    ),
    ("assistant", "Where does Atlas stand right now?"),
    ("user", "We're in the parallel run until mid-November."),
    ("system", "[interview] region:: self.values_goals · topic:: goals · mode:: ask"),
    ("assistant", "What are you working towards this year?"),
    ("user", "private"),
    (
        "system",
        "[interview] privacy:: private · the owner's next answer is private: never share it with the big model",
    ),
    ("assistant", "Understood. Go ahead."),
    (
        "system",
        "[interview] privacy:: private · the owner's next answer is private: never share it with the big model",
    ),
    ("user", "I want to leave Northwind by next summer and start my own consultancy."),
]


def test_interview_markers_and_private_answers(mvault: Vault, tmp_path) -> None:
    transcript = write_transcript(mvault, INTERVIEW, at=AT, channel="interview")
    segments = segmentation(
        {"first_turn": 1, "last_turn": 3, "topic": "atlas status", "regions": []},
        {
            "first_turn": 4,
            "last_turn": 10,
            "topic": "goals",
            "regions": ["self.values_goals"],
            "gist": "Alex wants to start a consultancy",
        },
    )
    atlas = extraction(
        [
            {
                "subject": "Atlas Migration",
                "predicate": "status",
                "object": "parallel run",
                "label": "new",
            }
        ]
    )
    goals = extraction(
        [
            {
                "subject": "self",
                "predicate": "goal",
                "object": "Start own consultancy",
                "label": "new",
            }
        ],
        observations=["Alex is planning to leave Northwind"],
    )
    llm = fake([segments, atlas, goals])
    report = process_session(
        mvault,
        llm,
        transcript,
        channel="interview",
        today=TODAY,
        private_turns=[10],
        regions_hint=["projects"],
    )
    assert report.status == "processed", report.notes
    # markers are shown to the model as NOTE lines, never as the owner's words
    sent = [c[1]["messages"][1]["content"] for c in llm.calls if c[0] == "structured"]
    assert (
        "[1] NOTE: next question: topic status, region projects, about projects/atlas-migration"
        in sent[0]
    )
    assert "[7] NOTE: the owner's next answer is private" in sent[0]
    assert "[interview]" not in sent[0]
    first, second = [e for e in mvault.episodes() if e.session == transcript.session_id]
    # the marker gives the first episode its region and entity
    assert first.regions == ["projects"] and first.entities == ["projects/atlas-migration"]
    assert first.privacy == "normal" and second.privacy == "private"
    assert "privacy: private" in mvault.episode_path(second).read_text()
    goal = next(f for p in mvault.pages() for f in p.facts if f.p == "goal" and p.subject == "self")
    assert goal.extra == {"privacy": "private"}
    db = open_index(tmp_path / "i.sqlite", embed_model=llm.embed_model, dim=llm.embed_dim)
    reindex(Vault(mvault.root), db, llm)
    rows = {r["id"]: r["privacy"] for r in db.execute("SELECT id, privacy FROM items")}
    assert rows[goal.id] == "private" and rows[second.id] == "private"
    obs = next(o for o in mvault.observations() if o.src == [second.id])
    assert rows[obs.id] == "private"
    from brain.vault.generated import build_core

    assert "consultancy" in build_core(mvault, today=TODAY)
    assert "consultancy" not in build_core(mvault, today=TODAY, include_private=False)


def test_private_marker_alone_marks_the_next_answer(mvault: Vault) -> None:
    from brain.memory.conversation import private_turns
    from brain.vault.raw import Turn

    turns = [Turn(n + 1, r, AT, t) for n, (r, t) in enumerate(INTERVIEW)]  # type: ignore[arg-type]
    assert private_turns(turns) == {10}
    assert private_turns(turns, [3]) == {3, 10}


def test_mangled_entity_id_and_ungrounded_objects(mvault: Vault) -> None:
    turns = [("user", "Correction about Sam: we met at a hackathon in Bristol in 2013.")]
    transcript = write_transcript(mvault, turns, at=AT)
    segments = segmentation(
        {
            "first_turn": 1,
            "last_turn": 1,
            "topic": "sam",
            "mentions": [{"name": "Sam", "type": "person"}],
        }
    )
    facts = extraction(
        [
            {
                "subject": "Sam Okafor",
                "predicate": "met_at",
                "object": "hackathon in Bristol",
                "start": "2013",
                "label": "contradicts",
                "known": "f-b2c3d4",
            },
            # a value that links to a page the episode never names: dropped
            {"subject": "Sam Okafor", "predicate": "likes", "object": "Northwind Labs"},
        ],
        entities=[{"name": "Sam Okafor", "type": "person", "id": "sam-okafor"}],  # bare id
    )
    report = run(mvault, fake([segments, facts]), transcript)
    [ep] = report.episodes
    assert ep.queued == 1, ep.notes  # Sam's met_at supersession, kept despite the bare id
    review = (mvault.reviews_pending_dir / f"{ep.review_id}.md").read_text()
    assert "hackathon in Bristol" in review and "f-b2c3d4" in review
    assert any("never names 'Northwind Labs'" in n for n in ep.notes)


def test_a_correction_of_a_past_job_stays_a_job(mvault: Vault) -> None:
    turns = [("user", "That's wrong: I left Fabrikam in March 2023, not January.")]
    transcript = write_transcript(mvault, turns, at=AT)
    segments = segmentation({"first_turn": 1, "last_turn": 1, "topic": "fabrikam"})
    facts = extraction(
        [
            {
                "subject": "self",
                "predicate": "works_at",
                "object": "Fabrikam Analytics",
                "text": "Left Fabrikam Analytics in March 2023",
                "end": "2023-03",
                "label": "contradicts",
                "known": "f-f6b7a2",
            },
        ]
    )
    report = run(mvault, fake([segments, facts]), transcript)
    assert report.episodes[0].corrected == 1, report.episodes[0].notes
    _, old, _ = mvault.find_fact("f-f6b7a2")
    _, new, _ = mvault.find_fact(old.by)
    assert (new.p, new.valid_from, new.valid_to) == ("works_at", "2019-06", "2023-03")
