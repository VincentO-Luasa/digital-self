"""``process_session(..., turns=…)``: a running session processed part by part (an interview
saving its blocks). Fictional persona only."""

from __future__ import annotations

import subprocess
from datetime import date, datetime

from memtools import (  # type: ignore[import-not-found]
    Decider,
    extraction,
    fake,
    segmentation,
    write_transcript,
)

from brain.memory import is_processed, pending_transcripts, process_session
from brain.memory.pipeline import segmented_through
from brain.vault.store import Vault

AT = datetime(2026, 10, 27, 20, 30)
TODAY = date(2026, 10, 27)
TURNS = [
    ("assistant", "How was your week?"),
    ("user", "Fine, thanks."),
    ("assistant", "Where does Atlas stand?"),
    ("user", "Atlas is in the parallel run; we moved the scheduler to Kiln."),
    ("assistant", "And Sam?"),
    ("user", "Sam just started a new job at Contoso."),
]
SAM = extraction(
    [{"subject": "Sam", "predicate": "works_at", "object": "Contoso", "start": "2026-10"}],
    entities=[
        {"name": "Sam", "type": "person", "id": "people/sam-okafor"},
        {"name": "Contoso", "type": "org"},
    ],
)
KILN = extraction([{"subject": "Atlas Migration", "predicate": "uses", "object": "Kiln"}])


def run(vault: Vault, llm, transcript, **kw):
    return process_session(vault, llm, transcript, channel="interview", today=TODAY, **kw)


def commits(vault: Vault) -> list[str]:
    out = subprocess.run(
        ["git", "log", "--format=%s"], cwd=vault.root, capture_output=True, text=True
    )
    return out.stdout.splitlines()


def test_parts_are_segmented_once_each(mvault: Vault) -> None:
    transcript = write_transcript(mvault, TURNS, at=AT, channel="interview")
    _, turns = transcript.read()

    # small talk first: the gate finds nothing; the part is recorded all the same
    quiet = fake([], Decider(personal=0.05))
    report = run(mvault, quiet, transcript, turns=turns[:2], refresh=False)
    assert report.status == "no episode"
    assert segmented_through(mvault, transcript) == 2
    assert "segmented turns 1-2: no episode (gate: p(personal)=0.050" in (
        mvault.log_path.read_text()
    )
    assert not is_processed(mvault, transcript)  # owner turns after the part

    llm = fake([segmentation({"first_turn": 3, "last_turn": 4, "topic": "atlas"}), KILN])
    before = len(commits(mvault))
    report = run(mvault, llm, transcript, turns=turns[:4], refresh=False)
    assert report.status == "processed" and len(report.episodes) == 1
    sent = [c[1]["messages"][1]["content"] for c in llm.calls if c[0] == "structured"]
    assert "[3] ASSISTANT: Where does Atlas stand?" in sent[0]
    assert "[1]" not in sent[0]  # only the part not segmented yet
    # the record is committed with the episode (a crash cannot separate them)
    log = mvault.log_path.read_text()
    assert "segmented turns 3-4: 1 episode(s)" in log
    assert len(commits(mvault)) - before == 2  # episodes + record, then the extraction

    # the run at close: only the rest
    llm = fake([segmentation({"first_turn": 5, "last_turn": 6, "topic": "sam"}), SAM])
    report = run(mvault, llm, transcript)
    assert [e.id for e in report.episodes] == ["ep-20261027-2034-sam"]
    assert segmented_through(mvault, transcript) == 6
    assert is_processed(mvault, transcript) and not pending_transcripts(mvault)
    again = fake([])
    assert run(mvault, again, transcript).status == "already processed" and not again.calls


def test_a_part_without_owner_turns_waits(mvault: Vault) -> None:
    transcript = write_transcript(mvault, TURNS[:1], at=AT, channel="interview")
    _, turns = transcript.read()
    llm = fake([])
    report = run(mvault, llm, transcript, turns=turns, refresh=False)
    assert report.status == "already processed" and not llm.calls
    assert segmented_through(mvault, transcript) == 0  # the question goes with its answer


def test_interview_transcripts_get_their_region_as_hint(mvault: Vault) -> None:
    transcript = write_transcript(mvault, TURNS[2:4], at=AT, channel="interview")
    text = transcript.path.read_text().replace(
        "channel: interview\n", "channel: interview\nregion: projects\n"
    )
    transcript.path.write_text(text)
    llm = fake([segmentation({"first_turn": 1, "last_turn": 2, "topic": "atlas"}), KILN])
    process_session(mvault, llm, transcript, channel="interview", today=TODAY, refresh=False)
    sent = [c[1]["messages"][1]["content"] for c in llm.calls if c[0] == "structured"]
    assert "The owner was interviewed about: projects." in sent[0]
