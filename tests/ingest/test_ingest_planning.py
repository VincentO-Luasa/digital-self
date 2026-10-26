"""Model-planned split of a long document, with scripted plans (no model server)."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import DOCS  # type: ignore[import-not-found]

from brain.ingest import planning
from brain.ingest.chunking import chunk_markdown, join_blocks, split_blocks
from brain.ingest.planning import (
    SplitPlan,
    build_messages,
    long_stretches,
    outline_line,
    plan_chunks,
    windows,
)
from brain.ingest.readers import load_document
from brain.llm import OutputTruncated, StructuredOutputError
from brain.llm.cache import StructuredCache
from brain.llm.fake import FakeLLM
from brain.tokens import estimate_tokens

CAP = 120


def para(topic: str, n: int = 5) -> str:
    return " ".join(f"{topic} detail {i}." for i in range(n))


# headings at blocks 0, 2, 5, 8; ~40 tokens per paragraph; the hobbies are over the cap
DOC = "\n\n".join(
    [
        "# Alex Rivera",
        para("Intro", 2),
        "## Northwind Labs",
        para("Northwind first"),
        para("Northwind second"),
        "## Fabrikam Retail",
        para("Fabrikam first"),
        para("Fabrikam second"),
        "## Hobbies",
        para("Bouldering"),
        para("Bouldering grades"),
        para("Sourdough"),
        para("Sourdough starter"),
    ]
)


def plan(*parts: tuple[int, str]) -> dict:
    return {
        "reasoning": "A profile with jobs and hobbies.",
        "parts": [{"first_block": i, "title": t} for i, t in parts],
    }


def run(*responses, text: str = DOC, cap: int = CAP):
    llm = FakeLLM(structured_responses=responses)
    return plan_chunks(llm, text, doc_title="Alex Rivera", max_tokens=cap), llm


def covers_every_block(report, text: str = DOC, cap: int = CAP) -> bool:
    blocks = split_blocks(text, cap)
    return "\n\n".join(c.text for c in report.chunks) == join_blocks(blocks).strip()


def test_blocks_and_outline() -> None:
    text = "# A\n\nIntro line\nstill intro\n\n```\ncode\n\n# not a heading\n```\n\n- one\n- two"
    blocks = split_blocks(text)
    assert [(b.text, b.heading) for b in blocks] == [
        ("# A", True),
        ("Intro line\nstill intro", False),
        ("```\ncode\n\n# not a heading\n```", False),
        ("- one\n- two", False),
    ]
    assert all(b.trail == ("A",) for b in blocks)
    assert outline_line(blocks[0]) == "[0] # A"
    assert outline_line(blocks[1]) == "[1] (7 tok) Intro line still intro"
    long = split_blocks(para("Long", 60), 100)
    assert len(long) > 1 and all(b.tokens <= 100 for b in long)
    assert "…" in outline_line(long[0])


def test_valid_plan_becomes_chunks() -> None:
    report, llm = run(
        plan((0, "Intro"), (2, "Job at Northwind"), (5, "Job at Fabrikam"), (8, "Hobbies"))
    )
    assert report.planned and report.calls == 1
    assert [(p.title, p.start, p.end) for p in report.parts] == [
        ("Intro; Job at Northwind", 0, 5),  # the intro is under a quarter of the cap: joined
        ("Job at Fabrikam", 5, 8),
        ("Hobbies (1/2)", 8, 12),  # over the cap: split at a paragraph
        ("Hobbies (2/2)", 12, 13),
    ]
    assert report.notes == [
        "plan: part 'Hobbies' (143 tok) split into 2",
        "plan: joined small parts 'Intro' and 'Job at Northwind'",
    ]
    assert covers_every_block(report)
    assert all(p.tokens <= CAP for p in report.parts)
    northwind = next(c for c in report.chunks if "Northwind first" in c.text)
    assert "Northwind second" in northwind.text and "Fabrikam" not in northwind.text
    fabrikam = next(c for c in report.chunks if "Fabrikam first" in c.text)
    assert fabrikam.text.startswith("## Fabrikam Retail") and "Fabrikam second" in fabrikam.text
    assert fabrikam.heading == "Alex Rivera > Fabrikam Retail > Job at Fabrikam"
    # the request: a pinned system prompt and an outline (not the text) in the user message
    messages = llm.calls[0][1]["messages"]
    assert messages[0]["content"].startswith("You split a long document")
    assert "[2] ## Northwind Labs" in messages[1]["content"]
    assert "Token cap per part: 120" in messages[1]["content"]
    assert "Northwind second detail 4" not in messages[1]["content"]  # only first words


def test_plan_is_repaired() -> None:
    # unsorted, a duplicate, no part at 0, one index out of range, a start just after a heading
    report, _ = run(
        plan((6, "Fabrikam"), (3, "Northwind"), (3, "Again"), (99, "Bad"), (9, "Climb"))
    )
    # 3, 6 and 9 move back onto their headings (2, 5, 8); the first title comes from the heading
    assert [(p.title, p.start) for p in report.parts] == [
        ("Alex Rivera; Northwind", 0),
        ("Fabrikam", 5),
        ("Climb (1/2)", 8),
        ("Climb (2/2)", 12),
    ]
    assert covers_every_block(report)
    assert report.notes[:3] == [
        "plan: dropped part starts outside blocks 0-12: [99]",
        "plan: part starts were unsorted or repeated; sorted and deduplicated",
        "plan: no part started at block 0; added one",
    ]


def test_heading_only_part_joins_the_next() -> None:
    report, _ = run(plan((0, "Intro"), (2, "Career"), (3, "Northwind"), (5, "Fabrikam")))
    northwind = next(p for p in report.parts if "Northwind" in p.title)
    assert northwind.start in (0, 2)  # the "## Northwind Labs" heading is not left behind
    assert not any(c.text.rstrip().endswith("## Northwind Labs") for c in report.chunks)


def test_part_over_the_cap_is_split_at_headings_and_paragraphs() -> None:
    report, llm = run(plan((0, "Everything")))
    assert report.planned and len(llm.calls) == 1  # no second model call
    assert len(report.parts) > 2 and all(p.tokens <= CAP for p in report.parts)
    assert any("split into" in n for n in report.notes)
    assert covers_every_block(report)
    assert report.parts[0].title.startswith("Everything (1/")


@pytest.mark.parametrize(
    "response",
    [
        StructuredOutputError("bad json", "{"),
        OutputTruncated("cut off", '{"reasoning"'),
        plan(),  # no parts
        plan((50, "a"), (60, "b"), (0, "c")),  # mostly out of range
    ],
)
def test_failed_plan_falls_back_to_headings(response) -> None:
    report, _ = run(response)
    assert not report.planned and report.calls == 1
    assert [c.text for c in report.chunks] == [c.text for c in chunk_markdown(DOC, CAP)]
    assert report.notes[-1] == "plan: split at headings instead"


def test_small_document_is_not_planned() -> None:
    report, llm = run(text="# Alex Rivera\n\nData engineer at Northwind Labs.")
    assert not llm.calls and not report.planned and len(report.chunks) == 1
    homepage = load_document(DOCS / "alex-rivera.html").text
    report, llm = run(text=homepage, cap=10_000)
    assert not llm.calls and len(report.chunks) == 1


def test_long_stretches_are_named_in_the_request() -> None:
    blocks = split_blocks(DOC, CAP)
    assert long_stretches(blocks, CAP, 0, len(blocks)) == [(8, 13, 142)]
    assert long_stretches(blocks, CAP, 9, 13) == [(9, 13, 139)]
    assert long_stretches(blocks, 10_000, 0, len(blocks)) == []
    user = build_messages(blocks, 0, len(blocks), doc_title="t", total=len(blocks), cap=CAP)[1]
    assert (
        "- blocks 8 to 12 (" in user["content"]
        and "## Hobbies): split into 2 or more" in (user["content"])
    )
    small = build_messages(blocks, 0, 5, doc_title="t", total=len(blocks), cap=CAP)[1]
    assert small["content"].endswith("no heading inside:\n- none")


def test_long_outline_is_planned_in_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    sections = [
        f"## Topic {i}\n\n{para(f'Topic {i} a', 6)}\n\n{para(f'Topic {i} b', 6)}" for i in range(40)
    ]
    text = "\n\n".join(sections)
    monkeypatch.setattr(planning, "WINDOW_TOKENS", 400)
    blocks = split_blocks(text, CAP)
    spans = windows(blocks)
    assert len(spans) >= 3
    assert all(spans[k + 1][0] < spans[k][1] for k in range(len(spans) - 1))  # they overlap
    # each window's plan: a part at every heading it shows (its first block, too)
    responses = [
        plan(*[(b.index, b.text[3:]) for b in blocks[lo:hi] if b.heading or b.index == lo])
        for lo, hi in spans
    ]
    report, llm = run(*responses, text=text)
    assert report.planned and report.calls == len(spans) == len(llm.calls)
    for (lo, hi), call in zip(spans, llm.calls, strict=True):
        user = call[1]["messages"][1]["content"]
        assert f"Outline of blocks {lo} to {hi - 1} of {len(blocks)}" in user
        assert f"[{lo}] " in user and f"[{hi}] " not in user
    headings = [b.index for b in blocks if b.heading]
    assert [p.start for p in report.parts] == headings  # stitched: one part per topic
    assert [p.title for p in report.parts] == [f"Topic {i}" for i in range(40)]
    assert covers_every_block(report, text)


def test_failed_window_is_split_at_headings(monkeypatch: pytest.MonkeyPatch) -> None:
    sections = [f"## Topic {i}\n\n{para(f'Topic {i}', 8)}" for i in range(30)]
    text = "\n\n".join(sections)
    monkeypatch.setattr(planning, "WINDOW_TOKENS", 300)
    blocks = split_blocks(text, CAP)
    spans = windows(blocks)
    responses = [
        plan(*[(b.index, b.text[3:]) for b in blocks[lo:hi] if b.heading]) for lo, hi in spans
    ]
    responses[1] = StructuredOutputError("bad json", "{")
    report, _ = run(*responses, text=text)
    assert report.planned and any("failed" in n for n in report.notes)
    assert all(p.tokens <= CAP for p in report.parts)
    assert covers_every_block(report, text)


def test_the_cache_covers_the_plan_call(tmp_path: Path) -> None:
    response = plan((0, "Intro"), (5, "Job at Fabrikam"), (8, "Hobbies"))
    cached = StructuredCache(FakeLLM(structured_responses=[response]), tmp_path)
    first = plan_chunks(cached, DOC, doc_title="Alex Rivera", max_tokens=CAP)
    again = plan_chunks(cached, DOC, doc_title="Alex Rivera", max_tokens=CAP)
    assert cached.hits == 1 and [c.text for c in again.chunks] == [c.text for c in first.chunks]
    assert list(tmp_path.glob("SplitPlan-*.json"))


def test_schema_puts_reasoning_first() -> None:
    assert list(SplitPlan.model_json_schema()["properties"]) == ["reasoning", "parts"]
    assert estimate_tokens(DOC) > CAP


# A "Side projects" section whose last subsection is followed by a "Career path" section that
# starts with a kicker line ("Career", above its heading): the shape of a real homepage.
SIDE_THEN_CAREER = "\n\n".join(
    [
        "# Alex Rivera",
        "## Side projects",
        "### Tidepool",
        para("Tidepool library"),
        "### A reading tutor for children",
        para("Reading tutor"),
        "Career",
        "## Career path",
        "### Data Engineer at Northwind Labs",
        para("Northwind job"),
        "Education",
        "## Education & toolkit",
        para("Westbridge degree"),
    ]
)


def test_part_label_comes_from_its_own_headings_not_the_previous_section() -> None:
    blocks = split_blocks(SIDE_THEN_CAREER, CAP)
    kicker = next(b.index for b in blocks if b.text == "Career")
    education = next(b.index for b in blocks if b.text == "Education")
    assert blocks[kicker].trail == (
        "Alex Rivera",
        "Side projects",
        "A reading tutor for children",
    )
    report, _ = run(
        plan(
            (0, "Side projects"),
            (kicker, "Data Engineer at Northwind Labs"),
            (education, "Education and certifications"),
        ),
        text=SIDE_THEN_CAREER,
    )
    labels = [c.heading for c in report.chunks]
    assert labels == [
        "Alex Rivera > Side projects",
        "Alex Rivera > Career path > Data Engineer at Northwind Labs",
        "Alex Rivera > Education & toolkit > Education and certifications",
    ]
    assert report.chunks[1].text.startswith("Career\n\n## Career path")


def test_part_label_of_a_continuation_and_of_a_part_spanning_two_sections() -> None:
    blocks = split_blocks(SIDE_THEN_CAREER, CAP)
    voice = next(b.index for b in blocks if b.text.startswith("Reading tutor"))
    # a part with no heading of its own continues its section
    assert planning.section_trail(blocks[voice : voice + 1]) == (
        "Alex Rivera",
        "Side projects",
        "A reading tutor for children",
    )
    # real text of the side project, then the career heading: only what both share
    assert planning.section_trail(blocks[voice : voice + 5]) == ("Alex Rivera",)
