"""The gate and the typed checks around extraction, with scripted decisions (no model server)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import httpx
from conftest import entity, fact  # type: ignore[import-not-found]

from brain.ingest import ingest
from brain.ingest.decisions import (
    ACT_ABOVE,
    GATE,
    SCREEN_EMPLOYERS,
    SCREEN_ENTITIES,
    SCREEN_LEISURE,
    SCREEN_SKILLS,
    SKIP_BELOW,
    check_part,
    compact,
    evidence,
    new_entities,
)
from brain.ingest.extract import Extraction
from brain.llm.decide import ChoiceQuestion, Decision, Question
from brain.llm.fake import FakeLLM
from brain.vault.store import Vault

TODAY = date(2026, 10, 25)

PROFILE = """# Alex Rivera

## Career

Since March 2025 I am a senior data engineer at Northwind Labs. I am part of Data Platform, the
group that owns the warehouse, and I lead the Atlas Migration, which moves nightly jobs to
Lakehouse Cloud. Tidepool is a small library I started in 2022. In 2024 I joined the
Harbor Guild, a rowing club.

- Python and SQL
- Bouldering and chess
"""
TUTORIAL = "## Quick guide: partitioning\n\n" + " ".join(
    f"Partitioning step {i} splits a large table by a key so queries read less data."
    for i in range(20)
)


def choice(name: str, value: str, conf: float, options: Sequence[str]) -> Decision:
    rest = (1 - conf) / (len(options) - 1)
    probs = {o: (conf if o == value else rest) for o in options}
    return Decision(name=name, type="choice", choice=value, probabilities=probs, confidence=conf)


def predicate(name: str, p: float) -> Decision:
    return Decision(
        name=name,
        type="predicate",
        choice="yes" if p >= 0.5 else "no",
        probabilities={"yes": p, "no": 1 - p},
        confidence=abs(2 * p - 1),
    )


# item named in a question → (answer, confidence)
ANSWERS = {
    "Northwind Labs": ("org", 0.99),
    "Harbor Guild": ("org", 0.99),
    "Data Platform": ("team", 0.97),
    "Atlas Migration": ("project", 0.98),
    "Lakehouse Cloud": ("tool", 0.95),
    "Tidepool": ("tool", 0.55),  # flagged, but under the threshold: the extractor's project stays
    "Python": ("skill", 0.99),
    "SQL": ("skill", 0.99),
    "Bouldering": ("hobby", 0.98),
    "chess": ("hobby", 0.6),  # under the threshold: has_skill stays, with a note
}


RELATIONS = {"Northwind Labs": ("employer", 0.98), "Harbor Guild": ("member", 0.95)}


def suspects(instructions: str, items: list[str]) -> list[str]:
    """The items a careful screen would flag: values for the entity screen, hobbies among skills,
    skills among interests (and Tidepool, so that its own question runs)."""
    if instructions.startswith("Which of these names"):
        return [i for i in items if ANSWERS[i][0] in ("team", "tool", "topic")]
    if "works or worked for" in instructions:
        return [i for i in items if RELATIONS[i][0] != "employer"]
    want = "hobby" if instructions.startswith("Which of these is a hobby") else "skill"
    return [i for i in items if ANSWERS[i][0] == want]


def decider(text: str, questions: Sequence[Question]) -> list[Decision]:
    out = []
    for q in questions:
        if q.type == "predicate":
            out.append(predicate(q.name, 0.02 if "partitioning" in text else 0.97))
            continue
        assert isinstance(q, ChoiceQuestion)
        options = [c.value for c in q.choices]
        if q.name == "screen":
            hits = suspects(q.instructions, options[:-1])
            probs = {o: 0.01 for o in options}
            for hit in hits:
                probs[hit] = 0.9 / len(hits)
            probs[options[-1]] = 1 - sum(v for k, v in probs.items() if k != options[-1])
            best = max(probs, key=probs.get)  # type: ignore[arg-type]
            out.append(
                Decision(
                    name=q.name, type="choice", choice=best, probabilities=probs, confidence=0.9
                )
            )
            continue
        item = re.search(r'"(.+?)"', q.instructions).group(1)  # type: ignore[union-attr]
        value, conf = RELATIONS[item] if q.name == "relation" else ANSWERS[item]
        out.append(choice(q.name, value, conf, options))
    return out


def extraction() -> Extraction:
    return Extraction.model_validate(
        {
            "reasoning": "Career and skills.",
            "owner_names": ["Alex Rivera"],
            "entities": [
                entity("Northwind Labs", "org"),
                entity("Data Platform", "org"),
                entity("Atlas Migration", "project"),
                entity("Lakehouse Cloud", "project"),
                entity("Tidepool", "project"),
                entity("Harbor Guild", "org"),
            ],
            "facts": [
                fact("self", "works_at", "Northwind Labs", "", "2025-03"),
                fact("self", "works_at", "Harbor Guild", "Joined the Harbor Guild"),
                fact("self", "member_of", "Data Platform", ""),
                fact("self", "worked_on", "Atlas Migration", "Leads the Atlas Migration"),
                fact("Atlas Migration", "uses", "Lakehouse Cloud", ""),
                fact("self", "worked_on", "Tidepool", "", "2022"),
                fact("self", "interest", "Python", ""),
                fact("self", "has_skill", "SQL", ""),
                fact("self", "has_skill", "Bouldering", "Skilled in bouldering"),
                fact("self", "has_skill", "chess", ""),
            ],
        }
    )


def test_evidence_is_the_sentence_or_line_naming_the_item() -> None:
    assert (
        evidence(PROFILE, "Tidepool") == "## Career\nTidepool is a small library I started in 2022."
    )
    assert evidence(PROFILE, "chess") == "## Career\n- Bouldering and chess"
    assert evidence(PROFILE, "Westbridge") == PROFILE  # not named: the whole text
    both = evidence(PROFILE, ["Tidepool", "chess"])
    assert both == (
        "## Career\nTidepool is a small library I started in 2022.\n- Bouldering and chess"
    )


def test_gate_sees_headings_and_first_words() -> None:
    view = compact(PROFILE, words=5)
    assert view.splitlines() == [
        "# Alex Rivera",
        "## Career",
        "Since March 2025 I am …",
        "- Python and SQL - …",
    ]


def test_new_entities_leave_out_known_owner_and_values() -> None:
    names = new_entities(
        extraction(),
        known=["Northwind Labs"],
        owner=["Alex Rivera"],
        literal={"tidepool": "a tool"},
    )
    assert names == [
        ("Data Platform", "org"),
        ("Atlas Migration", "project"),
        ("Lakehouse Cloud", "project"),
        ("Harbor Guild", "org"),
    ]


def test_checks_override_only_confident_answers() -> None:
    llm = FakeLLM(decider=decider)
    ex = extraction()
    checks = check_part(llm, PROFILE, ex, owner=["Alex Rivera"])
    assert checks.kinds == {"data platform": "team", "lakehouse cloud": "tool"}
    # screens: entities, skills; 3 kinds, 2 relations (two employers: asked directly), 3 uses
    assert checks.overridden == 2 and checks.asked == 10
    predicates = {f.object: f.predicate for f in ex.facts if f.subject == "self"}
    assert predicates["Python"] == "has_skill"  # interest → has_skill
    assert predicates["Bouldering"] == "interest"  # has_skill → interest, text re-rendered
    assert predicates["chess"] == "has_skill"  # 0.6 < ACT_ABOVE: kept
    assert next(f for f in ex.facts if f.object == "Bouldering").text == ""
    assert predicates["Harbor Guild"] == "member_of"  # a club joined, not an employer
    assert checks.rerouted == 3
    notes = "\n".join(checks.notes)
    assert "'Tidepool' kind tool (conf 0.55) under 0.8: kept the extractor's project" in notes
    assert "'Data Platform' is a team (conf 0.97), not an org: a value, no page" in notes
    assert "'chess' looks like a hobby (conf 0.60) under 0.8: kept has_skill" in notes
    # questions show the evidence, never the extractor's proposal: fixed screens and templates
    questions = [q for call in llm.calls for q in call[1]["questions"]]
    screens = {
        SCREEN_ENTITIES.instructions,
        SCREEN_EMPLOYERS.instructions,
        SCREEN_SKILLS.instructions,
        SCREEN_LEISURE.instructions,
    }
    for q in questions:
        assert q.instructions in screens or re.fullmatch(
            r"(What kind of thing is|In the text, is|In the text, what is the person\'s relation to) "
            r'"[^"]+"( in the text\?| a skill .*\?|\?)',
            q.instructions,
        ), q.instructions
    assert [q.name for q in questions].count("screen") == 2  # one round each
    inputs = [call[1]["text"] for call in llm.calls]
    assert all(len(t) < len(PROFILE) for t in inputs)  # sentences naming the items, not the part
    assert "Data Platform" in notes and "→ asking about" in notes
    assert ACT_ABOVE == 0.8


def test_ingest_applies_checks_and_records_every_decision(
    fresh_vault: Vault, tmp_path: Path
) -> None:
    doc = tmp_path / "profile.md"
    doc.write_text(PROFILE)
    llm = FakeLLM(structured_responses=[extraction()], decider=decider)
    report = ingest(doc, fresh_vault, llm, today=TODAY)
    assert not [c for c in llm.calls if c[1].get("questions") == [GATE]]  # one part: no gate
    pages = {p.id for p in fresh_vault.pages()}
    assert {"orgs/northwind-labs", "projects/atlas-migration", "projects/tidepool"} <= pages
    assert not any("data-platform" in p or "lakehouse" in p for p in pages)
    self_facts = {(f.p, f.o) for p in fresh_vault.pages() if p.subject == "self" for f in p.facts}
    assert ("team", "Data Platform") in self_facts  # member_of a team → team
    assert ("interest", "Bouldering") in self_facts and ("has_skill", "Python") in self_facts
    notes = report.normalised.notes
    assert any(n.startswith("check: 'Lakehouse Cloud' is a tool") for n in notes)
    [item] = report.chunks
    assert item.checks == 10 and item.changed == 5
    assert ("member_of", "[[orgs/harbor-guild]]") in self_facts and item.check_seconds >= 0


def two_parts() -> tuple[str, dict]:
    text = PROFILE + "\n" + TUTORIAL + "\n"
    plan = {
        "reasoning": "A profile, then a tutorial.",
        "parts": [{"first_block": 0, "title": "Career"}, {"first_block": 4, "title": "Guide"}],
    }
    return text, plan


def test_gate_skips_a_part_with_nothing_about_a_person(fresh_vault: Vault, tmp_path: Path) -> None:
    text, plan = two_parts()
    doc = tmp_path / "page.md"
    doc.write_text(text)
    seen: list[str] = []
    llm = FakeLLM(structured_responses=[plan, extraction()], decider=decider)
    report = ingest(
        doc,
        fresh_vault,
        llm,
        today=TODAY,
        on_chunk=lambda item, total: seen.append(f"{item.skipped} {item.p_about}"),
    )
    assert len(report.plan.chunks) == 2
    assert [c.chunk.index for c in report.skipped] == [1]
    assert report.skipped[0].p_about == 0.02 < SKIP_BELOW
    assert seen == ["False 0.97", "True 0.02"]
    assert len([c for c in llm.calls if c[0] == "structured"]) == 2  # the plan + one extraction
    assert any(
        n.startswith(
            "gate: skipped Alex Rivera > Quick guide: partitioning > Guide: p(about a person)=0.020 < 0.15"
        )
        for n in report.normalised.notes
    )


def test_failed_or_refused_decisions_never_skip_or_change(
    fresh_vault: Vault, tmp_path: Path
) -> None:
    text, plan = two_parts()
    doc = tmp_path / "page.md"
    doc.write_text(text)
    empty = {"reasoning": "r", "owner_names": [], "entities": [], "facts": []}
    # the gate of part 1 fails, the rest are refusals (FakeLLM's unscripted answer)
    llm = FakeLLM(
        structured_responses=[plan, extraction(), empty],
        decisions=[httpx.ConnectError("server gone")],
    )
    report = ingest(doc, fresh_vault, llm, today=TODAY)
    assert not report.skipped and len(report.chunks) == 2
    notes = report.normalised.notes
    assert any("no decision (server gone)" in n for n in notes)
    assert any(n.startswith("gate: ") and n.endswith("Guide: no answer; extracted") for n in notes)
    assert "orgs/data-platform" in {p.id for p in fresh_vault.pages()}  # the extractor stands


def test_speaks_a_non_language_is_dropped_on_a_confident_no() -> None:
    ex = Extraction.model_validate(
        {
            "reasoning": "r",
            "owner_names": [],
            "entities": [],
            "facts": [
                fact("self", "speaks", "Spanish", ""),  # a known language: not asked
                fact("self", "speaks", "email", "Can be reached by email"),
                fact("self", "speaks", "Ladino", ""),
            ],
        }
    )
    llm = FakeLLM(
        decider=lambda text, qs: [predicate(q.name, 0.03 if "email" in text else 0.9) for q in qs]
    )
    checks = check_part(llm, "I speak Spanish and Ladino. Reach me by email.", ex)
    assert [f.object for f in ex.facts] == ["Spanish", "Ladino"]
    assert checks.asked == 2 and checks.rerouted == 1
    assert "check: 'email' is not a language (p=0.03): dropped speaks" in checks.notes


def test_facts_about_entities_the_part_never_names_are_dropped() -> None:
    """On a part about something else the extractor restated facts about the known entities:
    a fact needs its entities named in the excerpt (a capitalised word of the name will do)."""
    text = (
        "## Privacy\n\nThis site stores one preference in your browser. Reach me by email.\n\n"
        "I still row with the Harbor\n  Guild and go to Westbridge alumni dinners."
    )
    ex = Extraction.model_validate(
        {
            "reasoning": "r",
            "owner_names": [],
            "entities": [entity("Bouldering", "project")],
            "facts": [
                fact("self", "works_at", "Porto library club", ""),
                fact("self", "worked_on", "Bouldering", ""),
                fact("Sam Okafor", "friend_of", "self", ""),
                fact("self", "member_of", "Harbor Guild", ""),  # named across a line break
                fact("self", "studied_at", "Westbridge University", ""),  # "Westbridge"
                fact("self", "has_skill", "Python", ""),  # a value: not checked
            ],
        }
    )
    checks = check_part(FakeLLM(), text, ex, owner=["Alex Rivera"])
    assert [f.object for f in ex.facts] == ["Harbor Guild", "Westbridge University", "Python"]
    assert checks.notes[0] == (
        "check: dropped 3 fact(s) about entities the part never names: "
        "works_at 'Porto library club', worked_on 'Bouldering', friend_of 'Sam Okafor'"
    )


def test_values_the_part_never_names_skip_the_screen() -> None:
    text = "## Weekends\n\nI bake bread and go climbing. At work: Python, SQL, Spark and dbt."
    ex = Extraction.model_validate(
        {
            "reasoning": "r",
            "owner_names": [],
            "entities": [],
            "facts": [
                fact("self", "has_skill", v, "")
                for v in ("Python", "SQL", "Spark", "dbt", "bouldering")
            ],
        }
    )
    llm = FakeLLM(decider=decider_for({"bouldering": ("hobby", 0.97)}))
    check_part(llm, text, ex)
    asked = [(q.name, c[1]["text"]) for c in llm.calls for q in c[1]["questions"]]
    # bouldering is not named: asked directly, on the whole part; the four named skills screened
    assert [name for name, _ in asked] == ["screen", "use"]
    assert asked[1][1] == text
    assert next(f for f in ex.facts if f.object == "bouldering").predicate == "interest"


def decider_for(answers: dict[str, tuple[str, float]]):
    def decide(text: str, questions: Sequence[Question]) -> list[Decision]:
        out = []
        for q in questions:
            options = [c.value for c in q.choices]  # type: ignore[union-attr]
            if q.name == "screen":
                probs = {o: 0.0 for o in options}
                probs[options[-1]] = 1.0
                out.append(
                    Decision(
                        name=q.name,
                        type="choice",
                        choice=options[-1],
                        probabilities=probs,
                        confidence=1.0,
                    )
                )
                continue
            item = re.search(r'"(.+?)"', q.instructions).group(1)  # type: ignore[union-attr]
            out.append(choice(q.name, *answers[item], options))
        return out

    return decide
