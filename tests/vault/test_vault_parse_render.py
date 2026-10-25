from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from brain.vault.models import Episode, Fact
from brain.vault.parse import (
    ParseError,
    parse_episode,
    parse_fact_line,
    parse_intentions,
    parse_journal,
    parse_links,
    parse_page,
)
from brain.vault.render import (
    render_episode,
    render_fact,
    render_frontmatter,
    render_intentions,
    render_journal,
    render_links,
    render_page,
)

FIXTURE_VAULT = Path(__file__).parents[1] / "fixtures" / "vault"
PAGES = sorted((FIXTURE_VAULT / "wiki").rglob("*.md"))
EPISODES = sorted((FIXTURE_VAULT / "episodes").rglob("*.md"))


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.stem)
def test_page_round_trip(path: Path) -> None:
    text = path.read_text()
    page, warnings = parse_page(text, path=path)
    assert warnings == []
    assert render_page(page) == text


@pytest.mark.parametrize("path", EPISODES, ids=lambda p: p.stem)
def test_episode_round_trip(path: Path) -> None:
    text = path.read_text()
    episode, warnings = parse_episode(text, path=path)
    assert warnings == []
    assert render_episode(episode) == text


def test_journal_and_intentions_round_trip() -> None:
    journal = (FIXTURE_VAULT / "journal" / "2026-10.md").read_text()
    observations, warnings = parse_journal(journal)
    assert warnings == [] and len(observations) == 2
    assert render_journal("2026-10", observations) == journal

    text = (FIXTURE_VAULT / "intentions.md").read_text()
    intentions, warnings = parse_intentions(text)
    assert warnings == []
    assert [i.done for i in intentions] == [False, True]
    assert intentions[0].when == ["projects/atlas-migration"]
    assert intentions[0].after == "2026-11-15"
    assert render_intentions(intentions) == text


def test_fact_fields() -> None:
    page, _ = parse_page((FIXTURE_VAULT / "wiki/projects/atlas-migration.md").read_text())
    inferred = page.facts[-1]
    assert inferred.kind == "inferred"
    assert inferred.o == "Timelines slip (twice so far)"
    assert inferred.evidence == [
        "ep-20261025-1412-career",
        "ep-20261026-0915-atlas",
        "we slipped again, sadly",
    ]
    profile, _ = parse_page((FIXTURE_VAULT / "wiki/self/profile.md").read_text())
    role = next(f for f in profile.facts if f.p == "role")
    assert role.src == ["ep-20261025-1412-career", "raw/docs/cv.md"]
    assert role.valid_from == "2025-03"
    skill = next(f for f in profile.facts if f.p == "has_skill")
    assert skill.extra == {"level": "expert"}
    old = profile.history[0]
    assert (old.valid_to, old.by) == ("2025-03", "f-m4n5o6")


def test_external_episode_has_provider() -> None:
    episode, _ = parse_episode((EPISODES[-1]).read_text())
    assert (episode.channel, episode.provider) == ("external", "claude-code")
    assert episode.gist.startswith("Planning the Atlas cut-over:")


def test_fact_line_tolerates_unknown_fields_and_order() -> None:
    fact = parse_fact_line(
        "* Knows Sam (kind:: stated) (o:: [[people/sam-okafor|Sam]]) (p:: knows) (rel:: cousin)"
        " (src:: ep-1) (rec:: 2026-10-25)   ^f-abcdef  "
    )
    assert fact.p == "knows" and fact.extra == {"rel": "cousin"} and fact.src == ["ep-1"]
    assert render_fact(fact) == (
        "- Knows Sam (p:: knows) (o:: [[people/sam-okafor|Sam]]) (rel:: cousin)"
        " (rec:: 2026-10-25) (src:: [[ep-1]]) (kind:: stated) ^f-abcdef"
    )


@pytest.mark.parametrize(
    "line, error",
    [
        ("- Likes tea (p:: likes) (o:: tea)", "block id"),
        ("- Likes tea (p:: likes) (o:: tea) (rec:: 2026-10-25) (kind:: stated) ^f-abcdef", "src"),
        ("- Likes tea (p:: likes) (o:: tea (rec:: 2026-10-25) ^f-abcdef", "unterminated"),
        ("- (p:: likes) (o:: tea) (rec:: 2026) (src:: [[e]]) (kind:: stated) ^f-abcdef", "text"),
        ("- Tea (p:: likes) (o:: tea) (rec:: x) (src:: [[e]]) (kind:: stated) ^f-abcdef", "rec"),
        ("- Tea (p:: likes) (o:: tea) (rec:: 2026-10-25) (src:: [[e]]) (kind:: maybe) ^f-a", "kind"),
        ("- Tea (p:: likes) junk (o:: tea) (rec:: 2026-10-25) (src:: [[e]]) ^f-a", "between"),
    ],
)
def test_malformed_fact_lines(line: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        parse_fact_line(line)


HAND_EDITED = """---
id: people/sam-okafor
type: person
subject: people/sam-okafor
aliases: Sam
tags: [friends, football]
privacy: secret
created: 2026-10-21
updated: 2026-10-21
---


# Sam Okafor


Friend from university.


## Facts


- Likes football (p:: likes) (o:: football) (rec:: 2026-10-21) (src:: [[ep-1]]) (kind:: stated) ^f-abcdef

- Plays on Thursdays, added by hand
- Old (p:: likes) (o:: rugby) (rec:: 2026-10-21) (src:: [[ep-1]]) (kind:: stated) ^custom-id

## Links

- [[orgs/northwind-labs]]

## Notes

Keep this note.
## Not a section, part of notes
"""


def test_tolerant_page_parse_keeps_everything() -> None:
    page, warnings = parse_page(HAND_EDITED)
    messages = " | ".join(w.message for w in warnings)
    assert page.privacy == "private"  # unknown privacy fails closed
    assert "unknown privacy" in messages
    assert "malformed facts line" in messages
    assert "non-canonical fact id" in messages
    assert "unknown section" in messages
    assert page.aliases == ["Sam"]
    assert page.meta == {"tags": ["friends", "football"]}
    assert [f.id for f in page.facts] == ["f-abcdef", "custom-id"]
    assert page.unparsed == {"facts": ["- Plays on Thursdays, added by hand"]}
    assert page.notes.startswith("## Links\n\n- [[orgs/northwind-labs]]")
    assert page.notes.endswith("Keep this note.\n## Not a section, part of notes")

    rendered = render_page(page)
    assert "- Plays on Thursdays, added by hand" in rendered
    assert "tags: [friends, football]" in rendered
    again, _ = parse_page(rendered)
    assert render_page(again) == rendered  # normalised form is stable
    assert again.unparsed == page.unparsed


def test_page_without_frontmatter_uses_path_defaults() -> None:
    path = Path("/v/wiki/projects/garden-shed.md")
    page, warnings = parse_page("Some prose only.\n", path=path, today=date(2026, 10, 25))
    assert (page.id, page.type, page.subject) == (
        "projects/garden-shed",
        "project",
        "projects/garden-shed",
    )
    assert page.title == "Garden Shed"
    assert page.summary == "Some prose only."
    assert page.created == date(2026, 10, 25)
    assert warnings and all(w.path == path for w in warnings)


def test_page_with_wrong_id_uses_path() -> None:
    text = (FIXTURE_VAULT / "wiki/orgs/northwind-labs.md").read_text()
    page, warnings = parse_page(text, path=Path("/v/wiki/orgs/northwind.md"))
    assert page.id == "orgs/northwind"
    assert any("does not match path" in w.message for w in warnings)


def test_missing_optional_sections() -> None:
    text = (
        "---\nid: topics/dbt\ntype: topic\nsubject: topics/dbt\nprivacy: normal\n"
        "created: 2026-10-25\nupdated: 2026-10-25\n---\n# dbt\n"
    )
    page, warnings = parse_page(text)
    assert warnings == []
    assert page.facts == [] and page.summary == "" and page.notes == ""
    assert render_page(page) == text.replace(
        "subject: topics/dbt\n", "subject: topics/dbt\naliases: []\nregions: []\n"
    ) + ("\n## Facts\n")


def test_unparseable_page_raises() -> None:
    with pytest.raises(ParseError):
        parse_page("---\nid: x\n")  # unclosed frontmatter
    with pytest.raises(ParseError):
        parse_page("---\nid: misc/x\n---\n# X\n")  # no type and no known folder


def test_journal_and_intentions_warn_on_bad_lines() -> None:
    journal = (
        "# Journal 2026-10\n\n- 2026-10-25 · imp 7 · ok · [[ep-1]] ^o-abcdef\n"
        "- 2026-10-25 · important · bad ^o-bcdefg\nloose text\n"
    )
    observations, warnings = parse_journal(journal)
    assert len(observations) == 1 and len(warnings) == 2
    intentions, warnings = parse_intentions(
        "# Intentions\n\n- [ ] Ask about Atlas (when:: [[projects/atlas-migration]]) "
        "(priority:: high) ^i-abcdef\n- [ ] no id\n"
    )
    assert len(intentions) == 1
    assert [w.line for w in warnings] == [3, 4]
    assert "priority" in warnings[0].message


def test_links_codec() -> None:
    items = ["ep-1", "raw/docs/cv.md", 'he said "no", twice']
    rendered = render_links(items)
    assert rendered == "[[ep-1]], [[raw/docs/cv.md]], \"he said 'no', twice\""
    assert parse_links(rendered) == ["ep-1", "raw/docs/cv.md", "he said 'no', twice"]


def test_frontmatter_quotes_only_when_needed() -> None:
    fm = render_frontmatter(
        [
            ("gist", "Plain words"),
            ("q", "a: b"),
            ("b", "yes"),
            ("n", "2026-10"),
            ("lst", ["[[x]]", "plain"]),
            ("skip", None),
        ]
    )
    assert fm == (
        '---\ngist: Plain words\nq: "a: b"\nb: "yes"\nn: 2026-10\nlst: ["[[x]]", plain]\n---\n'
    )


def test_render_episode_without_provider_omits_it() -> None:
    episode = Episode(
        id="ep-20261027-0800-x",
        session="conv-1",
        turns=(1, 2),
        channel="chat",
        importance=3,
        gist="g",
        started="2026-10-27T08:00:30",
    )
    text = render_episode(episode)
    assert "provider" not in text and "started: 2026-10-27T08:00:30" in text
    assert parse_episode(text)[0] == episode


def test_render_fact_is_canonical_order() -> None:
    fact = Fact(
        id="f-abcdef",
        text="t",
        p="note",
        o="x",
        to="2026",
        rec=date(2026, 1, 2),
        src=["a"],
        kind="inferred",
        evidence=["b"],
        **{"from": "2025"},
    )
    assert render_fact(fact) == (
        "- t (p:: note) (o:: x) (from:: 2025) (to:: 2026) (rec:: 2026-01-02) (src:: [[a]])"
        " (kind:: inferred) (evidence:: [[b]]) ^f-abcdef"
    )
