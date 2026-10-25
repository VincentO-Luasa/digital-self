"""Render models to canonical vault markdown. Deterministic: ``render(parse(x)) == x``."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

import yaml

from brain.vault.models import Episode, Fact, Intention, Observation, Page

_ID_LIKE = re.compile(r"^[^\s,\"\[\]|#^]+$")
_YAML_UNSAFE = set('[]{},"#\n\t')


def render_links(items: Iterable[str]) -> str:
    """Ids become ``[[id]]``; free text (e.g. a quote used as evidence) is double-quoted."""
    out = []
    for item in items:
        if _ID_LIKE.match(item):
            out.append(f"[[{item}]]")
        else:
            out.append('"' + item.replace('"', "'") + '"')
    return ", ".join(out)


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, datetime):
        if value.second or value.microsecond:
            return value.isoformat(timespec="seconds")
        return value.strftime("%Y-%m-%dT%H:%M")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool | int | float):
        return json.dumps(value)
    text = str(value)
    return text if _is_plain(text) else json.dumps(text, ensure_ascii=False)


def _is_plain(text: str) -> bool:
    if not text or text != text.strip() or _YAML_UNSAFE & set(text):
        return False
    if text[0] in "!&*|>'%@`-?:" or ": " in text or text.endswith(":"):
        return False
    try:
        return yaml.safe_load(text) == text
    except yaml.YAMLError:
        return False


def _scalar_like(value: Any) -> bool:
    return isinstance(value, str | int | float | date)


def _frontmatter_line(key: str, value: Any) -> str:
    if isinstance(value, list | tuple) and all(_scalar_like(v) for v in value):
        return f"{key}: [" + ", ".join(_yaml_scalar(v) for v in value) + "]"
    if _scalar_like(value):
        return f"{key}: {_yaml_scalar(value)}"
    dumped = yaml.safe_dump(
        {key: value}, sort_keys=False, allow_unicode=True, default_flow_style=None, width=1000
    )
    return dumped.rstrip("\n")


def render_frontmatter(items: Iterable[tuple[str, Any]]) -> str:
    """``---`` block in the given key order; ``None`` values are omitted."""
    lines = [_frontmatter_line(k, v) for k, v in items if v is not None]
    return "---\n" + "\n".join(lines) + "\n---\n"


def render_fact(fact: Fact) -> str:
    fields: list[tuple[str, str | None]] = [("p", fact.p), ("o", fact.o)]
    fields += list(fact.extra.items())
    fields += [
        ("from", fact.valid_from),
        ("to", fact.valid_to),
        ("rec", fact.rec.isoformat()),
        ("src", render_links(fact.src)),
        ("kind", fact.kind),
        ("evidence", render_links(fact.evidence) if fact.evidence else None),
        ("by", fact.by),
    ]
    inline = " ".join(f"({k}:: {v})" for k, v in fields if v)
    return f"- {fact.text} {inline} ^{fact.id}"


def render_page(page: Page) -> str:
    fm = [
        ("id", page.id),
        ("type", page.type),
        ("subject", page.subject),
        ("aliases", page.aliases),
        ("regions", page.regions),
        ("privacy", page.privacy),
        ("created", page.created),
        ("updated", page.updated),
    ]
    out = render_frontmatter(fm + list(page.meta.items())) + f"# {page.title}\n"
    if page.summary:
        out += f"\n{page.summary}\n"
    facts = [render_fact(f) for f in page.facts] + page.unparsed.get("facts", [])
    out += "\n## Facts\n"
    if facts:
        out += "\n" + "\n".join(facts) + "\n"
    history = [render_fact(f) for f in page.history] + page.unparsed.get("history", [])
    if history:
        out += "\n## History\n\n" + "\n".join(history) + "\n"
    if page.notes:
        out += f"\n## Notes\n\n{page.notes}\n"
    return out


def render_episode(episode: Episode) -> str:
    fm = render_frontmatter(
        [
            ("id", episode.id),
            ("type", "episode"),
            ("session", episode.session),
            ("turns", list(episode.turns)),
            ("channel", episode.channel),
            ("provider", episode.provider),
            ("regions", episode.regions),
            ("entities", [f"[[{e}]]" for e in episode.entities]),
            ("importance", episode.importance),
            ("gist", episode.gist),
            ("started", episode.started),
            ("status", episode.status),
        ]
    )
    return fm + (f"{episode.summary}\n" if episode.summary else "")


def render_observation(obs: Observation) -> str:
    parts = [obs.day.isoformat(), f"imp {obs.importance}", obs.text]
    if obs.src:
        parts.append(render_links(obs.src))
    return "- " + " · ".join(parts) + f" ^{obs.id}"


def render_journal(month: str, observations: Iterable[Observation]) -> str:
    lines = [render_observation(o) for o in observations]
    return f"# Journal {month}\n" + ("\n" + "\n".join(lines) + "\n" if lines else "")


def render_intention(intention: Intention) -> str:
    fields = [
        ("when", render_links(intention.when) if intention.when else None),
        ("after", intention.after),
        ("src", render_links(intention.src) if intention.src else None),
    ]
    inline = "".join(f" ({k}:: {v})" for k, v in fields if v)
    box = "x" if intention.done else " "
    return f"- [{box}] {intention.text}{inline} ^{intention.id}"


def render_intentions(intentions: Iterable[Intention]) -> str:
    lines = [render_intention(i) for i in intentions]
    return "# Intentions\n" + ("\n" + "\n".join(lines) + "\n" if lines else "")
