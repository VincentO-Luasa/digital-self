"""Parse vault markdown (docs/vault-schema.md) into models.

Parsing is tolerant of hand edits: malformed lines become ``ParseWarning``s (and, on pages, are
kept verbatim in ``Page.unparsed``) instead of exceptions. Only a file that cannot yield a model
at all raises ``ParseError``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, get_args

import yaml
from pydantic import ValidationError

from brain.vault.models import (
    SELF_ID,
    WIKILINK,
    EntityType,
    Episode,
    Fact,
    Intention,
    Observation,
    Page,
    Privacy,
    link_target,
)

FIELD_START = re.compile(r"\(([A-Za-z_][\w-]*)::[ \t]*")
BLOCK_ID = re.compile(r"\s\^([A-Za-z0-9-]+)\s*$")
FACT_ID = re.compile(r"^f-[a-z2-7]{6}$")
LIST_ITEM = re.compile(r"^[-*+]\s+(.*)$")
TASK_ITEM = re.compile(r"^[-*+]\s+\[([ xX])\]\s+(.*)$")
OBS_SEP = " · "

FACT_FIELDS = ("p", "o", "from", "to", "rec", "src", "kind", "evidence", "by")
PAGE_KEYS = ("id", "type", "subject", "aliases", "regions", "privacy", "created", "updated")
DIR_TYPES: dict[str, str] = {
    "self": "self",
    "people": "person",
    "orgs": "org",
    "projects": "project",
    "places": "place",
    "topics": "topic",
    "events": "event",
}


@dataclass(frozen=True)
class ParseWarning:
    path: Path | None
    line: int  # 1-based; 0 = whole file
    message: str

    def __str__(self) -> str:
        where = f"{self.path or '<text>'}:{self.line}"
        return f"{where}: {self.message}"


class ParseError(ValueError):
    """The text cannot be turned into a model at all."""


class _Warn:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.items: list[ParseWarning] = []

    def __call__(self, line: int, message: str) -> None:
        self.items.append(ParseWarning(self.path, line, message))


# --- small codecs -----------------------------------------------------------------------------


def parse_links(value: str) -> list[str]:
    """``[[a]], [[b|B]], "a quote", bare`` → ``["a", "b", "a quote", "bare"]``."""
    items: list[str] = []
    for m in re.finditer(r'\s*(\[\[[^\]]*\]\]|"[^"]*"|[^,]+)', value):
        token = m.group(1).strip()
        if not token:
            continue
        if token.startswith("[["):
            target = link_target(token)
            if target:
                items.append(target)
        elif token.startswith('"') and token.endswith('"') and len(token) >= 2:
            items.append(token[1:-1])
        else:
            items.append(token)
    return items


def split_inline_fields(line: str) -> tuple[str, list[tuple[str, str]], list[str]]:
    """Split ``text (k:: v) (k2:: v2)`` into text, fields in order, and stray fragments.

    Values may contain balanced parentheses and wikilinks. Raises ValueError on an unterminated
    field.
    """
    first = FIELD_START.search(line)
    if not first:
        return line.strip(), [], []
    text, fields, stray = line[: first.start()].strip(), [], []
    pos = first.start()
    while pos < len(line):
        m = FIELD_START.match(line, pos)
        if not m:
            nxt = FIELD_START.search(line, pos)
            end = nxt.start() if nxt else len(line)
            if frag := line[pos:end].strip():
                stray.append(frag)
            if not nxt:
                break
            pos = end
            continue
        end = _field_end(line, m.end())
        fields.append((m.group(1), line[m.end() : end - 1].strip()))
        pos = end
    return text, fields, stray


def _field_end(line: str, i: int) -> int:
    """Index just past the ``)`` closing a field whose value starts at ``i``."""
    depth = 1
    while i < len(line):
        if line.startswith("[[", i) and (j := line.find("]]", i)) != -1:
            i = j + 2
            continue
        depth += {"(": 1, ")": -1}.get(line[i], 0)
        i += 1
        if depth == 0:
            return i
    raise ValueError("unterminated inline field")


def _split_block_id(body: str) -> tuple[str, str | None]:
    m = BLOCK_ID.search(" " + body)
    if not m:
        return body.rstrip(), None
    return (" " + body)[: m.start()].strip(), m.group(1)


def split_frontmatter(text: str) -> tuple[dict[str, Any] | None, str, int]:
    """Return (frontmatter or None, body, number of lines before the body)."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None, text, 0
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            try:
                data = yaml.safe_load("\n".join(lines[1:i])) or {}
            except yaml.YAMLError as exc:
                raise ParseError(f"invalid YAML frontmatter: {exc}") from exc
            if not isinstance(data, dict):
                raise ParseError("frontmatter is not a mapping")
            return data, "\n".join(lines[i + 1 :]), i + 1
    raise ParseError("frontmatter is not closed with ---")


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _trim_blank(lines: list[str]) -> str:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end])


# --- facts -------------------------------------------------------------------------------------


def parse_fact_line(line: str) -> Fact:
    """Parse one ``- text (p:: ...) ... ^f-xxxxxx`` line. Raises ValueError if malformed."""
    m = LIST_ITEM.match(line.strip())
    if not m:
        raise ValueError("not a list item")
    body, block_id = _split_block_id(m.group(1))
    if not block_id:
        raise ValueError("missing ^f-xxxxxx block id")
    text, fields, stray = split_inline_fields(body)
    if stray:
        raise ValueError(f"text between fields: {stray[0]!r}")
    if not text:
        raise ValueError("missing fact text")
    known: dict[str, str] = {}
    extra: dict[str, str] = {}
    for key, value in fields:
        target = known if key in FACT_FIELDS else extra
        if key in target:
            raise ValueError(f"duplicate field {key}::")
        target[key] = value
    missing = [k for k in ("p", "o", "rec", "src", "kind") if not known.get(k)]
    if missing:
        raise ValueError("missing " + ", ".join(f"{k}::" for k in missing))
    src = parse_links(known["src"])
    if not src:
        raise ValueError("src:: has no source")
    by = known.get("by")
    try:
        return Fact(
            id=block_id,
            text=text,
            p=known["p"],
            o=known["o"],
            valid_from=known.get("from") or None,
            valid_to=known.get("to") or None,
            rec=known["rec"],
            src=src,
            kind=known["kind"],
            evidence=parse_links(known.get("evidence", "")),
            by=(link_target(by) or by).lstrip("#^") if by else None,
            extra=extra,
        )
    except ValidationError as exc:
        err = exc.errors()[0]
        raise ValueError(f"{'.'.join(map(str, err['loc']))}: {err['msg']}") from exc


# --- pages -------------------------------------------------------------------------------------


@dataclass
class _Body:
    title: str | None = None
    summary: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    extra_sections: list[str] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    history: list[Fact] = field(default_factory=list)
    unparsed: dict[str, list[str]] = field(default_factory=dict)


def _parse_page_body(body: str, offset: int, warn: _Warn) -> _Body:
    out = _Body()
    state = "pre"
    for idx, line in enumerate(body.split("\n")):
        lineno = offset + idx + 1
        if state != "notes":
            if state == "pre" and (h1 := re.match(r"^#\s+(.+?)\s*$", line)):
                out.title, state = h1.group(1), "summary"
                continue
            if h2 := re.match(r"^##\s+(.+?)\s*#*\s*$", line):
                name = h2.group(1).strip().casefold()
                if name in ("facts", "history", "notes", "summary"):
                    state = name
                else:
                    warn(lineno, f"unknown section {line.strip()!r} kept in Notes")
                    out.extra_sections.append(line)
                    state = "extra"
                continue
        if state == "pre":
            if line.strip():
                warn(lineno, "text before the # title")
                out.summary.append(line)
        elif state == "summary":
            out.summary.append(line)
        elif state == "notes":
            out.notes.append(line)
        elif state == "extra":
            out.extra_sections.append(line)
        elif line.strip():
            _add_fact_line(line, state, lineno, out, warn)
    return out


def _add_fact_line(line: str, section: str, lineno: int, out: _Body, warn: _Warn) -> None:
    try:
        fact = parse_fact_line(line)
    except ValueError as exc:
        warn(lineno, f"malformed {section} line kept verbatim: {exc}")
        out.unparsed.setdefault(section, []).append(line.rstrip())
        return
    if not FACT_ID.match(fact.id):
        warn(lineno, f"non-canonical fact id ^{fact.id}")
    if section == "history" and not (fact.valid_to and fact.by):
        warn(lineno, f"history fact ^{fact.id} lacks to:: or by::")
    (out.facts if section == "facts" else out.history).append(fact)


def _page_id_from_path(path: Path | None) -> str | None:
    if path is None:
        return None
    parts = path.with_suffix("").parts
    if "wiki" in parts:
        return "/".join(parts[len(parts) - parts[::-1].index("wiki") :])
    return "/".join(parts[-2:])


def _frontmatter_field(fm: dict[str, Any], key: str, default: Any, warn: _Warn) -> Any:
    if fm.get(key) in (None, "", []):
        if default is None:
            raise ParseError(f"frontmatter lacks {key}")
        warn(1, f"frontmatter lacks {key}; using {default!r}")
        return default
    return fm[key]


def parse_page(
    text: str, *, path: Path | None = None, today: date | None = None
) -> tuple[Page, list[ParseWarning]]:
    """Parse a wiki page. ``path`` (``.../wiki/<id>.md``) fills in a missing or wrong id."""
    warn = _Warn(path)
    fm, body, offset = split_frontmatter(text)
    if fm is None:
        warn(1, "no frontmatter")
        fm = {}
    path_id = _page_id_from_path(path)
    page_id = str(_frontmatter_field(fm, "id", path_id, warn))
    if path_id and page_id != path_id:
        warn(1, f"id {page_id!r} does not match path; using {path_id!r}")
        page_id = path_id
    folder = page_id.split("/", 1)[0]
    page_type = _frontmatter_field(fm, "type", DIR_TYPES.get(folder), warn)
    if page_type not in get_args(EntityType):
        raise ParseError(f"unknown page type {page_type!r}")
    default_subject = SELF_ID if page_type == "self" else page_id
    subject = str(_frontmatter_field(fm, "subject", default_subject, warn))
    privacy = fm.get("privacy", "normal")
    if privacy not in get_args(Privacy):
        warn(1, f"unknown privacy {privacy!r}; treating the page as private")
        privacy = "private"
    stamp = today or date.today()
    created = _frontmatter_field(fm, "created", stamp, warn)
    updated = _frontmatter_field(fm, "updated", created, warn)

    parsed = _parse_page_body(body, offset, warn)
    title = parsed.title
    if not title:
        title = page_id.rsplit("/", 1)[-1].replace("-", " ").title()
        warn(offset + 1, f"no # title; using {title!r}")
    notes = _trim_blank(parsed.extra_sections + [""] + parsed.notes)
    try:
        page = Page(
            id=page_id,
            type=page_type,
            subject=link_target(subject) or subject,
            title=title,
            aliases=_str_list(fm.get("aliases")),
            regions=_str_list(fm.get("regions")),
            privacy=privacy,
            created=created,
            updated=updated,
            summary=_trim_blank(parsed.summary),
            facts=parsed.facts,
            history=parsed.history,
            notes=notes,
            meta={k: v for k, v in fm.items() if k not in PAGE_KEYS},
            unparsed=parsed.unparsed,
        )
    except ValidationError as exc:
        raise ParseError(f"invalid page frontmatter: {exc}") from exc
    return page, warn.items


# --- episodes ----------------------------------------------------------------------------------


def parse_episode(text: str, *, path: Path | None = None) -> tuple[Episode, list[ParseWarning]]:
    warn = _Warn(path)
    fm, body, _ = split_frontmatter(text)
    if fm is None:
        raise ParseError("episode has no frontmatter")
    data = {k: v for k, v in fm.items() if k != "type"}
    if fm.get("type", "episode") != "episode":
        warn(1, f"type is {fm['type']!r}, expected episode")
    data["entities"] = [link_target(e) or e for e in _str_list(fm.get("entities"))]
    data["regions"] = _str_list(fm.get("regions"))
    if data.get("started") is not None:
        data["started"] = str(data["started"])
    unknown = set(data) - set(Episode.model_fields)
    for key in sorted(unknown):
        warn(1, f"unknown episode field {key!r} ignored")
        del data[key]
    data["summary"] = body.strip()
    try:
        return Episode.model_validate(data), warn.items
    except ValidationError as exc:
        raise ParseError(f"invalid episode: {exc}") from exc


# --- journal -----------------------------------------------------------------------------------


def parse_observation_line(line: str) -> Observation:
    m = LIST_ITEM.match(line.strip())
    if not m:
        raise ValueError("not a list item")
    body, block_id = _split_block_id(m.group(1))
    if not block_id:
        raise ValueError("missing ^o-xxxxxx block id")
    parts = body.split(OBS_SEP)
    if len(parts) < 3:
        raise ValueError("expected 'date · imp N · text · [[src]]'")
    imp = re.fullmatch(r"imp\s+(\d+)", parts[1].strip())
    if not imp:
        raise ValueError(f"expected 'imp N', got {parts[1]!r}")
    src: list[str] = []
    rest = parts[2:]
    if len(rest) > 1 and WIKILINK.search(rest[-1]):
        src = parse_links(rest.pop())
    try:
        return Observation(
            id=block_id,
            day=parts[0].strip(),
            importance=int(imp.group(1)),
            text=OBS_SEP.join(rest).strip(),
            src=src,
        )
    except ValidationError as exc:
        err = exc.errors()[0]
        raise ValueError(f"{'.'.join(map(str, err['loc']))}: {err['msg']}") from exc


def parse_journal(
    text: str, *, path: Path | None = None
) -> tuple[list[Observation], list[ParseWarning]]:
    warn = _Warn(path)
    _, body, offset = split_frontmatter(text)
    out: list[Observation] = []
    for idx, line in enumerate(body.split("\n")):
        if not LIST_ITEM.match(line.strip()):
            if line.strip() and not line.startswith("#"):
                warn(offset + idx + 1, "ignored non-observation line")
            continue
        try:
            obs = parse_observation_line(line)
        except ValueError as exc:
            warn(offset + idx + 1, f"malformed observation: {exc}")
            continue
        if not obs.src:
            warn(offset + idx + 1, f"observation ^{obs.id} has no source")
        out.append(obs)
    return out, warn.items


# --- intentions --------------------------------------------------------------------------------


def parse_intention_line(line: str) -> tuple[Intention, list[str]]:
    """Return the intention and the names of ignored unknown fields."""
    m = TASK_ITEM.match(line.strip())
    if not m:
        raise ValueError("not a task item '- [ ] ...'")
    body, block_id = _split_block_id(m.group(2))
    if not block_id:
        raise ValueError("missing ^i-xxxxxx block id")
    text, fields, stray = split_inline_fields(body)
    if stray:
        raise ValueError(f"text between fields: {stray[0]!r}")
    values = dict(fields)
    unknown = [k for k in values if k not in ("when", "after", "src")]
    try:
        intention = Intention(
            id=block_id,
            text=text,
            done=m.group(1) != " ",
            when=parse_links(values.get("when", "")),
            after=values.get("after") or None,
            src=parse_links(values.get("src", "")),
        )
    except ValidationError as exc:
        err = exc.errors()[0]
        raise ValueError(f"{'.'.join(map(str, err['loc']))}: {err['msg']}") from exc
    return intention, unknown


def parse_intentions(
    text: str, *, path: Path | None = None
) -> tuple[list[Intention], list[ParseWarning]]:
    warn = _Warn(path)
    _, body, offset = split_frontmatter(text)
    out: list[Intention] = []
    for idx, line in enumerate(body.split("\n")):
        lineno = offset + idx + 1
        if not LIST_ITEM.match(line.strip()):
            if line.strip() and not line.startswith("#"):
                warn(lineno, "ignored non-intention line")
            continue
        try:
            intention, unknown = parse_intention_line(line)
        except ValueError as exc:
            warn(lineno, f"malformed intention: {exc}")
            continue
        for key in unknown:
            warn(lineno, f"unknown intention field {key}:: ignored")
        out.append(intention)
    return out, warn.items
