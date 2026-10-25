"""HTML (a homepage, a CV page) → markdown the extractor can read.

- Animated counters (``<span data-target="7">0</span>``) are read at their target value, not
  the placeholder the page shows before its script runs.
- Scripts, styles, navigation, footers, forms, cookie banners and hidden elements are dropped;
  link targets and images too (link text is kept). Headings are kept for chunking.
- schema.org ``Person`` data (JSON-LD) and ``<meta name="description">`` become a short header.
- Adjacent inline elements are kept apart (``<span>02</span><span>Lab</span>`` would read
  "02Lab"), and decorative ordinals ("01", "02" in front of a section's kicker or heading) are
  dropped: the model took "02Lab" for a name.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag
from markdownify import markdownify

from brain.ingest.readers.base import tidy

DROP_TAGS = [
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "iframe",
    "nav",
    "footer",
    "form",
    "button",
    "dialog",
]
DROP_ROLES = {"navigation", "contentinfo", "dialog", "alertdialog", "search"}
NOISE = re.compile(r"cookie|consent|gdpr", re.I)
# Attributes counter scripts animate towards (the visible text is a placeholder until then).
COUNTER_ATTRS = (
    "data-target",
    "data-count",
    "data-to",
    "data-value",
    "data-number",
    "data-end",
    "data-end-value",
    "data-count-to",
    "data-countup",
    "data-final",
)
NUMBER = re.compile(r"^[-+]?\d+(?:[.,]\d+)*\s*[+%kKmM]?\+?$")
PERSON_FIELDS = (  # (schema.org property, label); contact details are left out on purpose
    ("name", "Name"),
    ("alternateName", "Also known as"),
    ("jobTitle", "Job title"),
    ("worksFor", "Works for"),
    ("affiliation", "Affiliation"),
    ("alumniOf", "Alumni of"),
    ("homeLocation", "Lives in"),
    ("address", "Address area"),
    ("birthPlace", "Birth place"),
    ("birthDate", "Born"),
    ("nationality", "Nationality"),
    ("knowsLanguage", "Languages"),
    ("knowsAbout", "Knows about"),
    ("award", "Awards"),
    ("description", "Description"),
)
INLINE = frozenset(
    {
        *("a", "abbr", "b", "bdi", "bdo", "cite", "code", "data", "dfn", "em", "i", "kbd"),
        *("label", "mark", "q", "s", "samp", "small", "span", "strong", "time", "u", "var"),
    }
)
HEADINGS = frozenset(f"h{n}" for n in range(1, 7))
ORDINAL = re.compile(r"^(?:\d{1,2})[.)]?$")  # "01", "7", "12."
KICKER_WORDS = 8  # a line this short next to an ordinal is a kicker/eyebrow, not a sentence
MAX_FIELD_CHARS = 300
MAX_LIST_ITEMS = 15


def _number(value: str | None) -> str | None:
    value = (value or "").strip()
    return value if value and NUMBER.match(value) else None


def fix_counters(soup: BeautifulSoup) -> int:
    """Replace a counter's placeholder text with its target value; return how many changed."""
    changed = 0
    for el in soup.find_all(lambda t: any(t.has_attr(a) for a in COUNTER_ATTRS)):
        target = next((v for a in COUNTER_ATTRS if (v := _number(el.get(a)))), None)
        shown = el.get_text("", strip=True)
        if target is None or (shown and not NUMBER.match(shown)) or shown == target:
            continue
        suffix = "+" if shown.endswith("+") and not target.endswith("+") else ""
        prefix, after = el.get("data-prefix") or "", el.get("data-suffix") or ""
        el.clear()
        el.append(f"{prefix}{target}{suffix}{after}")
        changed += 1
    return changed


def _children(tag: Tag) -> list[Any]:
    return [c for c in tag.contents if not (isinstance(c, NavigableString) and not c.strip())]


def _next_element(tag: Tag) -> Tag | None:
    """The next sibling if it is an element (whitespace between them ignored)."""
    sibling = tag.next_sibling
    while isinstance(sibling, NavigableString) and not sibling.strip():
        sibling = sibling.next_sibling
    return sibling if isinstance(sibling, Tag) else None


def _decorative_ordinal(el: Tag) -> bool:
    """A bare 1–2 digit number used as a section index: the whole text of an element that comes
    right before a heading, or the first piece of a heading or of a short line followed by more
    elements ("<span>02</span><span>Research</span>"; zero-padded, or the line is a heading's
    kicker). A count in a sentence ("7 years") is followed by text and stays."""
    text = el.get_text("", strip=True)
    if not ORDINAL.match(text) or el.find(True) or any(el.has_attr(a) for a in COUNTER_ATTRS):
        return False
    after = _next_element(el)
    if after is not None and after.name in HEADINGS:
        return True
    parent = el.parent
    if not isinstance(parent, Tag) or el.name not in INLINE:
        return False
    pieces = _children(parent)
    if len(pieces) < 2 or pieces[0] is not el:
        return False
    rest = parent.get_text(" ", strip=True)[len(text) :].split()
    if not rest or len(rest) > KICKER_WORDS:
        return False
    if parent.name in HEADINGS:
        return True
    following = _next_element(parent)
    before_heading = following is not None and following.name in HEADINGS
    return isinstance(pieces[1], Tag) and (text.startswith("0") or before_heading)


def drop_ordinals(soup: BeautifulSoup) -> int:
    found = [el for el in soup.find_all(True) if _decorative_ordinal(el)]
    for el in found:
        el.decompose()
    return len(found)


def space_inline(soup: BeautifulSoup) -> None:
    """Put a space between two adjacent inline elements whose texts would otherwise touch
    ("<span>Lab</span><span>Notes</span>"); single characters (letter-by-letter animation
    spans) are left as they are."""
    for el in soup.find_all(INLINE):
        after = el.next_sibling
        if not isinstance(after, Tag) or after.name not in INLINE:
            continue
        left, right = el.get_text(""), after.get_text("")
        if len(left.strip()) < 2 or len(right.strip()) < 2:
            continue
        if not left[-1].isspace() and not right[0].isspace():
            el.insert_after(" ")


def _hidden(tag: Tag) -> bool:
    if tag.has_attr("hidden") or tag.get("role") in DROP_ROLES:
        return True
    marks = " ".join([tag.get("id") or "", *(tag.get("class") or [])])
    return bool(NOISE.search(marks))


def _walk(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, list):
        for item in node:
            yield from _walk(item)
    elif isinstance(node, dict):
        yield node
        for value in node.values():
            if isinstance(value, (dict, list)):
                yield from _walk(value)


def _is_person(node: dict[str, Any]) -> bool:
    kind = node.get("@type")
    kinds = kind if isinstance(kind, list) else [kind]
    return "Person" in kinds


def _value(value: Any) -> str:
    if isinstance(value, list):
        parts = [v for v in (_value(x) for x in value[:MAX_LIST_ITEMS]) if v]
        return "; ".join(dict.fromkeys(parts))
    if isinstance(value, dict):
        if value.get("name"):
            return _value(value["name"])
        place = [value.get(k) for k in ("addressLocality", "addressRegion", "addressCountry")]
        return ", ".join(_value(p) for p in place if p)
    if isinstance(value, (str, int, float)):
        return " ".join(str(value).split())[:MAX_FIELD_CHARS]
    return ""


def person_lines(soup: BeautifulSoup) -> list[str]:
    """Lines describing the first schema.org ``Person`` in the page's JSON-LD."""
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except ValueError:
            continue
        for node in _walk(data):
            if not _is_person(node):
                continue
            values = ((label, _value(node.get(key))) for key, label in PERSON_FIELDS)
            lines = [f"- {label}: {v}" for label, v in values if v]
            if lines:
                return lines
    return []


def page_description(soup: BeautifulSoup) -> str:
    for attrs in ({"name": "description"}, {"property": "og:description"}):
        meta = soup.find("meta", attrs=attrs)
        if isinstance(meta, Tag) and (content := " ".join(str(meta.get("content", "")).split())):
            return content[:MAX_FIELD_CHARS]
    return ""


def header(soup: BeautifulSoup) -> str:
    parts = []
    if description := page_description(soup):
        parts.append(f"Page description: {description}")
    if lines := person_lines(soup):
        parts.append("Profile data from the page:\n" + "\n".join(lines))
    return "\n\n".join(parts)


def html_to_markdown(html: str) -> tuple[str, str]:
    """(title, markdown)."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    top = header(soup)  # before scripts (JSON-LD) are dropped
    fix_counters(soup)
    for tag in soup(DROP_TAGS):
        tag.decompose()
    for tag in soup.find_all(_hidden):
        if not tag.decomposed:
            tag.decompose()
    drop_ordinals(soup)
    space_inline(soup)
    if not title and (h1 := soup.find("h1")):
        title = h1.get_text(" ", strip=True)
    body = soup.body or soup
    text = markdownify(str(body), heading_style="ATX", bullets="-", strip=["a", "img"])
    return title, tidy(f"{top}\n\n{text}" if top else text)
