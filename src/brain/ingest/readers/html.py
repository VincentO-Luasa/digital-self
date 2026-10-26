"""HTML (a homepage, a CV page) → markdown the extractor can read.

- Animated counters (``<span data-target="7">0</span>``) are read at their target value, not
  the placeholder the page shows before its script runs.
- Scripts, styles, navigation, footers, forms, cookie banners and hidden elements are dropped;
  link targets and images too (link text is kept). Headings are kept for chunking.
- schema.org ``Person`` data (JSON-LD) and ``<meta name="description">`` become a short header.
- Adjacent inline elements are kept apart (``<span>02</span><span>Lab</span>`` would read
  "02Lab"), and decorative ordinals ("01", "02" in front of a section's kicker or heading) are
  dropped: the model took "02Lab" for a name.
- The cells of one row are kept together (``rows``): a timeline row, card or list item laid out
  as a grid (``<div class="row"><span>2021 – 2022</span><div><h3>School</h3><p>MSc …</p></div>
  </div>``) would otherwise read as three lines, the date above the name it belongs to, and
  the model shifted dates by one row. Definition terms are joined with their definitions.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
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


# --- rows -------------------------------------------------------------------------------------

ROW_TAGS = frozenset({"div", "li", "article"})
KEPT_BLOCKS = frozenset({"ul", "ol", "table", "dl", "pre", "blockquote", "figure"})
ROW_MAX_WORDS = 80  # a row is small: a section or a card with a long write-up is not one
ROW_MAX_CELLS = 8
CELL_SEP = " · "
CELL_LONG = 30  # a cell longer than this is a write-up, not a cell of a row


@dataclass
class Cell:
    text: str
    element: Tag | None  # the block holding it (None: loose inline text in the row)
    heading: bool = False
    kept: bool = False  # a list, table, … kept as it is, after the row's line


def _is_block(node: Any) -> bool:
    return isinstance(node, Tag) and node.name not in INLINE and node.name != "br"


def _has_blocks(tag: Tag) -> bool:
    return any(_is_block(c) for c in tag.children)


def _text(node: Any) -> str:
    raw = node.get_text(" ") if isinstance(node, Tag) else str(node)
    return " ".join(raw.split())


def cells(tag: Tag) -> list[Cell]:
    """The row's text blocks in reading order: blocks with only inline content (descending into
    blocks that hold more blocks), runs of loose inline text, and kept blocks (lists, tables)."""
    out: list[Cell] = []
    run: list[str] = []

    def flush() -> None:
        if text := " ".join(" ".join(run).split()):
            out.append(Cell(text, None))
        run.clear()

    for child in tag.children:
        if not _is_block(child):
            run.append(_text(child))
            continue
        flush()
        if child.name in KEPT_BLOCKS:
            out.append(Cell(_text(child), child, kept=True))
        elif _has_blocks(child):
            out += cells(child)
        elif text := _text(child):
            out.append(Cell(text, child, heading=child.name in HEADINGS))
    flush()
    return out


def _signature(tag: Tag) -> tuple[str, frozenset[str]]:
    return tag.name, frozenset(tag.get("class") or [])


def _repeated(tag: Tag) -> bool:
    """A sibling of the same tag and classes that also holds blocks: one of a series of rows
    (a grid, a timeline, cards). A lone block of a page is a section, not a row."""
    parent = tag.parent
    if tag.name == "li" or not isinstance(parent, Tag):
        return tag.name == "li"
    sig = _signature(tag)
    return any(
        s is not tag and isinstance(s, Tag) and _signature(s) == sig and _has_blocks(s)
        for s in parent.children
    )


def _row_cells(tag: Tag) -> list[Cell] | None:
    """The cells of ``tag`` if it is a row: a repeated small block (``ROW_TAGS``) holding other
    blocks, with 2 to ``ROW_MAX_CELLS`` text cells of at most ``ROW_MAX_WORDS`` words in all and
    at most one heading (not the page's h1)."""
    if tag.name not in ROW_TAGS or not _has_blocks(tag) or not _repeated(tag):
        return None
    found = cells(tag)
    texts = [c for c in found if not c.kept]
    headings = [c for c in texts if c.heading]
    if not 2 <= len(texts) <= ROW_MAX_CELLS or len(headings) > 1:
        return None
    if any(c.element is not None and c.element.name == "h1" for c in headings):
        return None
    if sum(len(c.text.split()) for c in texts) > ROW_MAX_WORDS:
        return None
    return found


def _line(soup: BeautifulSoup, name: str, text: str) -> Tag:
    el = soup.new_tag(name)
    el.string = text
    return el


def join_rows(soup: BeautifulSoup) -> int:
    """Rewrite every innermost row as one list item ("- 2021 – 2022 · School B · MSc in X, …"),
    consecutive rows in one list, a row's own lists and tables kept under it. A series of rows
    (same parent) where any cell is a write-up (over ``CELL_LONG`` words: a project card) is
    left as it is (measured on the eval homepage: one line per card, or the card's label moved
    into its heading, did not help). Definition terms are joined with their definitions.
    Returns how many rows changed."""
    rows: list[tuple[Tag, list[Cell]]] = []
    for tag in soup.find_all(ROW_TAGS):
        if (found := _row_cells(tag)) is None:
            continue
        # document order: an ancestor comes first; the innermost row wins
        rows = [(t, c) for t, c in rows if t not in tag.parents]
        rows.append((tag, found))
    long_series = {
        id(t.parent)
        for t, found in rows
        if any(len(c.text.split()) > CELL_LONG for c in found if not c.kept)
    }
    rows = [(t, c) for t, c in rows if id(t.parent) not in long_series]
    last: Tag | None = None  # the list item made for the previous row
    for tag, found in rows:
        text = CELL_SEP.join(c.text for c in found if not c.kept)
        kept = [c.element for c in found if c.kept and c.element is not None]
        for el in kept:
            el.extract()
        if tag.name == "li":
            tag.clear()
            tag.append(text)
            for el in kept:
                tag.append(el)
            last = None
            continue
        item = _line(soup, "li", text)
        for el in kept:
            item.append(el)
        previous = tag.find_previous_sibling(True)
        if last is not None and previous is not None and previous is last.parent:
            last.parent.append(item)
            tag.decompose()
        else:
            ul = soup.new_tag("ul")
            ul.append(item)
            tag.replace_with(ul)
        last = item
    for dl in soup.find_all("dl"):
        _join_definitions(soup, dl)
    return len(rows)


def _join_definitions(soup: BeautifulSoup, dl: Tag) -> None:
    """``<dt>2021 – 2022</dt><dd>School B, MSc in X</dd>`` → a list item "2021 – 2022 · School
    B, MSc in X" (each term with the definitions that follow it)."""
    items: list[list[str]] = []
    for child in dl.find_all(["dt", "dd"], recursive=False):
        text = _text(child)
        if child.name == "dt" or not items:
            items.append([text] if text else [])
        elif text:
            items[-1].append(text)
    if not items:
        return
    ul = soup.new_tag("ul")
    for parts in items:
        if parts:
            ul.append(_line(soup, "li", CELL_SEP.join(parts)))
    dl.replace_with(ul)


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
    join_rows(soup)
    if not title and (h1 := soup.find("h1")):
        title = h1.get_text(" ", strip=True)
    body = soup.body or soup
    text = markdownify(str(body), heading_style="ATX", bullets="-", strip=["a", "img"])
    return title, tidy(f"{top}\n\n{text}" if top else text)
