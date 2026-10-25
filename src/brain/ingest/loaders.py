"""Load a source document as markdown text: html (bs4 + markdownify), pdf (pypdf), md/txt."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify

HTML_DROP = ["script", "style", "noscript", "template", "svg", "iframe", "nav", "form", "button"]
SUFFIXES = {
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
}


class UnsupportedDocument(ValueError):
    pass


@dataclass(frozen=True)
class Document:
    path: Path
    format: str  # html | pdf | md | txt
    title: str
    text: str  # markdown (headings preserved where the source has them)


def tidy(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\xa0", " ").split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def html_to_markdown(html: str) -> tuple[str, str]:
    """(title, markdown) with scripts, styles, navigation, forms, links and images removed."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for tag in soup(HTML_DROP):
        tag.decompose()
    if not title and (h1 := soup.find("h1")):
        title = h1.get_text(" ", strip=True)
    body = soup.body or soup
    text = markdownify(str(body), heading_style="ATX", bullets="-", strip=["a", "img"])
    return title, tidy(text)


def pdf_to_text(path: Path) -> tuple[str, str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    meta_title = (reader.metadata.title if reader.metadata else None) or ""
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return str(meta_title).strip(), tidy("\n\n".join(p for p in pages if p))


def markdown_title(text: str) -> str:
    m = re.search(r"^#\s+(.+?)\s*#*\s*$", text, re.M)
    return m.group(1) if m else ""


def load_document(path: Path) -> Document:
    path = Path(path)
    fmt = SUFFIXES.get(path.suffix.lower())
    if fmt is None:
        supported = ", ".join(sorted(SUFFIXES))
        raise UnsupportedDocument(f"{path.name}: unsupported type (supported: {supported})")
    if fmt == "pdf":
        title, text = pdf_to_text(path)
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if fmt == "html":
            title, text = html_to_markdown(raw)
        else:
            text = tidy(raw)
            title = markdown_title(text) if fmt == "md" else ""
    return Document(path=path, format=fmt, title=title or path.stem, text=text)
