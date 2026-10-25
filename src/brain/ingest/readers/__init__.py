"""Readers turn a source into markdown text: one module per source type.

``load_document(path)`` picks a reader by file suffix. Web pages are fetched by
``brain.ingest.readers.web``, saved to ``raw/docs/`` and then read from that copy like any file.
To support a new type, add a module with a ``read_<type>(...) -> (title, text)`` function and
register its suffixes in ``SUFFIXES`` and ``read_bytes``.
"""

from __future__ import annotations

from pathlib import Path

from brain.ingest.readers.base import Document, UnsupportedDocument, tidy
from brain.ingest.readers.html import html_to_markdown
from brain.ingest.readers.pdf import read_pdf
from brain.ingest.readers.text import markdown_title, read_text

SUFFIXES = {
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".pdf": "pdf",
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
}

__all__ = [
    "SUFFIXES",
    "Document",
    "UnsupportedDocument",
    "format_of",
    "html_to_markdown",
    "load_document",
    "markdown_title",
    "read_bytes",
    "tidy",
]


def format_of(path: Path) -> str:
    fmt = SUFFIXES.get(Path(path).suffix.lower())
    if fmt is None:
        supported = ", ".join(sorted(SUFFIXES))
        raise UnsupportedDocument(f"{Path(path).name}: unsupported type (supported: {supported})")
    return fmt


def read_bytes(data: bytes, fmt: str) -> tuple[str, str]:
    """(title, markdown text) of ``data`` in format ``fmt``."""
    if fmt == "pdf":
        return read_pdf(data)
    raw = data.decode("utf-8", errors="replace")
    if fmt == "html":
        return html_to_markdown(raw)
    return read_text(raw, markdown=fmt == "md")


def load_document(path: Path) -> Document:
    path = Path(path)
    fmt = format_of(path)
    title, text = read_bytes(path.read_bytes(), fmt)
    return Document(path=path, format=fmt, title=title or path.stem, text=text)
