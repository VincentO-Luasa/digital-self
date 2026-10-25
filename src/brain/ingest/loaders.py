"""Compatibility shim: the readers live in ``brain.ingest.readers``."""

from brain.ingest.readers import (
    Document,
    UnsupportedDocument,
    html_to_markdown,
    load_document,
    markdown_title,
    tidy,
)

__all__ = [
    "Document",
    "UnsupportedDocument",
    "html_to_markdown",
    "load_document",
    "markdown_title",
    "tidy",
]
