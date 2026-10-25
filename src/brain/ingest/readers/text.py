"""Markdown and plain text: tidied, title from the first ``# `` heading (markdown only)."""

from __future__ import annotations

import re

from brain.ingest.readers.base import tidy


def markdown_title(text: str) -> str:
    m = re.search(r"^#\s+(.+?)\s*#*\s*$", text, re.M)
    return m.group(1) if m else ""


def read_text(raw: str, *, markdown: bool) -> tuple[str, str]:
    """(title, text)."""
    text = tidy(raw)
    return (markdown_title(text) if markdown else ""), text
