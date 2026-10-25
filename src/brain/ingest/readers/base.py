"""What every reader returns: a document as markdown text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


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
