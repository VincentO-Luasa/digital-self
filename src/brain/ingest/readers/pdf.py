"""PDF: text of every page (pypdf), title from the document metadata."""

from __future__ import annotations

from io import BytesIO

from brain.ingest.readers.base import tidy


def read_pdf(data: bytes) -> tuple[str, str]:
    """(title, text)."""
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    meta_title = (reader.metadata.title if reader.metadata else None) or ""
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return str(meta_title).strip(), tidy("\n\n".join(p for p in pages if p))
