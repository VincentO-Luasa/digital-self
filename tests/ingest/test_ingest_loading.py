"""Loading documents as markdown and splitting them into heading-bounded chunks."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.ingest.chunking import chunk_markdown, split_sections
from brain.ingest.normalise import fact_text, normalise_date
from brain.ingest.readers import UnsupportedDocument, load_document
from brain.tokens import estimate_tokens

DOCS = Path(__file__).resolve().parents[1] / "fixtures" / "docs"


def minimal_pdf(text: str) -> bytes:
    """A one-page PDF with one line of text (enough for pypdf's text extraction)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for n, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return out


def test_html_becomes_clean_markdown() -> None:
    doc = load_document(DOCS / "alex-rivera.html")
    assert doc.format == "html"
    assert doc.title == "Alex Rivera — Data Engineer"
    # meta description and schema.org Person data first (contact details left out)
    assert doc.text.startswith(
        "Page description: Personal homepage of Alex Rivera, a fictional data engineer.\n\n"
        "Profile data from the page:\n- Name: Alex Rivera\n- Job title: Senior Data Engineer\n"
        "- Works for: Northwind Labs\n\n# Alex Rivera\n"
    )
    assert "## Experience" in doc.text and "### Fabrikam Retail — Lisbon" in doc.text
    assert "- Python and SQL" in doc.text
    # animated counters at their target value, not the placeholder 0
    assert "- 7+ years building data pipelines" in doc.text
    assert "- 2 university degrees" in doc.text and "0 university" not in doc.text
    gone = ("track:", "window.", "font-family", "https://", "alex.jpg", "Contact", "cookies")
    gone += ("Accept all", "newsletter", "alex@example.com", "©")
    for text in gone:
        assert text not in doc.text  # scripts, styles, links, images, nav, footer, banner, form
    assert "Northwind Labs" in doc.text  # link text is kept


def test_markdown_text_and_pdf(tmp_path: Path) -> None:
    md = tmp_path / "notes.md"
    md.write_text("Intro line\n\n# Atlas notes\n\nGo-live in November.\n")
    assert load_document(md).title == "Atlas notes"
    txt = tmp_path / "plain.txt"
    txt.write_text("Just text.\r\n\r\n\r\n\r\nMore.")
    doc = load_document(txt)
    assert (doc.title, doc.text) == ("plain", "Just text.\n\nMore.\n")
    pdf = tmp_path / "cv.pdf"
    pdf.write_bytes(minimal_pdf("Alex Rivera works at Northwind Labs"))
    doc = load_document(pdf)
    assert doc.format == "pdf" and "Alex Rivera works at Northwind Labs" in doc.text
    with pytest.raises(UnsupportedDocument):
        load_document(tmp_path / "photo.jpg")


def test_sections_follow_headings_and_ignore_code_fences() -> None:
    text = "# A\n\nintro\n\n## B\n\n## C\n\nbody c\n\n```\n# not a heading\n```\n\n### D\n\nd"
    sections = split_sections(text)
    trails = [s.trail for s in sections]
    assert trails == [("A",), ("A", "B"), ("A", "C", "D")]
    assert sections[1].text.startswith("## B\n\n## C")  # a bare heading joins the next section
    assert "# not a heading" in sections[1].text


def test_chunks_pack_sections_under_the_budget() -> None:
    doc = load_document(DOCS / "alex-rivera.html")
    chunks = chunk_markdown(doc.text)
    assert [c.heading for c in chunks] == ["", "Alex Rivera > Projects > Tidepool"]
    small = chunk_markdown(doc.text, max_tokens=300)
    assert len(small) > len(chunks)
    for chunk_list, budget in ((chunks, 500), (small, 300)):
        assert all(estimate_tokens(c.text) <= budget for c in chunk_list)
        assert all(c.text.startswith("#") for c in chunk_list[1:])  # at section boundaries
        joined = "".join(c.text for c in chunk_list)
        assert joined.count("Fabrikam") == doc.text.count("Fabrikam")


def test_oversized_section_is_split_by_paragraphs() -> None:
    paragraphs = [f"Paragraph {i}. " + "word " * 120 for i in range(6)]
    chunks = chunk_markdown("# Big\n\n" + "\n\n".join(paragraphs), max_tokens=400)
    assert len(chunks) > 1
    assert all(estimate_tokens(c.text) <= 400 for c in chunks)
    assert all(c.trail == ("Big",) for c in chunks)
    giant = chunk_markdown("x" * 10_000, max_tokens=500)
    assert all(estimate_tokens(c.text) <= 500 for c in giant)
    assert sum(len(c.text) for c in giant) == 10_000


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2023", "2023"),
        ("2023-02", "2023-02"),
        ("2023-2", "2023-02"),
        ("2023/02/05", "2023-02-05"),
        ("March 2025", "2025-03"),
        ("Sept. 2019", "2019-09"),
        ("02/2023", "2023-02"),
        ("present", None),
        ("", None),
        ("2023-13", None),
        ("spring 2024", None),
    ],
)
def test_normalise_date(value: str, expected: str | None) -> None:
    assert normalise_date(value) == expected


def test_fact_text_drops_leading_pronouns() -> None:
    assert fact_text("He speaks Spanish.", "x") == "Speaks Spanish."
    assert fact_text("  they   lead Atlas ", "x") == "Lead Atlas"
    assert fact_text("Is called Alex", "x") == "Is called Alex"
    assert fact_text("", "speaks Spanish") == "Speaks Spanish"
