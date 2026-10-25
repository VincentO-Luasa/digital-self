"""Render a plain-text CV into a small text PDF (no dependency beyond the standard library).

    uv run python tests/eval/ingest/make_pdf.py tests/eval/ingest/sources/maya-lindqvist-cv.txt \
        tests/eval/ingest/docs/maya-lindqvist-cv.pdf

Lines starting with ``# `` or ``## `` are set in bold (without the marker); everything else in
Helvetica, wrapped at the page width. Text is encoded as WinAnsi (cp1252), so en dashes and
middle dots survive the round trip through ``pypdf``.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

PAGE_W, PAGE_H, MARGIN = 595, 842, 56  # A4 in points
LEADING = {"h1": 20, "h2": 16, "p": 13}
SIZES = {"h1": 14, "h2": 11, "p": 10}
WRAP = 95


def _escape(text: str) -> bytes:
    raw = text.encode("cp1252", errors="replace")
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def layout(source: str) -> list[list[tuple[str, str]]]:
    """Pages of (style, line)."""
    lines: list[tuple[str, str]] = []
    for line in source.splitlines():
        if line.startswith("## "):
            lines.append(("h2", line[3:]))
        elif line.startswith("# "):
            lines += [("p", ""), ("h1", line[2:])]
        else:
            wrapped = textwrap.wrap(line, WRAP) or [""]
            lines += [("p", w) for w in wrapped]
    pages: list[list[tuple[str, str]]] = [[]]
    y = PAGE_H - MARGIN
    for style, text in lines:
        if y - LEADING[style] < MARGIN:
            pages.append([])
            y = PAGE_H - MARGIN
        y -= LEADING[style]
        pages[-1].append((style, text))
    return pages


def content_stream(page: list[tuple[str, str]]) -> bytes:
    out = [b"BT"]
    y = PAGE_H - MARGIN
    for style, text in page:
        y -= LEADING[style]
        font = b"/F2" if style in ("h1", "h2") else b"/F1"
        out.append(font + b" %d Tf 1 0 0 1 %d %d Tm (" % (SIZES[style], MARGIN, y))
        out[-1] += _escape(text) + b") Tj"
    out.append(b"ET")
    return b"\n".join(out)


def build_pdf(source: str, title: str) -> bytes:
    pages = layout(source)
    objects: list[bytes] = []  # object n is objects[n - 1]

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog = add(b"")  # filled in below
    pages_id = add(b"")
    f1 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    f2 = add(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
    )
    kids = []
    for page in pages:
        stream = content_stream(page)
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        kids.append(
            add(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %d %d] /Contents %d 0 R "
                b"/Resources << /Font << /F1 %d 0 R /F2 %d 0 R >> >> >>"
                % (pages_id, PAGE_W, PAGE_H, content, f1, f2)
            )
        )
    info = add(b"<< /Title (" + _escape(title) + b") /Producer (local_brain make_pdf) >>")
    objects[catalog - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id
    refs = b" ".join(b"%d 0 R" % k for k in kids)
    objects[pages_id - 1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (refs, len(kids))

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for n, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root %d 0 R /Info %d 0 R >>\n" % (
        len(objects) + 1,
        catalog,
        info,
    )
    out += b"startxref\n%d\n%%%%EOF\n" % xref
    return bytes(out)


def main(argv: list[str]) -> None:
    src, dst = Path(argv[1]), Path(argv[2])
    title = argv[3] if len(argv) > 3 else "Maya Lindqvist - CV"
    dst.write_bytes(build_pdf(src.read_text(encoding="utf-8"), title))


if __name__ == "__main__":
    main(sys.argv)
