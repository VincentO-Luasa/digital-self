"""HTML reader details, the web fetcher (``httpx.MockTransport``, no network) and raw copies of
fetched bytes."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest

from brain.ingest.readers import html_to_markdown, load_document
from brain.ingest.readers.web import (
    USER_AGENT,
    FetchError,
    check_url,
    fetch,
    is_url,
    same_origin_links,
)
from brain.vault.raw import save_doc_bytes
from brain.vault.store import Vault


def page(body: str, head: str = "") -> str:
    return f"<html><head><title>T</title>{head}</head><body>{body}</body></html>"


def test_counters_use_their_target_value() -> None:
    _, text = html_to_markdown(
        page(
            '<p><span class="c" data-target="7">0</span> years of data work</p>'
            '<p><span data-count="12">0+</span> talks</p>'
            '<p><span data-to="3" data-suffix="k">0</span> users</p>'
            '<p><span data-value="85">0</span>% coverage</p>'
            '<p><span data-target="9">Nine</span> lives</p>'  # real text: left alone
            '<p><span data-target="#about">0</span> links</p>'  # not a number: left alone
            '<p><span data-number="4"></span> books</p>'  # empty placeholder
        )
    )
    for line in ("7 years", "12+ talks", "3k users", "85% coverage", "Nine lives", "4 books"):
        assert line in text
    assert "0 links" in text


def test_person_json_ld_and_description_header() -> None:
    head = (
        '<meta property="og:description" content="Alex builds   data platforms.">'
        '<script type="application/ld+json">{not json</script>'
        '<script type="application/ld+json">{"@graph": [{"@type": "WebSite", "name": "Site"},'
        '{"@type": ["Person"], "name": "Alex Rivera", "alternateName": "Alex",'
        '"address": {"@type": "PostalAddress", "addressLocality": "Porto",'
        '"addressCountry": "PT", "streetAddress": "1 Secret St"},'
        '"knowsLanguage": ["Spanish", {"name": "English"}], "telephone": "+351 000",'
        '"alumniOf": [{"@type": "CollegeOrUniversity", "name": "Westbridge University"}]}]}'
        "</script>"
    )
    _, text = html_to_markdown(page("<h1>Alex</h1>", head))
    assert text.startswith(
        "Page description: Alex builds data platforms.\n\nProfile data from the page:\n"
        "- Name: Alex Rivera\n- Also known as: Alex\n- Alumni of: Westbridge University\n"
        "- Address area: Porto, PT\n- Languages: Spanish; English\n\n# Alex\n"
    )
    assert "Secret St" not in text and "+351" not in text  # contact details stay out


def test_noise_is_dropped_and_headings_kept() -> None:
    _, text = html_to_markdown(
        page(
            '<div role="navigation">Home About</div>'
            '<div class="gdpr-popup">We value your privacy</div>'
            "<div hidden>Old bio</div>"
            "<h2>Experience</h2><p>Data engineer.</p>"
            "<footer>© 2026</footer><dialog>Subscribe!</dialog>"
        )
    )
    assert text == "## Experience\n\nData engineer.\n"


def test_load_document_reads_xhtml(tmp_path: Path) -> None:
    doc = tmp_path / "page.xhtml"
    doc.write_text(page("<h1>Alex Rivera</h1>"))
    assert load_document(doc).format == "html"


# --- web ---------------------------------------------------------------------------------------


def test_url_checks() -> None:
    assert is_url("https://alex.example/cv") and is_url("FTP://alex.example")
    assert not is_url("notes/cv.html") and not is_url("C:\\cv.html")
    assert check_url(" https://alex.example ") == "https://alex.example"
    for bad in ("ftp://alex.example/cv", "file:///etc/hosts", "https://", "javascript:alert(1)"):
        with pytest.raises(FetchError):
            check_url(bad)


def client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_fetch_follows_redirects_and_sends_a_browser_user_agent() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "https://alex.example/cv.pdf"})
        return httpx.Response(200, content=b"%PDF-1.4", headers={"content-type": "application/pdf"})

    with client(handler) as http:
        http.headers["User-Agent"] = USER_AGENT
        got = fetch("https://alex.example/old", http)
    assert (got.url, got.suffix, got.data) == ("https://alex.example/cv.pdf", ".pdf", b"%PDF-1.4")
    assert got.name == "alex-example-cv-pdf"
    assert seen[0].headers["user-agent"].startswith("Mozilla/5.0")


def test_fetch_refuses_big_bodies_and_unreadable_types() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/big":
            return httpx.Response(200, content=b"x" * 2000, headers={"content-type": "text/html"})
        if request.url.path == "/declared":
            headers = {"content-type": "text/html", "content-length": "99999999"}
            return httpx.Response(200, content=b"<p>hi</p>", headers=headers)
        if request.url.path == "/photo":
            return httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})
        if request.url.path == "/notes.md":
            return httpx.Response(200, content=b"# Notes", headers={"content-type": ""})
        return httpx.Response(500)

    with client(handler) as http:
        with pytest.raises(FetchError, match="larger than 1000 bytes"):
            fetch("https://alex.example/big", http, max_bytes=1000)
        with pytest.raises(FetchError, match="larger than"):
            fetch("https://alex.example/declared", http)
        with pytest.raises(FetchError, match="unsupported content type 'image/png'"):
            fetch("https://alex.example/photo", http)
        with pytest.raises(FetchError, match="HTTP 500"):
            fetch("https://alex.example/broken", http)
        assert fetch("https://alex.example/notes.md", http).suffix == ".md"  # from the path

    def offline(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with client(offline) as http, pytest.raises(FetchError, match="ConnectError"):
        fetch("https://alex.example/", http)


def test_same_origin_links_in_page_order() -> None:
    html = (
        '<a href="cv">CV</a><a href="/cv/">CV again</a><a href="#top">Top</a>'
        '<a href="https://alex.example/talks.html#2024">Talks</a>'
        '<a href="http://alex.example/insecure">http</a><a href="https://cdn.alex.example/a">CDN</a>'
        '<a href="/files/cv.pdf">PDF</a><a href="/img/me.png">Me</a>'
        '<a href="mailto:alex@example.com">Mail</a><a href="https://alex.example/">Home</a>'
    )
    assert same_origin_links(html, "https://alex.example/") == [
        "https://alex.example/cv",
        "https://alex.example/talks.html",
        "https://alex.example/files/cv.pdf",
    ]


def test_save_doc_bytes(tmp_vault: Vault) -> None:
    today = date(2026, 10, 25)
    path, created = save_doc_bytes(tmp_vault, b"<p>cv</p>", "Alex Example CV", ".html", today=today)
    assert created and path.name == "2026-10-25-alex-example-cv.html"
    again, created = save_doc_bytes(tmp_vault, b"<p>cv</p>", "other", ".html", today=today)
    assert again == path and not created
    other, created = save_doc_bytes(tmp_vault, b"<p>v2</p>", "Alex Example CV", "HTML", today=today)
    assert created and other.name == "2026-10-25-alex-example-cv-2.html"
