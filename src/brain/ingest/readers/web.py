"""Fetch a web page (http/https only, size-capped) and find same-origin links to follow.

The fetched bytes are saved to ``raw/docs/`` by the pipeline and read from there, so a web
page is ingested exactly like a downloaded file and survives in the vault as ground truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urldefrag, urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from brain.vault.raw import slugify

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,text/plain;q=0.8,*/*;q=0.5",
    "Accept-Language": "en;q=0.9,*;q=0.5",
}
TIMEOUT_S = 20.0
MAX_BYTES = 5_000_000
CONTENT_TYPES = {
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/x-markdown": ".md",
}
READABLE_SUFFIXES = {"", ".html", ".htm", ".xhtml", ".pdf", ".md", ".txt", ".php", ".asp", ".aspx"}
SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)


class FetchError(ValueError):
    pass


@dataclass(frozen=True)
class Fetched:
    url: str  # after redirects
    data: bytes
    suffix: str  # .html | .pdf | .txt | .md

    @property
    def name(self) -> str:
        """A file-name stem for the raw copy, e.g. ``alex-example-com-cv``."""
        parts = urlsplit(self.url)
        return slugify(f"{parts.hostname or ''} {parts.path}".replace(".", " "))


def is_url(text: str) -> bool:
    """True for anything with a scheme (``ftp://`` too, so it is refused, not read as a path)."""
    return bool(SCHEME.match(text.strip()))


def check_url(url: str) -> str:
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise FetchError(f"{url}: only http(s) URLs can be fetched")
    return url.strip()


def make_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(
        headers=HEADERS, timeout=TIMEOUT_S, follow_redirects=True, transport=transport
    )


def suffix_for(content_type: str, url: str) -> str:
    mime = content_type.split(";")[0].strip().lower()
    if mime in CONTENT_TYPES:
        return CONTENT_TYPES[mime]
    suffix = PurePosixPath(urlsplit(url).path).suffix.lower()
    if suffix in (".html", ".htm", ".pdf", ".md", ".txt"):
        return ".html" if suffix == ".htm" else suffix
    raise FetchError(f"{url}: unsupported content type {mime or 'unknown'!r}")


def fetch(url: str, client: httpx.Client, *, max_bytes: int = MAX_BYTES) -> Fetched:
    """GET ``url``; raise ``FetchError`` on a bad scheme, an HTTP error, an unreadable type or
    a body over ``max_bytes``."""
    check_url(url)
    try:
        with client.stream("GET", url) as resp:
            final = str(resp.url)
            check_url(final)
            if resp.status_code >= 400:
                raise FetchError(f"{url}: HTTP {resp.status_code}")
            declared = resp.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > max_bytes:
                raise FetchError(f"{url}: larger than {max_bytes} bytes")
            suffix = suffix_for(resp.headers.get("content-type", ""), final)
            body = bytearray()
            for piece in resp.iter_bytes():
                body += piece
                if len(body) > max_bytes:
                    raise FetchError(f"{url}: larger than {max_bytes} bytes")
    except httpx.HTTPError as err:
        raise FetchError(f"{url}: {err.__class__.__name__}: {err}") from err
    return Fetched(final, bytes(body), suffix)


def _key(url: str) -> str:
    return urldefrag(url)[0].rstrip("/")


def same_origin_links(html: bytes | str, base_url: str) -> list[str]:
    """Links on the page to other readable pages of the same origin, in page order (navigation
    included: on a personal site it points at the CV and about pages)."""
    soup = BeautifulSoup(html, "html.parser")
    base = urlsplit(base_url)
    origin = (base.scheme, base.hostname, base.port)
    seen = {_key(base_url)}
    out: list[str] = []
    for a in soup.find_all("a", href=True):
        url = urldefrag(urljoin(base_url, str(a["href"]).strip()))[0]
        parts = urlsplit(url)
        if (parts.scheme, parts.hostname, parts.port) != origin:
            continue
        if PurePosixPath(parts.path).suffix.lower() not in READABLE_SUFFIXES:
            continue
        if (key := _key(url)) not in seen:
            seen.add(key)
            out.append(url)
    return out
