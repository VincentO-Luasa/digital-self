"""Document ingest: read a file or URL, keep a raw copy, chunk, extract, normalise, apply."""

from brain.ingest.chunking import Chunk, chunk_markdown
from brain.ingest.extract import Extraction, extract_chunk
from brain.ingest.normalise import Normalised, build_changeset
from brain.ingest.pipeline import IngestReport, WebReport, ingest, ingest_url
from brain.ingest.readers import Document, UnsupportedDocument, load_document
from brain.ingest.readers.web import FetchError, is_url

__all__ = [
    "Chunk",
    "Document",
    "Extraction",
    "FetchError",
    "IngestReport",
    "Normalised",
    "UnsupportedDocument",
    "WebReport",
    "build_changeset",
    "chunk_markdown",
    "extract_chunk",
    "ingest",
    "ingest_url",
    "is_url",
    "load_document",
]
