"""Document ingest: load, keep a raw copy, chunk, extract operations, apply, reindex."""

from brain.ingest.chunking import Chunk, chunk_markdown
from brain.ingest.extract import Extraction, extract_chunk
from brain.ingest.loaders import Document, UnsupportedDocument, load_document
from brain.ingest.normalise import Normalised, build_changeset
from brain.ingest.pipeline import IngestReport, ingest

__all__ = [
    "Chunk",
    "Document",
    "Extraction",
    "IngestReport",
    "Normalised",
    "UnsupportedDocument",
    "build_changeset",
    "chunk_markdown",
    "extract_chunk",
    "ingest",
    "load_document",
]
