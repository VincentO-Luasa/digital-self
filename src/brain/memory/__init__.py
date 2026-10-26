"""Memory pipeline: transcript → topic episodes → facts, journal lines and intentions
(docs/architecture.md, Flow 3). ``process_session`` is the entry point."""

from brain.memory.pipeline import (
    EpisodeReport,
    SessionReport,
    is_processed,
    pending_transcripts,
    process_session,
)

__all__ = [
    "EpisodeReport",
    "SessionReport",
    "is_processed",
    "pending_transcripts",
    "process_session",
]
