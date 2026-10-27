"""What the live interview test records and the replay test checks."""

from __future__ import annotations

from pathlib import Path

from brain.llm.recorder import Recorder

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "interview" / "llm"
REGIONS = ("career.current_job", "projects", "self.values_goals")
REPLY = {
    "projects": "We went live in December, a few weeks late. Now I'm cleaning up the old cluster."
}


class SeqRecorder(Recorder):
    """Numbers the exchanges of one turn, so file names sort in call order."""

    def __init__(self, directory: Path, *, prefix: str = "") -> None:
        super().__init__(directory, prefix=prefix)
        self.base = prefix
        self.n = 0

    def save(self, kind: str, request: dict, response: dict) -> Path:
        self.n += 1
        self.prefix = f"{self.base}{self.n}-"
        return super().save(kind, request, response)
