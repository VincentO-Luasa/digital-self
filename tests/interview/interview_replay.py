"""What the live interview test records and the replay test checks."""

from __future__ import annotations

from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "interview" / "llm"
REGIONS = ("career.current_job", "projects", "self.values_goals")
REPLY = {
    "projects": "We went live in December, a few weeks late. Now I'm cleaning up the old cluster."
}
