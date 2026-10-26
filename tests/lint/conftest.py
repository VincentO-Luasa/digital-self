"""Lint fixtures: the eval's clean and dirty fictional vaults (Alex Rivera persona).

Recorded decisions (``tests/fixtures/lint/llm``) are replayed by ``linttools.Replay``. Re-record
after changing a prompt, a question or the persona:
``BRAIN_RECORD_FIXTURES=1 uv run pytest -m llm tests/lint/test_lint_live.py -s``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from linttools import REPLAY_SEED  # type: ignore[import-not-found]

from brain.eval import lint_eval as ev
from brain.vault.store import Vault


@pytest.fixture
def clean_vault(tmp_path: Path, git_env: None) -> Vault:
    return ev.clean_vault(tmp_path / "vault")


@pytest.fixture
def dirty(tmp_path: Path, git_env: None) -> tuple[Vault, ev.Gold]:
    vault = ev.clean_vault(tmp_path / "vault")
    return vault, ev.inject(vault, REPLAY_SEED)
