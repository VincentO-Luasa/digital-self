"""Interview fixtures (helpers in interview_helpers.py)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import pytest
from interview_helpers import TODAY, Env, build_env  # type: ignore[import-not-found]

from brain.vault.store import Vault


@pytest.fixture
def env(tmp_vault: Vault) -> Env:
    e = build_env(tmp_vault.root)
    yield e
    e.db.close()  # type: ignore[attr-defined]


@pytest.fixture
def env_factory(tmp_vault: Vault) -> Callable[[date], Env]:
    made: list[Env] = []

    def make(today: date = TODAY) -> Env:
        e = build_env(tmp_vault.root, today)
        made.append(e)
        return e

    yield make
    for e in made:
        e.db.close()  # type: ignore[attr-defined]
