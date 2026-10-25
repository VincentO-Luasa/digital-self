"""Shared fixtures. Vaults are built from tests/fixtures/ (fictional persona) in tmp dirs."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from brain.vault import gitops
from brain.vault.store import Vault

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_VAULT = FIXTURES / "vault"


@pytest.fixture
def git_env(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Isolate git from the developer's global config (hooks, signing, identity)."""
    home = tmp_path_factory.mktemp("git-home")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(home / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def tmp_vault(tmp_path: Path, git_env: None) -> Vault:
    """A git-initialised copy of the fixture vault."""
    root = tmp_path / "vault"
    shutil.copytree(FIXTURE_VAULT, root)
    gitops.commit_all(root, "fixture vault")
    return Vault(root)
