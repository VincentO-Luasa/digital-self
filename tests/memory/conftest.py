"""Memory pipeline fixtures: a vault with a pinned life map (helpers live in memtools.py).

Fictional persona only (Alex Rivera, Northwind Labs, Sam Okafor, Atlas Migration, Contoso).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from brain.vault import gitops
from brain.vault.store import Vault

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
LIFEMAP = FIXTURES / "memory" / "life-map.yaml"  # pinned: the prompt lists the regions


@pytest.fixture
def mvault(tmp_path: Path, git_env: None) -> Vault:
    """The fixture vault with a pinned life map (a copy of config/life-map.yaml at the time the
    fixtures were recorded), committed."""
    root = tmp_path / "vault"
    shutil.copytree(FIXTURES / "vault", root)
    shutil.copy(LIFEMAP, root / "life-map.yaml")
    gitops.commit_all(root, "fixture vault")
    return Vault(root)
