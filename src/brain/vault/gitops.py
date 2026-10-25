"""Thin wrappers over ``git`` for the vault's local-only repository. Never pushes."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import NoReturn

FALLBACK_NAME = "brain"
FALLBACK_EMAIL = "brain@localhost"


class GitError(RuntimeError):
    pass


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc


def is_repo(root: Path) -> bool:
    return (root / ".git").exists()


def init(root: Path) -> None:
    """Create a local repository (no remote is ever added)."""
    if not is_repo(root):
        _git(root, "init", "-q")


def add_all(root: Path) -> None:
    _git(root, "add", "-A")


def has_staged_changes(root: Path) -> bool:
    return _git(root, "diff", "--cached", "--quiet", check=False).returncode == 1


def _identity(root: Path) -> list[str]:
    """Use the configured identity; fall back to a local one so commits never fail."""
    args: list[str] = ["-c", "commit.gpgsign=false"]
    if not _git(root, "config", "user.name", check=False).stdout.strip():
        args += ["-c", f"user.name={FALLBACK_NAME}"]
    if not _git(root, "config", "user.email", check=False).stdout.strip():
        args += ["-c", f"user.email={FALLBACK_EMAIL}"]
    return args


def commit(root: Path, message: str) -> str | None:
    """Commit staged changes; return the new sha, or None when there was nothing to commit."""
    if not has_staged_changes(root):
        return None
    _git(root, *_identity(root), "commit", "-q", "-m", message)
    return head(root)


def commit_all(root: Path, message: str) -> str | None:
    """``git add -A`` + commit, initialising the repository if needed."""
    init(root)
    add_all(root)
    return commit(root, message)


def head(root: Path) -> str | None:
    proc = _git(root, "rev-parse", "HEAD", check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def remotes(root: Path) -> list[str]:
    return _git(root, "remote", check=False).stdout.split()


def push(*_args: object, **_kwargs: object) -> NoReturn:
    raise GitError("the vault is local-only: brain never pushes it anywhere")
