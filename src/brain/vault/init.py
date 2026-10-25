"""``brain init-vault``: create (or complete) a vault without touching existing user files."""

from __future__ import annotations

import json
import shlex
import shutil
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from brain.config import REPO_DIR
from brain.vault import gitops
from brain.vault.models import SCHEMA_VERSION, SELF_ID, Page
from brain.vault.render import render_page
from brain.vault.store import SELF_PAGE, WIKI_DIRS

LAYOUT = [
    "raw/docs",
    "raw/conversations",
    "raw/external",
    "raw/inbox",
    "episodes",
    *(f"wiki/{d}" for d in WIKI_DIRS),
    "journal",
    "reviews/pending",
    "reviews/done",
    ".brain",
    ".privacy",
]
ROOT_FILES = {
    "core.md": "# Core\n",
    "index.md": "# Index\n",
    "log.md": "# Log\n",
    "questions.md": "# Questions\n",
    "intentions.md": "# Intentions\n",
    ".privacy/denylist.txt": (
        "# Terms the code repository's pre-commit hook refuses to commit, one per line.\n"
        "# brain adds the titles and aliases of private pages; add your own below.\n"
    ),
}
GITIGNORE_LINES = [".brain/", ".DS_Store"]
TEMPLATE_SKIP = {Path("README.md")}  # documents the template itself; not installed
JUNK = {".DS_Store", "__pycache__"}


@dataclass
class InitResult:
    root: Path
    created: list[Path] = field(default_factory=list)
    updated: list[Path] = field(default_factory=list)
    kept: list[Path] = field(default_factory=list)  # existing files left untouched
    warnings: list[str] = field(default_factory=list)
    commit: str | None = None


def _schema_md(repo_dir: Path) -> str:
    return (
        f"---\nschema_version: {json.dumps(SCHEMA_VERSION)}\n---\n"
        "# Vault schema\n\n"
        f"This vault follows the brain's vault schema v{SCHEMA_VERSION}. The canonical spec is\n"
        f"`docs/vault-schema.md` in the code repository ({repo_dir}).\n\n"
        "## Vault-local conventions\n\nNone yet.\n"
    )


def _profile(today: date) -> str:
    page = Page(
        id=SELF_PAGE,
        type="self",
        subject=SELF_ID,
        title="Profile",
        regions=["identity"],
        created=today,
        updated=today,
    )
    return render_page(page)


def _write_new(path: Path, text: str, result: InitResult) -> None:
    if path.exists():
        result.kept.append(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    result.created.append(path)


def _ensure_gitignore(root: Path, result: InitResult) -> None:
    path = root / ".gitignore"
    if not path.exists():
        _write_new(path, "\n".join(GITIGNORE_LINES) + "\n", result)
        return
    text = path.read_text(encoding="utf-8")
    missing = [line for line in GITIGNORE_LINES if line not in text.splitlines()]
    if missing:
        path.write_text(text.rstrip("\n") + "\n" + "\n".join(missing) + "\n", encoding="utf-8")
        result.updated.append(path)
    else:
        result.kept.append(path)


# --- template ----------------------------------------------------------------------------------


def _uv_command(command: str, uv: str) -> str:
    if command == "uv":
        return uv
    if command.startswith("uv "):
        return shlex.quote(uv) + command[2:]
    return command


def _swap_uv(node: Any, uv: str) -> Any:
    if isinstance(node, dict):
        return {
            k: _uv_command(v, uv) if k == "command" and isinstance(v, str) else _swap_uv(v, uv)
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_swap_uv(v, uv) for v in node]
    return node


def _render_template(
    src: Path, repo_dir: Path, vault: Path, uv: str | None, result: InitResult
) -> bytes | str:
    try:
        text = src.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return src.read_bytes()
    is_json = src.suffix == ".json"

    def esc(p: Path) -> str:
        return json.dumps(str(p), ensure_ascii=False)[1:-1] if is_json else str(p)

    text = text.replace("{{REPO_DIR}}", esc(repo_dir)).replace("{{VAULT_DIR}}", esc(vault))
    if not is_json:
        return text
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        result.warnings.append(f"{src.name}: not valid JSON, copied as is")
        return text
    if uv is None:
        if _swap_uv(data, "/abs/uv") != data:
            result.warnings.append(f"{src.name}: uv not found on PATH; kept the bare 'uv' command")
        return text
    swapped = _swap_uv(data, uv)
    return text if swapped == data else json.dumps(swapped, indent=2, ensure_ascii=False) + "\n"


def _template_files(template: Path) -> list[Path]:
    files = []
    for path in sorted(template.rglob("*")):
        rel = path.relative_to(template)
        if path.is_file() and rel not in TEMPLATE_SKIP and not JUNK & set(rel.parts):
            files.append(path)
    return files


def _install_template(repo_dir: Path, root: Path, update: bool, result: InitResult) -> None:
    template = repo_dir / "vault_template"
    if not template.is_dir():
        result.warnings.append(f"no template at {template}; skipped")
        return
    uv = shutil.which("uv")
    for src in _template_files(template):
        dest = root / src.relative_to(template)
        content = _render_template(src, repo_dir, root, uv, result)
        data = content.encode("utf-8") if isinstance(content, str) else content
        if dest.exists() and (not update or dest.read_bytes() == data):
            result.kept.append(dest)
            continue
        existed = dest.exists()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        shutil.copymode(src, dest)
        (result.updated if existed else result.created).append(dest)


# --- entry point -------------------------------------------------------------------------------


def init_vault(
    path: Path,
    repo_dir: Path = REPO_DIR,
    *,
    update_template: bool = False,
    today: date | None = None,
) -> InitResult:
    """Create the vault layout, root files, template and git repo. Idempotent: existing user
    files are never overwritten; template files only when ``update_template`` is True."""
    root = Path(path).expanduser().resolve()
    repo_dir = Path(repo_dir).expanduser().resolve()
    if root == repo_dir or root.is_relative_to(repo_dir):
        raise ValueError(f"refusing to create a vault inside the code repository: {root}")
    today = today or date.today()
    result = InitResult(root=root)
    for rel in LAYOUT:
        (root / rel).mkdir(parents=True, exist_ok=True)
    _write_new(root / "SCHEMA.md", _schema_md(repo_dir), result)
    for rel, text in ROOT_FILES.items():
        _write_new(root / rel, text, result)
    _ensure_gitignore(root, result)
    _write_new(root / "wiki" / f"{SELF_PAGE}.md", _profile(today), result)
    _install_template(repo_dir, root, update_template, result)
    first = not gitops.is_repo(root)
    message = "brain: init vault" if first else "brain: update vault (init-vault)"
    result.commit = gitops.commit_all(root, message)
    return result
