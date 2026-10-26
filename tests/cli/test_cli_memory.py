"""``brain chat`` remembering on close, ``--no-memory`` and ``brain memory process``, on tmp
vaults with a scripted fake model (fictional persona)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from brain import cli
from brain.config import Settings
from brain.llm.fake import FakeLLM
from brain.vault.store import Vault

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "memory"))
from memtools import Decider, extraction, segmentation  # type: ignore[import-not-found]  # noqa: E402

runner = CliRunner()
SEGMENTS = segmentation(
    {
        "first_turn": 1,
        "last_turn": 2,
        "topic": "bouldering",
        "regions": ["self.routine"],
        "gist": "Alex started bouldering twice a week",
    }
)
FACTS = extraction([{"subject": "self", "predicate": "habit", "object": "Bouldering twice a week"}])


@pytest.fixture
def env(tmp_path: Path, git_env: None, monkeypatch: pytest.MonkeyPatch):
    settings = Settings(vault=tmp_path / "vault")
    state = {"llm": FakeLLM()}
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "make_llm", lambda _s: state["llm"])
    assert runner.invoke(cli.app, ["init-vault"]).exit_code == 0
    return settings, state


def invoke(*args: str):
    result = runner.invoke(cli.app, list(args))
    assert result.exit_code == 0, result.output
    return result


def test_chat_remembers_when_the_session_closes(env) -> None:
    settings, state = env
    state["llm"] = FakeLLM(
        chat_responses=["Nice!"], structured_responses=[SEGMENTS, FACTS], decider=Decider()
    )
    out = invoke("chat", "-m", "I started bouldering twice a week.").output
    assert "memory: 1 episode · +1 facts · 0 superseded · 0 queued" in out
    vault = Vault(settings.vault)
    assert [e.status for e in vault.episodes()] == ["extracted"]
    assert any(f.p == "habit" for p in vault.pages() for f in p.facts)
    assert invoke("memory", "process", "--pending").output.strip() == "Nothing to process."


def test_no_memory_then_process_pending(env) -> None:
    settings, state = env
    state["llm"] = FakeLLM(chat_responses=["Nice!"])
    out = invoke("chat", "--no-memory", "-m", "I started bouldering twice a week.").output
    assert "memory: skipped" in out
    vault = Vault(settings.vault)
    assert not vault.episodes()
    state["llm"] = FakeLLM(structured_responses=[SEGMENTS, FACTS], decider=Decider())
    out = invoke("memory", "process", "--pending").output
    assert "raw/conversations/" in out and "1 episode" in out
    assert [e.status for e in vault.episodes()] == ["extracted"]
    transcript = next((settings.vault / "raw" / "conversations").rglob("*.md"))
    assert "already processed" in invoke("memory", "process", str(transcript)).output


def test_memory_process_needs_a_transcript_or_pending(env) -> None:
    result = runner.invoke(cli.app, ["memory", "process"])
    assert result.exit_code == 1 and "TRANSCRIPT or --pending" in result.output
