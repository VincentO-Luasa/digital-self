"""``brain interview`` through typer, with a fake model and a fake memory pipeline.

The command is registered in ``brain.cli`` by the lead; here it gets its own app.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from interview_helpers import ScriptedLLM  # type: ignore[import-not-found]
from typer.testing import CliRunner

from brain import cli as main
from brain.config import Settings
from brain.interview import cli as icli

runner = CliRunner()
app = typer.Typer()
app.command("interview")(icli.interview)
app.command("noop")(lambda: None)  # keep typer from collapsing to a single command


def decision(label: str) -> list[dict]:
    return [
        {"name": "answer", "type": "choice", "choice": label, "probabilities": {label: 1.0}},
        {"name": "new_thread", "type": "predicate", "choice": "no", "probabilities": {"yes": 0.0}},
    ]


@pytest.fixture
def cli_env(env, monkeypatch: pytest.MonkeyPatch):
    state: dict = {"llm": ScriptedLLM(), "calls": []}
    monkeypatch.setattr(main, "get_settings", lambda: Settings(vault=env.vault.root))
    monkeypatch.setattr(main, "make_llm", lambda settings: state["llm"])

    def process(vault, llm, transcript, **kwargs):
        state["calls"].append((transcript.rel, kwargs))
        return None

    monkeypatch.setattr(icli, "make_process", lambda: process)
    state["env"] = env
    return state


def conversations(env) -> list[Path]:
    return sorted((env.vault.raw_dir / "conversations").rglob("*.md"))


def test_coverage_and_plan_are_read_only(cli_env) -> None:
    result = runner.invoke(app, ["interview", "--coverage"])
    assert result.exit_code == 0, result.output
    assert "self.values_goals" in result.output and "private" in result.output
    assert "next interview: self.values_goals" in result.output
    result = runner.invoke(app, ["interview", "relationships.work", "--plan"])
    assert result.exit_code == 0, result.output
    assert "1. [ask] Colleagues & network: Maya Lindqvist: How do you work with Maya" in (
        result.output.replace("\n", " ")
    )
    assert not conversations(cli_env["env"])
    assert not [c for c in cli_env["llm"].calls if c[0] in ("structured", "decide")]


def test_unknown_region_and_server_down(cli_env) -> None:
    result = runner.invoke(app, ["interview", "garden"])
    assert result.exit_code == 1 and "unknown region" in result.output
    cli_env["llm"] = ScriptedLLM(healthy=False)
    result = runner.invoke(app, ["interview"])
    assert result.exit_code == 1 and "not reachable" in result.output
    result = runner.invoke(app, ["interview", "--coverage"])  # the existing index is enough
    assert result.exit_code == 0, result.output


def test_scripted_interview(cli_env, tmp_path: Path) -> None:
    cli_env["llm"] = ScriptedLLM(
        structured_responses=[
            {"reasoning": "r", "message": "Who do you report to?"},
            {"reasoning": "r", "message": "What does a usual week look like?"},
        ],
        decisions=[decision("answered")],
    )
    answers = tmp_path / "answers.txt"
    answers.write_text("Theo Marchetti.\n\nstop\n", encoding="utf-8")
    result = runner.invoke(app, ["interview", "current job", "--answers", str(answers)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Interview: Current job" in out
    assert "brain › Who do you report to?" in out
    assert "you › Theo Marchetti." in out and "you › stop" in out
    assert "transcript: raw/conversations/" in out
    [(rel, kwargs)] = cli_env["calls"]
    assert kwargs["channel"] == "interview" and kwargs["regions_hint"] == ["career.current_job"]
    assert [
        p.relative_to(cli_env["env"].vault.root).as_posix() for p in conversations(cli_env["env"])
    ] == [rel]
    assert "(origin:: coverage)" in cli_env["env"].vault.questions_path.read_text()


def test_missing_pipeline_keeps_the_transcript(cli_env, tmp_path: Path, monkeypatch) -> None:
    def missing():
        raise ImportError("no brain.memory yet")

    monkeypatch.setattr(icli, "make_process", missing)
    cli_env["llm"] = ScriptedLLM(
        structured_responses=[{"reasoning": "r", "message": "What matters most to you?"}] * 3,
        decisions=[decision("answered")],
    )
    answers = tmp_path / "answers.txt"
    answers.write_text("Honesty.\n", encoding="utf-8")  # input ends: the session closes
    result = runner.invoke(app, ["interview", "--answers", str(answers)])
    assert result.exit_code == 0, result.output
    assert "memory pipeline failed" in result.output and "retries it" in result.output
    assert len(conversations(cli_env["env"])) == 1
