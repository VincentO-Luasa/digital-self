"""The ``brain`` CLI on tmp vaults with a fake model (no server, never the real vault)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from brain import cli
from brain.config import Settings
from brain.llm.fake import FakeLLM
from brain.vault import pending_reviews
from brain.vault.store import Vault

DOCS = Path(__file__).resolve().parents[1] / "fixtures" / "docs"
FIXTURE = sorted(
    (Path(__file__).resolve().parents[1] / "fixtures" / "llm" / "ingest").glob("alex-rivera-*.json")
)
runner = CliRunner()


def recorded(**kwargs) -> FakeLLM:
    """A fake replaying the recorded ingest of the fictional homepage: the split plan, then per
    part the gate, the extraction and the checks."""
    return FakeLLM.from_fixtures(FIXTURE, **kwargs)


def recorded_parts() -> int:
    return sum(json.loads(p.read_text())["kind"] == "structured" for p in FIXTURE) - 1


@pytest.fixture
def env(tmp_path: Path, git_env: None, monkeypatch: pytest.MonkeyPatch):
    """Settings pointing at a tmp vault, and a fake model the test can script."""
    settings = Settings(vault=tmp_path / "vault")
    state = {"llm": FakeLLM()}
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "make_llm", lambda _s: state["llm"])
    return settings, state


def invoke(*args: str, input: str | None = None):
    result = runner.invoke(cli.app, list(args), input=input)
    assert result.exit_code == 0, result.output
    return result


def test_init_status_reindex(env) -> None:
    settings, _ = env
    out = runner.invoke(cli.app, ["status"])
    assert out.exit_code == 0 and "missing" in out.output
    assert runner.invoke(cli.app, ["reindex"]).exit_code == 1  # no vault yet

    invoke("init-vault")
    assert (settings.vault / "SCHEMA.md").exists()
    assert "Name: The owner" in (settings.vault / "core.md").read_text()
    out = invoke("status").output
    assert "up" in out and "1 pages" in out and "0 pending" in out and "not built" in out
    out = invoke("reindex").output
    assert "index:" in out
    assert "items" in invoke("status").output


def test_init_vault_path_option(env, tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    invoke("init-vault", "--path", str(target))
    assert (target / "SCHEMA.md").exists()


def test_model_server_down(env) -> None:
    settings, state = env
    invoke("init-vault")
    state["llm"] = FakeLLM(healthy=False)
    result = runner.invoke(cli.app, ["ingest", str(DOCS / "alex-rivera.html")])
    assert result.exit_code == 1 and "not reachable" in result.output
    assert "down" in invoke("status").output


def test_ingest_review_and_chat(env) -> None:
    settings, state = env
    invoke("init-vault")
    state["llm"] = recorded(chat_responses=["You work at Northwind Labs."])
    out = invoke("ingest", str(DOCS / "alex-rivera.html")).output
    parts = recorded_parts()
    assert f"split into {parts} parts, planned by the model" in out
    plan = json.loads(
        json.loads(FIXTURE[0].read_text())["response"]["choices"][0]["message"]["content"]
    )
    assert f"1. {plan['parts'][0]['title']}" in out
    assert "gate p=" in out and "checks" in out  # decisions per chunk
    assert f"chunk 1/{parts}" in out and f"chunk {parts}/{parts}" in out
    assert "raw copy: raw/docs/" in out and "review: r-" in out
    vault = Vault(settings.vault)
    assert vault.page("orgs/northwind-labs") is not None

    listing = invoke("review", "--list").output
    review_id = pending_reviews(vault)[0].id
    assert review_id in listing and "ingest" in listing

    out = invoke("review", input="a\n").output
    assert "friend_of" in out and "approved" in out
    assert not pending_reviews(vault)
    sam = vault.page("people/sam-okafor")
    assert sam is not None and any(f.p == "friend_of" for f in sam.facts)
    assert "No pending reviews." in invoke("review").output

    out = invoke("chat", "-m", "Where do I work?").output
    assert "You work at Northwind Labs." in out and "transcript: raw/conversations/" in out


def test_ingest_url_and_bad_sources(env, monkeypatch: pytest.MonkeyPatch) -> None:
    settings, state = env
    invoke("init-vault")
    homepage = (DOCS / "alex-rivera.html").read_bytes()
    index = b'<html><body><h1>Alex</h1><a href="/cv">CV</a><a href="/gone">Gone</a></body></html>'

    def handler(request: httpx.Request) -> httpx.Response:
        pages = {"/": index, "/cv": homepage}
        if request.url.path not in pages:
            return httpx.Response(404)
        return httpx.Response(
            200, content=pages[request.url.path], headers={"content-type": "text/html"}
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(cli, "make_http", lambda: httpx.Client(transport=transport))
    empty = {"reasoning": "r", "owner_names": [], "entities": [], "facts": []}
    state["llm"] = recorded()
    state["llm"].structured_queue.appendleft(empty)  # the index page: one part, no decisions
    out = invoke("ingest", "https://alex.example/", "--follow", "2").output
    assert "page 1/3 https://alex.example/" in out and "page 2/3 https://alex.example/cv" in out
    assert "raw copy: raw/docs/2026" in out and "alex-example-cv.html" in out
    assert "skipped linked page https://alex.example/gone: HTTP" in out
    assert Vault(settings.vault).page("orgs/northwind-labs") is not None

    for args, message in (
        (["https://alex.example/missing"], "HTTP 404"),
        (["ftp://alex.example/cv"], "only http(s) URLs"),
        (["no/such/file.html"], "no such file"),
        ([str(DOCS / "alex-rivera.html"), "--follow", "1"], "--follow needs a URL"),
    ):
        result = runner.invoke(cli.app, ["ingest", *args])
        assert result.exit_code == 1 and message in result.output, result.output


def test_review_reject_and_skip(env) -> None:
    settings, state = env
    invoke("init-vault")
    state["llm"] = recorded()
    invoke("ingest", str(DOCS / "alex-rivera.html"))
    vault = Vault(settings.vault)
    invoke("review", input="s\n")
    assert len(pending_reviews(vault)) == 1
    invoke("review", input="r\nnot now\n")
    assert not pending_reviews(vault)
    done = list((settings.vault / "reviews" / "done").rglob("*.md"))
    assert len(done) == 1 and "reason: not now" in done[0].read_text()


def test_chat_repl_reads_until_quit(env) -> None:
    settings, state = env
    invoke("init-vault")
    state["llm"] = FakeLLM(chat_responses=["Hello Alex."])
    out = invoke("chat", input="hi\n/big\n/mem\n/quit\n").output
    assert "Hello Alex." in out
    assert "Phase 3" in out
    assert "Memory notes" in out
    assert "transcript:" in out


def test_ingest_prints_skipped_parts(env, tmp_path: Path) -> None:
    from brain.llm.decide import Decision

    invoke("init-vault")
    _, state = env
    doc = tmp_path / "page.md"
    tutorial = " ".join(f"Partitioning step {i} splits a table by a key." for i in range(26))
    doc.write_text(
        "# Alex Rivera\n\n## Career\n\n"
        + "I am a data engineer at Northwind Labs in Porto. " * 20
        + "\n\n## Quick guide\n\n"
        + tutorial
        + "\n"
    )
    plan = {
        "reasoning": "r",
        "parts": [{"first_block": 0, "title": "Career"}, {"first_block": 3, "title": "Guide"}],
    }
    empty = {"reasoning": "r", "owner_names": [], "entities": [], "facts": []}

    def gate(text, questions):
        p = 0.02 if "Partitioning" in text else 0.99
        return [
            Decision(
                name=q.name, type="predicate", choice="yes", probabilities={"yes": p, "no": 1 - p}
            )
            for q in questions
        ]

    state["llm"] = FakeLLM(structured_responses=[plan, empty], decider=gate)
    out = " ".join(invoke("ingest", str(doc)).output.split())  # rich wraps long lines
    assert "skipped, nothing about a person (p=0.020" in out
    assert "skipped part with nothing about a person" in out and "gate: skipped" in out
