"""Chat turns against the real local model on the fixture vault. ``uv run pytest -m llm``."""

from __future__ import annotations

import time
from datetime import date

import pytest

from brain.chat.session import ChatSession
from brain.config import Settings, load_settings
from brain.index.db import open_index
from brain.index.reindex import reindex
from brain.llm import LLM
from brain.vault.generated import refresh_generated
from brain.vault.store import Vault

pytestmark = pytest.mark.llm


@pytest.fixture
def session(tmp_vault: Vault):
    client = LLM(load_settings().llm)
    if not client.health():
        client.close()
        pytest.skip("local model server is not running")
    refresh_generated(tmp_vault, today=date(2026, 10, 25))
    db = open_index(tmp_vault.db_path, embed_model=client.embed_model, dim=client.embed_dim)
    reindex(tmp_vault, db, client)
    yield ChatSession(tmp_vault, client, db, Settings(vault=tmp_vault.root))
    client.close()


def test_answers_from_memory_and_admits_gaps(session: ChatSession) -> None:
    start = time.perf_counter()
    answer = session.ask("When is the Atlas go-live planned?")
    print(f"\n[{time.perf_counter() - start:.1f}s] {answer}")
    assert ("15" in answer and "Nov" in answer) or "2026-11-15" in answer

    start = time.perf_counter()
    answer = session.ask("What is the name of my dentist?")
    print(f"[{time.perf_counter() - start:.1f}s] {answer}")
    lowered = answer.casefold()
    assert any(
        w in lowered
        for w in ("don't know", "do not know", "not sure", "no record", "don't have", "haven't")
    )
    assert session.close()
