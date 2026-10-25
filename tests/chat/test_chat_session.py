"""ChatSession without a terminal: prompt layout, memory packing, history, transcript, commands."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from brain.chat.memory import EMPTY, HEADER, pack
from brain.chat.session import BIG_MESSAGE, HISTORY_TRIM, INTERRUPTED, ChatSession
from brain.config import Settings
from brain.index.db import open_index
from brain.index.reindex import reindex
from brain.index.search import Hit
from brain.llm.fake import FakeLLM
from brain.prompts import load_prompt
from brain.vault.generated import refresh_generated
from brain.vault.raw import parse_transcript
from brain.vault.store import Vault


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 25, 20, 0)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=5)
        return self.now


@pytest.fixture
def session_factory(tmp_vault: Vault):
    refresh_generated(tmp_vault, today=datetime(2026, 10, 25).date())
    settings = Settings(vault=tmp_vault.root)

    def make(llm: FakeLLM, **kw) -> ChatSession:
        db = open_index(tmp_vault.db_path, embed_model=llm.embed_model, dim=llm.embed_dim)
        reindex(tmp_vault, db, llm)
        return ChatSession(tmp_vault, llm, db, settings, clock=Clock(), **kw)

    return make


def last_messages(llm: FakeLLM) -> list[dict]:
    return [c for c in llm.calls if c[0] == "chat"][-1][1]["messages"]


def test_turn_streams_and_lays_out_the_prompt(session_factory) -> None:
    llm = FakeLLM(chat_responses=["You lead the Atlas Migration at Northwind Labs."])
    session = session_factory(llm)
    pieces = list(session.turn("How is the Atlas go-live going?"))
    assert len(pieces) > 1 and "".join(pieces).startswith("You lead")

    system, user = last_messages(llm)
    assert system["role"] == "system"
    assert system["content"].startswith(load_prompt("chat_system"))
    assert "Name: Alex Rivera" in system["content"]  # core.md is in the cached prefix
    assert "<!--" not in system["content"]
    assert user["content"].startswith("<memory>\nToday is 2026-10-25.\n" + HEADER)
    assert user["content"].endswith("</memory>\n\nHow is the Atlas go-live going?")
    assert "2026-11-15" in user["content"]  # the deadline fact was retrieved
    assert session.last_memory is not None and "f-q5r6s7" in session.last_memory.used


def test_history_is_plain_and_trimmed_in_blocks(session_factory) -> None:
    llm = FakeLLM()
    session = session_factory(llm, record=False)
    system_first = None
    for i in range(9):
        session.ask(f"message {i}")
        system_first = system_first or last_messages(llm)[0]
        assert last_messages(llm)[0] == system_first  # the prefix never changes
    # 9 exchanges > 8: the oldest 4 were dropped at once
    assert len(session.history) == 2 * (9 - HISTORY_TRIM)
    assert session.history[0] == {"role": "user", "content": f"message {HISTORY_TRIM}"}
    assert all("<memory>" not in m["content"] for m in session.history)
    assert session.transcript is None


def test_transcript_records_each_turn(session_factory, tmp_vault: Vault) -> None:
    llm = FakeLLM(chat_responses=["Northwind Labs.", "No idea yet, tell me!"])
    session = session_factory(llm)
    session.ask("Where do I work?")
    session.ask("What is my favourite film?")
    assert session.transcript is not None
    _, turns = parse_transcript(session.transcript.path.read_text())
    assert [(t.role, t.text) for t in turns] == [
        ("user", "Where do I work?"),
        ("assistant", "Northwind Labs."),
        ("user", "What is my favourite film?"),
        ("assistant", "No idea yet, tell me!"),
    ]
    assert session.close()
    assert session.transcript.rel.startswith("raw/conversations/2026/10/conv-20261025-")


def test_interrupted_answer_is_kept(session_factory) -> None:
    llm = FakeLLM(chat_responses=["one two three four"])
    session = session_factory(llm)
    stream = session.turn("Count for me")
    next(stream)
    stream.close()
    assert session.history[-1]["content"] == f"one {INTERRUPTED}"
    _, turns = parse_transcript(session.transcript.path.read_text())
    assert turns[-1].text == f"one {INTERRUPTED}"


def test_commands(session_factory) -> None:
    session = session_factory(FakeLLM(), record=False)
    assert session.command("hello") is None
    assert session.command("/mem").output == "No memory retrieved yet."
    assert session.command("/big").output == BIG_MESSAGE
    assert session.command("/quit").quit
    assert "Unknown command" in session.command("/nope").output
    session.ask("Atlas deadline")
    mem = session.command("/mem").output
    assert mem.startswith(HEADER) and "used: " in mem


def hit(id_: str, kind: str, text: str, date: str | None = None) -> Hit:
    return Hit(
        id=id_,
        kind=kind,
        entity_id=None,
        text=text,
        score=1.0,
        privacy="normal",
        sources=(),
        date=date,
    )


def test_pack_is_gist_first_and_budgeted() -> None:
    long_summary = "Talked through the Atlas schedule. " + "Detail. " * 40
    hits = [
        hit(
            "ep-1",
            "episode",
            "Alex worries about the Atlas deadline\n" + long_summary,
            "2026-10-25T14:12",
        ),
        hit("f-1", "fact", "Atlas go-live is planned for 2026-11-15"),
        hit("f-2", "fact", "Lives in Porto (since 2024)"),
        hit("summary:x", "summary", "Northwind Labs is a company. It is mid-sized."),
    ]
    block = pack(hits, 1000, skip_text="- Lives in Porto (since 2024)\n")
    assert block.used == ["ep-1", "f-1", "summary:x"]  # f-2 is already in the core card
    assert "- [ep-1] (2026-10-25) Alex worries about the Atlas deadline" in block.text
    assert "Detail." in block.text  # budget left: the top episode is expanded
    small = pack(hits, 40)
    assert small.used and "Detail." not in small.text and small.tokens <= 40
    assert pack([], 1000).text == EMPTY


def test_pack_skips_facts_the_card_rewords() -> None:
    card = "Name: Alex Rivera · lives in Porto · speaks English, Spanish"
    hits = [
        hit("f-1", "fact", "Speaks Spanish"),
        hit("f-2", "fact", "Lived in Lisbon"),  # past city: not in the card
        hit("ep-1", "episode", "Speaks Spanish with Sam on Thursdays"),  # episodes are kept
    ]
    assert pack(hits, 1000, skip_text=card).used == ["f-2", "ep-1"]
