"""A read-only chat session: prompt assembly, retrieval, streaming, transcript capture.

Prompt layout (docs/retrieval.md), ordered for prefix caching:

    [system prompt + core.md]          stable for the whole session → cached prefix
    [history, trimmed 4 turns at a time]
    [memory block + user turn]         the latest user message only

History keeps the plain user text, so earlier turns never change once sent. Phase 1 has no
router or escalation: the model is told to say it doesn't know.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Protocol

from brain.chat.memory import MemoryBlock, pack
from brain.config import Settings
from brain.index.search import search
from brain.llm.base import RetrievalModel
from brain.prompts import load_prompt
from brain.vault.generated import GENERATED_NOTE, build_core
from brain.vault.raw import Role, Transcript
from brain.vault.store import Vault

HISTORY_MAX_TURNS = 8  # user/assistant exchanges kept in the prompt
HISTORY_TRIM = 4  # dropped at once, so the cached prefix survives between trims
MAX_ANSWER_TOKENS = 1024
INTERRUPTED = "[interrupted]"
BIG_MESSAGE = "Escalation to the big model arrives in Phase 3."
HELP = "Commands: /mem (last memory block), /big (ask the big model), /quit"


class ChatModel(RetrievalModel, Protocol):
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 1024,
        tools: Any = None,
        stream: bool = False,
    ) -> Any: ...


@dataclass(frozen=True)
class CommandResult:
    output: str
    quit: bool = False


def core_for_prompt(core: str) -> str:
    """core.md without its generated-file comment and top heading."""
    lines = [
        line
        for line in core.strip().split("\n")
        if line.strip() not in (GENERATED_NOTE, "# Core") and not line.startswith("<!--")
    ]
    return "\n".join(lines).strip()


class ChatSession:
    def __init__(
        self,
        vault: Vault,
        llm: ChatModel,
        db: sqlite3.Connection,
        settings: Settings,
        *,
        record: bool = True,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.vault = vault
        self.llm = llm
        self.db = db
        self.settings = settings
        self.record = record
        self.clock = clock
        if vault.core_path.exists():
            core = vault.core_path.read_text(encoding="utf-8")
        else:
            core = build_core(vault, max_tokens=settings.core_max_tokens, today=clock().date())
        self.core = core_for_prompt(core)
        self.system = load_prompt("chat_system") + "\n# Owner profile\n\n" + self.core + "\n"
        self.history: list[dict[str, str]] = []
        self.transcript: Transcript | None = None
        self.last_memory: MemoryBlock | None = None

    # -- prompt -----------------------------------------------------------------------------

    def retrieve(self, text: str) -> MemoryBlock:
        hits = search(
            self.db,
            self.llm,
            text,
            k=self.settings.retrieval.rerank_top,
            include_private=True,  # local chat: private memories never leave the machine here
            settings=self.settings.retrieval,
        )
        return pack(hits, self.settings.retrieval.budget_tokens, skip_text=self.core)

    def build_messages(self, text: str, block: MemoryBlock, today: date) -> list[dict[str, str]]:
        latest = f"<memory>\nToday is {today.isoformat()}.\n{block.text}\n</memory>\n\n{text}"
        return [
            {"role": "system", "content": self.system},
            *self.history,
            {"role": "user", "content": latest},
        ]

    def _trim(self) -> None:
        if len(self.history) > 2 * HISTORY_MAX_TURNS:
            self.history = self.history[2 * HISTORY_TRIM :]

    # -- transcript ---------------------------------------------------------------------------

    def _record(self, role: Role, text: str) -> None:
        if not self.record:
            return
        if self.transcript is None:
            meta = {"model": self.settings.llm.chat_model}
            self.transcript = Transcript.start(self.vault, now=self.clock(), meta=meta)
        self.transcript.append(role, text, at=self.clock())

    def close(self) -> str | None:
        """Commit the transcript (if any turn was recorded)."""
        return self.transcript.commit() if self.transcript else None

    # -- turns --------------------------------------------------------------------------------

    def turn(self, text: str) -> Iterator[str]:
        """Stream the answer to one user message (retrieval runs on first iteration)."""
        text = text.strip()
        block = self.retrieve(text)
        self.last_memory = block
        messages = self.build_messages(text, block, self.clock().date())
        self._record("user", text)
        pieces: list[str] = []
        finished = False
        try:
            stream = self.llm.chat(messages, stream=True, max_tokens=MAX_ANSWER_TOKENS)
            for delta in stream:
                pieces.append(delta)
                yield delta
            finished = True
        finally:
            answer = "".join(pieces).strip()
            if not finished:
                answer = f"{answer} {INTERRUPTED}".strip()
            self.history += [
                {"role": "user", "content": text},
                {"role": "assistant", "content": answer},
            ]
            self._trim()
            self._record("assistant", answer)

    def ask(self, text: str) -> str:
        return "".join(self.turn(text))

    def command(self, text: str) -> CommandResult | None:
        """Handle ``/quit``, ``/mem``, ``/big``, ``/help``; None if ``text`` is not a command."""
        word = text.strip().split(maxsplit=1)[0].casefold() if text.strip() else ""
        if not word.startswith("/"):
            return None
        if word in ("/quit", "/exit", "/q"):
            return CommandResult("Bye.", quit=True)
        if word == "/big":
            return CommandResult(BIG_MESSAGE)
        if word == "/mem":
            return CommandResult(self.describe_memory())
        if word == "/help":
            return CommandResult(HELP)
        return CommandResult(f"Unknown command {word}. {HELP}")

    def describe_memory(self) -> str:
        block = self.last_memory
        if block is None:
            return "No memory retrieved yet."
        unused = [h.id for h in block.hits if h.id not in block.used]
        lines = [block.text, "", f"~{block.tokens} tokens · used: {', '.join(block.used) or '-'}"]
        if unused:
            lines.append(f"retrieved but not packed: {', '.join(unused)}")
        return "\n".join(lines)
