"""One interview: ask one thing at a time, follow up, and hand the transcript to the memory
pipeline at the end (docs/architecture.md, Flow 2).

Per owner answer, at most two model calls:

1. ``LLM.decide`` classifies the answer (``ANSWER``: answered / partial / declined / unclear)
   and asks whether it opened something worth a follow-up (``NEW_THREAD``). Code then decides:
   follow up (partial, or a new thread, at most ``MAX_FOLLOW_UPS`` per step), clarify once
   (unclear), or move on (declined, or nothing new). ``MAX_DECLINES`` declines in a row, the end
   of the plan, ``minutes`` or ``max_questions`` end the session.
2. A structured call writes the next message (``interview_system`` + core card as the cached
   prefix, then the conversation, then a PLAN block on the latest message). Code checks it is
   grounded (every name and number in it appears in what the brain knows or was told) and asks
   one thing; otherwise a template built from the hint is used.

Owner commands: ``skip`` (next step), ``private`` or ``private: <answer>`` (the answers of this
step are private: never shared with the big model), ``stop``.

The transcript lands in ``raw/conversations/`` with ``channel: interview``. System turns carry
machine-readable markers for the memory pipeline: ``[interview] region:: … · slot:: …`` before
each question and ``[interview] privacy:: private …`` before each private answer.
"""

from __future__ import annotations

import inspect
import logging
import re
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from brain.chat.session import core_for_prompt
from brain.config import Settings
from brain.interview.coverage import CoverageReport
from brain.interview.planner import PlanItem, context_lines
from brain.interview.questions import refresh_questions, resolve_question
from brain.llm.decide import Choice, ChoiceQuestion, PredicateQuestion
from brain.prompts import load_prompt
from brain.vault.generated import build_core
from brain.vault.raw import Role, Transcript
from brain.vault.store import Vault

log = logging.getLogger(__name__)

MAX_FOLLOW_UPS = 2  # per plan step
MAX_DECLINES = 3  # in a row: the owner has had enough of this region
MAX_QUESTIONS = 30
FOLLOW_ABOVE = 0.5  # p(new thread) at or above which an answered step gets a follow-up
HISTORY_MAX_TURNS = 12  # question/answer exchanges kept in the prompt
HISTORY_TRIM = 4  # dropped at once, so the cached prefix survives between trims
TEMPERATURE = 0.4
MAX_MESSAGE_TOKENS = 300

START = "Let's do an interview."
PRIVATE_ACK = (
    "Understood: what you say about this stays private and is never shared with the big model. "
    "Go ahead."
)
PRIVATE_MARKER = (
    "[interview] privacy:: private · the owner's next answer is private: "
    "never share it with the big model"
)
SKIP_MARKER = "[interview] skip · the owner skipped this question"
CLOSING = {
    "end": "That's all I wanted to ask about {label} for now. Thanks, I'll file this away.",
    "time": "We've been at it for about {minutes}, so let's stop here. Thanks!",
    "stop": "Okay, stopping here. Thanks, I'll file this away.",
    "declined": "Let's leave {label} there for today. Thanks!",
    "limit": "That's plenty for one session. Thanks, I'll file this away.",
}
FOLLOW_UP_HINT = "one specific thing from their last answer that is worth knowing more about"
EMPTY_PLAN = "I have nothing to ask about {label} right now."

SKIP_WORDS = frozenset(["skip", "/skip", "pass", "next", "/next"])
STOP_WORDS = frozenset(["stop", "/stop", "quit", "/quit", "/q", "exit", "/exit"])
PRIVATE_PLACEHOLDER = "(a private answer: never mention, quote or ask about it again)"
PRIVATE_WORDS = frozenset(["private", "/private"])
_PRIVATE_INLINE = re.compile(r"^\s*/?private\s*[:,\-–—]\s*(.+)$", re.S | re.I)

# interview_answer v1: the questions asked about an answer (input: interview_answer.md)
ANSWER = ChoiceQuestion(
    name="answer",
    instructions="How does the owner's reply relate to the interviewer's question?",
    choices=[
        Choice(value="answered", description="it gives the information that was asked for"),
        Choice(value="partial", description="it gives only part of it, or stays vague"),
        Choice(
            value="declined",
            description="the owner does not know, would rather not say, or it does not apply",
        ),
        Choice(value="unclear", description="it is off-topic or cannot be understood"),
    ],
)
NEW_THREAD = PredicateQuestion(
    name="new_thread",
    instructions=(
        "Does the owner's reply bring up something specific that was not asked about (a "
        "person, a project, an event, a change or a feeling) that a friend would naturally "
        "ask about next?"
    ),
)

Event = Literal[
    "question",
    "follow_up",
    "clarify",
    "private",
    "skip",
    "end",
    "time",
    "stop",
    "declined",
    "limit",
    "empty",
]
ProcessSession = Callable[..., Any]


class InterviewMessage(BaseModel):
    reasoning: str = Field(description="what to ask and which known fact to start from")
    message: str = Field(description="the message to the owner: one question")


@dataclass(frozen=True)
class Reply:
    message: str
    done: bool = False
    event: Event = "question"


@dataclass
class CloseResult:
    transcript: str | None = None  # vault-relative path
    report: Any = None  # the memory pipeline's SessionReport
    error: str | None = None
    answered: list[str] = field(default_factory=list)  # questions.md items ticked
    questions_changed: bool = False


def incomplete(report: Any) -> bool:
    """The memory pipeline failed, or left an episode unextracted."""
    if report is None:
        return False
    if getattr(report, "status", "processed") == "failed" or getattr(report, "error", None):
        return True
    return any(getattr(e, "error", None) for e in getattr(report, "episodes", []))


def default_process() -> ProcessSession:
    from brain.memory import process_session  # built by the memory pipeline (Phase 2)

    return process_session


def _fold(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def fill(template: str, **values: str) -> str:
    """``{name}`` placeholders, without ``str.format`` (fact text may contain braces)."""
    for name, value in values.items():
        template = template.replace("{" + name + "}", value)
    return template


def parse_command(text: str) -> tuple[str | None, str]:
    """``("skip" | "stop" | "private" | None, answer text)``."""
    word = text.strip().casefold().rstrip(".!")
    if word in SKIP_WORDS:
        return "skip", ""
    if word in STOP_WORDS:
        return "stop", ""
    if word in PRIVATE_WORDS:
        return "private", ""
    if m := _PRIVATE_INLINE.match(text):
        return "private", m.group(1).strip()
    return None, text.strip()


# --- checks on generated messages --------------------------------------------------------------

_WORD = re.compile(r"[\w][\w'’\-]*")
_SENTENCE = re.compile(r"(?:^|(?<=[.?!:;—–])\s+|\n+)")
ALWAYS = frozenset(["i", "i'm", "i'd", "i'll", "i've", "ok", "okay"])


_DATE = re.compile(r"\b(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?\b")
MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]


def _norm(word: str) -> str:
    word = re.sub(r"(?:'s|’s)$", "", word.casefold())
    return re.sub(r"^(\d+)(?:st|nd|rd|th)$", r"\1", word)


def date_words(text: str) -> set[str]:
    """How the dates in ``text`` may be written: ``2026-11-15`` → 2026, november, nov, 15."""
    out: set[str] = set()
    for m in _DATE.finditer(text):
        out.add(m.group(1))
        if m.group(2) and 1 <= int(m.group(2)) <= 12:
            month = MONTHS[int(m.group(2)) - 1]
            out |= {month, month[:3], m.group(2), str(int(m.group(2)))}
        if m.group(3):
            out |= {m.group(3), str(int(m.group(3)))}
    return out


def ungrounded(message: str, allowed: str) -> list[str]:
    """Names (capitalised words not opening a sentence) and numbers in ``message`` that never
    appear in ``allowed``: the model may not bring in facts it was not given."""
    known = {_norm(w) for w in _WORD.findall(allowed)} | date_words(allowed) | ALWAYS
    out = []
    for sentence in (s for s in _SENTENCE.split(message) if s and s.strip()):
        words = _WORD.findall(sentence)
        for n, word in enumerate(words):
            name = word[:1].isupper() and n > 0
            if (name or any(c.isdigit() for c in word)) and _norm(word) not in known:
                out.append(word)
    return out


def one_question(message: str) -> str:
    """Keep the message up to its first question mark (one thing at a time)."""
    message = " ".join(message.split())
    if message.count("?") > 1:
        message = message[: message.index("?") + 1]
    return message


def fallback(item: PlanItem, mode: str) -> str:
    if mode == "confirm" and item.cite:
        return f"I have noted: {item.cite.rstrip('.')}. Is that still right?"
    if mode == "more" and item.cite:
        return f"So far I know: {item.cite.rstrip('.')}. {item.hint}"
    if mode == "follow_up":
        return "Could you tell me a bit more about that?"
    if mode == "clarify":
        return f"Sorry, I didn't quite follow. {item.hint}"
    return item.hint


def _accepts(fn: Callable[..., Any], name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


# --- the session -------------------------------------------------------------------------------


class InterviewSession:
    def __init__(
        self,
        vault: Vault,
        llm: Any,
        report: CoverageReport,
        region_id: str,
        plan: Sequence[PlanItem],
        settings: Settings,
        *,
        minutes: float = 12,
        max_questions: int = MAX_QUESTIONS,
        record: bool = True,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.vault = vault
        self.llm = llm
        self.report = report
        self.region_id = region_id
        region = report.regions[region_id]
        self.label = region.region.label or region_id
        self.private_region = region.private
        self.plan = list(plan)
        self.settings = settings
        self.minutes = minutes
        self.max_questions = max_questions
        self.record = record
        self.clock = clock
        if vault.core_path.exists():
            core = vault.core_path.read_text(encoding="utf-8")
        else:
            core = build_core(vault, max_tokens=settings.core_max_tokens, today=clock().date())
        self.core = core_for_prompt(core)
        self.system = load_prompt("interview_system") + "\n# Owner profile\n\n" + self.core + "\n"
        self.history: list[dict[str, str]] = [{"role": "user", "content": START}]
        self.transcript: Transcript | None = None
        self.started: datetime | None = None
        self.idx = 0
        self.mode: str = self.plan[0].mode if self.plan else "ask"
        self.follow_ups = 0
        self.declines = 0
        self.private = self.private_region  # a private region: every answer is private
        self.question = ""
        self.asked = 0
        self.answers = 0
        self.done = False
        self.resolved: list[str] = []
        self.skipped: list[str] = []
        self.private_turns: list[int] = []
        self.trace: list[tuple[str, str]] = []  # (step key, what happened), for tests and -v

    # -- state ------------------------------------------------------------------------------

    @property
    def item(self) -> PlanItem:
        return self.plan[min(self.idx, len(self.plan) - 1)]

    def elapsed_minutes(self) -> float:
        if self.started is None:
            return 0.0
        return (self.clock() - self.started).total_seconds() / 60

    # -- transcript -------------------------------------------------------------------------

    def _record(self, role: Role, text: str) -> int | None:
        if not self.record:
            return None
        if self.transcript is None:
            meta = {"model": self.settings.llm.chat_model, "region": self.region_id}
            self.transcript = Transcript.start(
                self.vault, channel="interview", now=self.clock(), meta=meta
            )
        return self.transcript.append(role, text, at=self.clock())

    def _marker(self, item: PlanItem, mode: str) -> str:
        parts = [f"region:: {item.region}", f"topic:: {item.topic}"]
        if item.slot:
            parts.append(f"slot:: {item.slot}")
        if item.entity:
            parts.append(f"about:: [[{item.entity}]]")
        parts.append(f"mode:: {mode}")
        return "[interview] " + " · ".join(parts)

    # -- prompt -----------------------------------------------------------------------------

    def plan_block(self, item: PlanItem, mode: str, today: date) -> str:
        known = "\n".join(f"- {line}" for line in item.known) or "- (nothing yet)"
        ctx = context_lines(self.report, item)
        context = "\n".join(f"- {line}" for line in ctx) or "- (none)"
        hint = FOLLOW_UP_HINT if mode == "follow_up" else item.hint
        return fill(
            load_prompt("interview_turn"),
            topic=item.topic,
            mode=mode,
            hint=hint,
            known=known,
            context=context,
            today=today.isoformat(),
        ).strip()

    def build_messages(self, block: str) -> list[dict[str, str]]:
        last = self.history[-1]["content"]
        return [
            {"role": "system", "content": self.system},
            *self.history[:-1],
            {"role": "user", "content": f"{last}\n\n{block}"},
        ]

    def _trim(self) -> None:
        if len(self.history) > 1 + 2 * HISTORY_MAX_TURNS:
            self.history = self.history[:1] + self.history[1 + 2 * HISTORY_TRIM :]

    def _allowed(self, block: str) -> str:
        return "\n".join([self.core, block, *(m["content"] for m in self.history)])

    def compose(self, item: PlanItem, mode: str) -> str:
        """The next message: generated, checked, or the fallback template."""
        block = self.plan_block(item, mode, self.clock().date())
        try:
            out = self.llm.structured(
                self.build_messages(block),
                InterviewMessage,
                temperature=TEMPERATURE,
                max_tokens=MAX_MESSAGE_TOKENS,
            )
            message = one_question(out.message)
        except Exception as err:  # a model hiccup must not end the interview
            log.warning("interview message failed (%s); using the template", err)
            self.trace.append((item.key, f"fallback: {type(err).__name__}"))
            return fallback(item, mode)
        if not message.strip():
            self.trace.append((item.key, "fallback: empty"))
            return fallback(item, mode)
        if _fold(message) in {
            _fold(m["content"]) for m in self.history if m["role"] == "assistant"
        }:
            self.trace.append((item.key, f"fallback: repeats {message!r}"))
            return fallback(item, mode)
        if bad := ungrounded(message, self._allowed(block)):
            log.info("ungrounded words %s in %r; using the template", bad, message)
            self.trace.append((item.key, f"fallback: ungrounded {', '.join(bad)} in {message!r}"))
            return fallback(item, mode)
        return message

    def _ask(self, event: Event) -> Reply:
        if self.asked >= self.max_questions:
            return self._finish("limit")
        item = self.item
        self._record("system", self._marker(item, self.mode))
        message = self.compose(item, self.mode)
        self.question = message
        self.history.append({"role": "assistant", "content": message})
        self._record("assistant", message)
        self.asked += 1
        self.trace.append((item.key, self.mode))
        return Reply(message, event=event)

    def _finish(self, event: Event) -> Reply:
        minutes = f"{self.minutes:g} minute" + ("" if self.minutes == 1 else "s")
        message = fill(CLOSING[event], label=self.label.casefold(), minutes=minutes)
        self.history.append({"role": "assistant", "content": message})
        self._record("assistant", message)
        self.done = True
        self.trace.append((self.item.key if self.plan else self.region_id, event))
        return Reply(message, done=True, event=event)

    def _advance(self, event: Event = "question") -> Reply:
        self.idx += 1
        self.follow_ups = 0
        self.private = self.private_region
        if self.idx >= len(self.plan):
            return self._finish("end")
        if self.elapsed_minutes() >= self.minutes:
            return self._finish("time")
        self.mode = self.item.mode
        return self._ask(event)

    # -- turns ------------------------------------------------------------------------------

    def opening(self) -> Reply:
        """The first question (or a note that there is nothing to ask)."""
        if not self.plan:
            self.done = True
            return Reply(fill(EMPTY_PLAN, label=self.label.casefold()), done=True, event="empty")
        self.started = self.clock()
        return self._ask("question")

    def classify(self, answer: str) -> tuple[str, float]:
        """``(answer label, p(new thread))``; a failed or refused call counts as answered."""
        text = fill(
            load_prompt("interview_answer"),
            topic=self.item.topic,
            question=self.question,
            answer=answer,
        ).strip()
        try:
            label_d, thread_d = self.llm.decide(text, [ANSWER, NEW_THREAD])
        except Exception as err:
            log.warning("answer classification failed (%s)", err)
            return "answered", 0.0
        label = label_d.choice if not label_d.refusal and label_d.choice else "answered"
        return label, thread_d.p_yes or 0.0

    def reply(self, text: str) -> Reply:
        """Handle the owner's message; return what the brain says next."""
        if self.done:
            return Reply("", done=True, event="end")
        if not text.strip():
            return Reply("", event="empty")
        command, answer = parse_command(text)
        if command == "stop":
            self._record("user", text.strip())
            return self._finish("stop")
        if command == "skip":
            self._record("user", text.strip())
            self._record("system", SKIP_MARKER)
            self.history.append({"role": "user", "content": "skip"})
            self.skipped.append(self.item.key)
            self.trace.append((self.item.key, "skipped"))
            return self._advance("skip")
        if command == "private":
            self.private = True
            if not answer:
                self._record("user", text.strip())
                self._record("system", PRIVATE_MARKER)
                self.history += [
                    {"role": "user", "content": text.strip()},
                    {"role": "assistant", "content": PRIVATE_ACK},
                ]
                self._record("assistant", PRIVATE_ACK)
                self.trace.append((self.item.key, "private"))
                return Reply(PRIVATE_ACK, event="private")
        return self._answer(answer)

    def _answer(self, answer: str) -> Reply:
        if self.private:
            self._record("system", PRIVATE_MARKER)
        turn = self._record("user", answer)
        if self.private and turn is not None:
            self.private_turns.append(turn)
        # Outside a private region, later questions must not quote a private answer (they would
        # land in a non-private episode): the interviewer sees a placeholder and moves on.
        hidden = self.private and not self.private_region
        self.history.append({"role": "user", "content": PRIVATE_PLACEHOLDER if hidden else answer})
        self._trim()
        self.answers += 1
        label, p_thread = self.classify(answer)
        item = self.item
        self.trace.append((item.key, f"{label} p_thread={p_thread:.2f}"))
        self.declines = self.declines + 1 if label == "declined" else 0
        answered = label in ("answered", "partial")
        if item.question_id and answered and item.question_id not in self.resolved:
            self.resolved.append(item.question_id)
        if self.declines >= MAX_DECLINES:
            return self._finish("declined")
        if self.elapsed_minutes() >= self.minutes:
            return self._finish("time")
        if self.follow_ups < MAX_FOLLOW_UPS and not hidden:
            if label == "partial" or (label == "answered" and p_thread >= FOLLOW_ABOVE):
                self.follow_ups += 1
                self.mode = "follow_up"
                return self._ask("follow_up")
            if label == "unclear" and self.follow_ups == 0:
                self.follow_ups += 1
                self.mode = "clarify"
                return self._ask("clarify")
        return self._advance()

    # -- close ------------------------------------------------------------------------------

    def close(
        self,
        process: ProcessSession | None = None,
        *,
        db: sqlite3.Connection | None = None,
        commit: bool = True,
    ) -> CloseResult:
        """Save the transcript, run the memory pipeline on it (``channel="interview"``), tick
        the questions.md items that were answered, and rebuild questions.md when ``db`` is
        given. A pipeline failure is reported, never raised: the transcript is saved and can
        be processed later."""
        result = CloseResult()
        if self.transcript is None:
            return result
        if commit:
            self.transcript.commit()
        result.transcript = self.transcript.rel
        if not self.answers:
            return result
        today = self.clock().date()
        try:
            process = process or default_process()
            kwargs: dict[str, Any] = {
                "channel": "interview",
                "regions_hint": [self.region_id],
                "today": today,
            }
            if self.private_turns and _accepts(process, "private_turns"):
                kwargs["private_turns"] = list(self.private_turns)
            if db is not None and _accepts(process, "db"):
                kwargs["db"] = db  # one connection: questions.md is rebuilt from the same index
            result.report = process(self.vault, self.llm, self.transcript, **kwargs)
        except Exception as err:
            log.warning("memory pipeline failed on %s: %s", result.transcript, err)
            result.error = f"{type(err).__name__}: {err}"
            return result
        if incomplete(result.report):
            # an episode is still pending: tick the items once their answers are in memory, by
            # asking them again rather than losing them
            return result
        for qid in self.resolved:
            if resolve_question(self.vault, qid):
                result.answered.append(qid)
        if db is not None:
            result.questions_changed = refresh_questions(self.vault, db, today=today, commit=False)
        if commit and (result.answered or result.questions_changed):
            from brain.vault import gitops

            gitops.commit_all(self.vault.root, "brain: update questions.md after interview")
        return result
