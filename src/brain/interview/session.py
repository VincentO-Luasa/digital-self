"""One interview: ask one thing at a time, follow up, and hand the transcript to the memory
pipeline at the end (docs/architecture.md, Flow 2).

Per owner answer, a few model calls:

1. ``LLM.decide`` classifies the answer (``ANSWER``: answered / partial / declined / unclear /
   asked back) and asks whether it opened something worth a follow-up (``NEW_THREAD``). Code then
   decides: follow up (partial, or a new thread, at most ``MAX_FOLLOW_UPS`` per step), clarify
   once (unclear), answer a question back once with what the vault has (``asked``: the item's
   one-line descriptions and sources, then the question again), or move on. ``MAX_DECLINES``
   declines in a row, the end of the plan or ``max_questions`` end the session.
2. **Answer linking**: before the next planned step (and a few after it that share its entity or
   words) one ``LLM.decide`` call asks, per step, whether the session so far already answers it,
   or the owner declined it (or the same kind of thing) earlier. Answered or declined steps are
   skipped (answered ones count as answered); a generated follow-up the decision finds answered
   or declined is dropped and the interview moves on. Follow-ups see the relevant earlier
   exchanges and what the owner declined (``already said`` / ``declined`` in the PLAN block).
3. A structured call writes the next message (``interview_system`` + core card as the cached
   prefix, then the conversation, then a PLAN block on the latest message). Code checks it is
   grounded (every name and number in it appears in what the brain knows or was told) and asks
   one thing; otherwise a template built from the hint is used.

Time is soft: at ``minutes`` the current entity block is finished, then the brain asks whether
to go on (a yes adds half the time; anything else stops). Lint doubts get at most
``DOUBT_SHARE`` of the time. At close, answered lint doubts are settled
(``brain.interview.settle``) before the memory pipeline runs.

Owner commands: ``skip`` (next step), ``private`` or ``private: <answer>`` (the answers of this
step are private: never shared with the big model, never quoted in later questions, never used to
settle a doubt), ``stop``.

The transcript lands in ``raw/conversations/`` with ``channel: interview``. System turns carry
machine-readable markers for the memory pipeline: ``[interview] region:: … · slot:: …`` before
each question and ``[interview] privacy:: private …`` before each private answer.
"""

from __future__ import annotations

import inspect
import logging
import re
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from brain.chat.session import core_for_prompt
from brain.config import Settings
from brain.interview.coverage import CoverageReport
from brain.interview.doubts import Doubt, EntityInfo
from brain.interview.planner import PlanItem, context_lines
from brain.interview.questions import refresh_questions, resolve_question
from brain.interview.settle import Settlement, Settler
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
LINK_AHEAD = 1  # planned steps checked per linking decision (each costs ~0.5 s uncached)
LINK_LOOK = 8  # how far ahead the others are looked for
LINK_P = 0.6  # p(answered) at or above which a step is skipped
REFUSED_P = 0.35  # p(asks again for what the owner refused to tell) to skip it
REFUSAL_P = 0.9  # p(refused) for an answer to count as a refusal ("nothing more for now": 0.7)
SHOW_EXCHANGES = 6  # earlier exchanges shown to a follow-up or used to settle a doubt
LINK_WINDOW = 10  # exchanges shown to a linking decision (dropped LINK_TRIM at a time, so the
LINK_TRIM = 4  # text grows by appending and its prefix stays in the server's cache)
ANSWER_WORDS = 60  # an owner answer is clipped to this in those
SAID_LINES = 3
DECLINED_LINES = 3
DOUBT_SHARE = 0.4  # lint doubts are asked only in the first part of the session
EXTEND = 0.5  # "keep going" adds minutes × EXTEND

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
CONTINUE = "We've covered {topics}. Keep going, or stop here?"
CONTEXT = "Here's what I have: {lines}. {question}"
NO_CONTEXT = "I don't have anything on that yet, that's why I'm asking. {question}"
FOLLOW_UP_HINT = "one specific thing from their last answer that is worth knowing more about"
EMPTY_PLAN = "I have nothing to ask about {label} right now."

SKIP_WORDS = frozenset(["skip", "/skip", "pass", "next", "/next"])
STOP_WORDS = frozenset(["stop", "/stop", "quit", "/quit", "/q", "exit", "/exit"])
PRIVATE_PLACEHOLDER = "(a private answer: never mention, quote or ask about it again)"
PRIVATE_WORDS = frozenset(["private", "/private"])
_PRIVATE_INLINE = re.compile(r"^\s*/?private\s*[:,\-–—]\s*(.+)$", re.S | re.I)

# interview_answer v2: the questions asked about an answer (input: interview_answer.md)
ANSWER = ChoiceQuestion(
    name="answer",
    instructions="How does the owner's reply relate to the interviewer's question?",
    choices=[
        Choice(value="answered", description="it gives the information that was asked for"),
        Choice(value="partial", description="it gives only part of it, or stays vague"),
        Choice(
            value="declined",
            description="the owner does not know, has nothing to add, or it does not apply",
        ),
        Choice(
            value="refused",
            description="the owner would rather not say: it is private, confidential or they "
            "keep it to themselves",
        ),
        Choice(value="unclear", description="it is off-topic or cannot be understood"),
        Choice(
            value="asked",
            description="instead of answering, the owner asks the interviewer something back or "
            "asks for more context",
        ),
    ],
)
# interview_link v1: per candidate step (input: interview_link.md); "open" first, as the safe
# default (a listed answer draws the eye). Refusals are a predicate of their own, asked only when
# the owner refused something: as a third option, "declined" took 0.7 for plain questions after
# answers like "nothing more on that" (docs/decisions.md).
LINK_CHOICES = [
    Choice(value="open", description="no: nothing said so far gives this; asking is new"),
    Choice(
        value="answered",
        description="yes: the owner already gave this information; asking would make them "
        "repeat themselves",
    ),
]


def link_question(n: int, question: str) -> ChoiceQuestion:
    return ChoiceQuestion(
        name=f"q{n}",
        instructions=f"The interviewer may ask next: “{question}” Is that already answered?",
        choices=LINK_CHOICES,
    )


def refused_question(n: int, question: str) -> PredicateQuestion:
    return PredicateQuestion(
        name=f"r{n}",
        instructions=f"Would “{question}” ask the owner again for something they refused to "
        "tell, or for the same kind of information?",
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
    "context",
    "continue",
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
class Exchange:
    """One question and the owner's answer to it, as the interviewer saw them."""

    step: int  # plan index
    question: str
    answer: str  # the placeholder for a hidden private answer
    private: bool
    hidden: bool
    label: str = "answered"


@dataclass
class CloseResult:
    transcript: str | None = None  # vault-relative path
    report: Any = None  # the memory pipeline's SessionReport
    error: str | None = None
    answered: list[str] = field(default_factory=list)  # questions.md items ticked
    questions_changed: bool = False
    settled: list[Settlement] = field(default_factory=list)  # lint doubts mapped from answers


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


_CLAIM = re.compile(
    r"\byou(?:'ve| have)? (?:mentioned|said|told me|listed|wrote|noted|described|called)\b"
    r"(?P<rest>[^.?!:;]*)",
    re.I,
)


def invented_claim(message: str, owner_said: str) -> str | None:
    """A claim that the owner said or listed something ("You listed Stream processing as a
    project") that their answers in this conversation do not back: most of the claimed words
    must appear in what they said. Ingest made the page, not the owner."""
    said = content_words(owner_said)
    for m in _CLAIM.finditer(message):
        claimed = content_words(m.group("rest"))
        if not claimed or len(claimed & said) * 2 < len(claimed):
            return m.group(0).strip()
    return None


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


STOP_LIST = frozenset(
    [
        "the",
        "a",
        "an",
        "of",
        "to",
        "and",
        "or",
        "in",
        "on",
        "at",
        "for",
        "by",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "my",
        "your",
        "our",
        "their",
        "its",
        "that",
        "this",
        "what",
        "which",
        "who",
        "how",
        "when",
        "where",
        "why",
        "you",
        "i",
        "it",
        "do",
        "did",
        "does",
        "have",
        "has",
        "had",
        "not",
        "just",
        "about",
        "there",
        "here",
        "them",
        "they",
        "from",
        "into",
        "than",
        "then",
        "also",
        "more",
        "some",
        "any",
    ]
)
_YES = re.compile(
    r"^\W*(?:y|yes|yeah|yep|yup|sure|ok|okay|alright|go on|go ahead|keep going|continue|"
    r"carry on|let'?s (?:go|continue|keep going))\b",
    re.I,
)
_TO_ASK = re.compile(r"^\s*(?:[-*]\s*)?To ask:")
_NO = re.compile(r"^\W*(?:no|nope|nah|not)\b", re.I)


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.casefold()) if len(w) > 3 and w not in STOP_LIST}


def wants_more(text: str) -> bool:
    """A clear yes to "keep going?" (anything else stops)."""
    return bool(_YES.match(text)) and not _NO.match(text)


def clip(text: str, words: int = ANSWER_WORDS) -> str:
    parts = text.split()
    return " ".join(parts[:words]) + (" …" if len(parts) > words else "")


def _and(parts: Sequence[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _block(item: PlanItem) -> str | None:
    return item.block or item.entity


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
        infos: Mapping[str, EntityInfo] | None = None,
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
        self.deadline = minutes  # soft: "keep going?" moves it
        self.max_questions = max_questions
        self.record = record
        self.clock = clock
        self.infos = dict(infos or {})
        if vault.core_path.exists():
            core = vault.core_path.read_text(encoding="utf-8")
        else:
            core = build_core(vault, max_tokens=settings.core_max_tokens, today=clock().date())
        # the card's "To ask" lines are this plan's questions: as profile text they pulled
        # follow-ups towards them
        self.core = "\n".join(
            line for line in core_for_prompt(core).split("\n") if not _TO_ASK.match(line)
        )
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
        self.context_given = False  # answered a question back in this step
        self.awaiting_continue = False
        self.resolved: list[str] = []
        self.skipped: list[str] = []
        self.private_turns: list[int] = []
        self.exchanges: list[Exchange] = []
        self.evidence: dict[int, list[int]] = {}  # plan index → exchanges that answer it
        self.link_cache: dict[int, str] = {}  # plan index → open / answered / declined
        self.asked_steps: list[int] = []
        self.timings: list[tuple[str, float]] = []  # (classify | link | compose, seconds)
        self.trace: list[tuple[str, str]] = []  # (step key, what happened), for tests and -v

    # -- state ------------------------------------------------------------------------------

    @property
    def item(self) -> PlanItem:
        return self.plan[min(self.idx, len(self.plan) - 1)]

    def elapsed_minutes(self) -> float:
        if self.started is None:
            return 0.0
        return (self.clock() - self.started).total_seconds() / 60

    def over_time(self) -> bool:
        return self.elapsed_minutes() >= self.deadline

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

    def _say(self, message: str) -> None:
        self.history.append({"role": "assistant", "content": message})
        self._record("assistant", message)

    # -- what was said ----------------------------------------------------------------------

    def _visible(self) -> list[tuple[int, Exchange]]:
        return [(n, e) for n, e in enumerate(self.exchanges) if not e.hidden]

    def relevant(self, texts: Sequence[str], *, skip_last: bool = False) -> list[int]:
        """Earlier exchanges that matter for ``texts`` (sharing a content word with them), and
        the last two; at most ``SHOW_EXCHANGES``, the most words in common first, returned
        oldest first. Hidden private answers never."""
        visible = self._visible()
        if skip_last and visible and visible[-1][0] == len(self.exchanges) - 1:
            visible = visible[:-1]
        want = set().union(*(content_words(t) for t in texts)) if texts else set()
        recent = [n for n, _ in visible[-2:]]
        scored = [
            (len(want & content_words(f"{e.question} {e.answer}")), n)
            for n, e in visible
            if n not in recent
        ]
        best = [n for score, n in sorted(scored, reverse=True) if score][: SHOW_EXCHANGES - 2]
        return sorted([*best, *recent])

    def naming(self, item: PlanItem) -> list[int]:
        """Exchanges that name the step's entities (by title), else those sharing its words."""
        titles = [self.infos[e].title for e in self._entities(item) if e in self.infos]
        if titles:
            folded = [_fold(t) for t in titles]
            return [
                n
                for n, e in self._visible()
                if any(f" {t} " in f" {_fold(e.question + ' ' + e.answer)} " for t in folded)
            ][-SHOW_EXCHANGES:]
        return self.relevant([item.hint])

    @staticmethod
    def _entities(item: PlanItem) -> list[str]:
        ents = [e for k in item.doubts if (d := Doubt.parse(k)) for e in d.entities]
        return list(dict.fromkeys([*ents, *([item.entity] if item.entity else [])]))

    def _render(self, indexes: Sequence[int]) -> str:
        return "\n".join(
            f"Interviewer: {self.exchanges[n].question}\nOwner: {clip(self.exchanges[n].answer)}"
            for n in indexes
        )

    def declined_lines(self) -> list[str]:
        """What the owner would not tell (refusals, not "nothing to add")."""
        declined = [e for _, e in self._visible() if e.label == "refused"]
        return [
            f"- asked “{clip(e.question, 25)}”, the owner said: “{clip(e.answer, 25)}”"
            for e in declined[-DECLINED_LINES:]
        ]

    def said_lines(self, item: PlanItem) -> list[str]:
        last = self.exchanges[-1].answer if self.exchanges else ""
        indexes = self.relevant([item.hint, last], skip_last=True)
        lines = [
            f"- {clip(self.exchanges[n].answer, 30)}"
            for n in indexes
            if self.exchanges[n].label in ("answered", "partial")
        ]
        return lines[-SAID_LINES:]

    # -- answer linking ---------------------------------------------------------------------

    def link(self, questions: Sequence[str], pinned: Sequence[int] = ()) -> list[str]:
        """Per question: ``open``, ``answered`` or ``declined`` given the session so far (one
        typed decision; ``open`` without a session, an answer or a measured p ≥ LINK_P)."""
        visible = [n for n, _ in self._visible()]
        if not visible:
            return ["open"] * len(questions)
        first = 0  # the window moves in blocks: the text grows by appending between trims
        if len(visible) > LINK_WINDOW:
            first = -(-(len(visible) - LINK_WINDOW) // LINK_TRIM) * LINK_TRIM
        window = visible[first:]
        # older exchanges about the same things go after the window (the cached prefix stays)
        older = [n for n in (pinned or self.relevant(questions)) if n < window[0]]
        declined = self.declined_lines()
        exchanges = self._render(window)
        if older:
            exchanges += "\n\nEarlier in the interview:\n" + self._render(older[-SHOW_EXCHANGES:])
        text = fill(
            load_prompt("interview_link"),
            exchanges=exchanges,
            declined="\n".join(declined) or "- (nothing)",
        ).strip()
        asks: list[ChoiceQuestion | PredicateQuestion] = [
            link_question(n, q) for n, q in enumerate(questions)
        ]
        # "asks again for what was refused?" only after a refusal, and only for questions sharing a
        # word with a refused one: over a long session the predicate drifted up for unrelated ones
        refused = set().union(
            *(content_words(e.question) for _, e in self._visible() if e.label == "refused")
        )
        asks += [
            refused_question(n, q) for n, q in enumerate(questions) if refused & content_words(q)
        ]
        start = time.perf_counter()
        try:
            decisions = self.llm.decide(text, asks)
        except Exception as err:  # a model hiccup only means asking anyway
            log.warning("answer linking failed (%s)", err)
            return ["open"] * len(questions)
        finally:
            self.timings.append(("link", time.perf_counter() - start))
        by_name = {d.name: d for d in decisions}
        out = []
        for n in range(len(questions)):
            answered, refused = by_name.get(f"q{n}"), by_name.get(f"r{n}")
            if refused is not None and (refused.p_yes or 0.0) >= REFUSED_P:
                out.append("declined")
            elif answered is not None and not answered.refusal and answered.choice == "answered":
                p = answered.probabilities.get("answered", 0.0)
                out.append("answered" if p >= LINK_P else "open")
            else:
                out.append("open")
        return out

    def _ahead(self, idx: int) -> list[int]:
        """Later steps worth checking with step ``idx``: same entity or shared words first."""
        here = self.plan[idx]
        last = self.exchanges[-1].answer if self.exchanges else ""
        want = content_words(f"{here.hint} {last}")
        later = [
            i
            for i in range(idx + 1, min(len(self.plan), idx + 1 + LINK_LOOK))
            if self.plan[i].mode != "discover" and i not in self.link_cache
        ]

        def score(i: int) -> tuple[int, int]:
            item = self.plan[i]
            same = bool(item.entity) and item.entity == here.entity
            return (-(2 * same + len(want & content_words(item.hint))), i)

        return sorted(later, key=score)[: LINK_AHEAD - 1]

    def status(self, idx: int) -> str:
        if idx in self.link_cache:
            return self.link_cache[idx]
        if self.plan[idx].mode == "discover" or not self._visible():
            self.link_cache[idx] = "open"
            return "open"
        steps = [idx, *self._ahead(idx)]
        pinned = sorted({n for i in steps for n in self.naming(self.plan[i])})
        statuses = self.link([self.plan[i].hint for i in steps], pinned)
        for i, st in zip(steps, statuses, strict=True):
            self.link_cache[i] = st
        return self.link_cache[idx]

    def _skip_linked(self, idx: int, status: str) -> None:
        item = self.plan[idx]
        self.trace.append((item.key, f"skipped: {status} earlier"))
        if status != "answered":
            return
        self.evidence[idx] = self.naming(item)
        for qid, doubt in self._qids(item):
            if not doubt and qid not in self.resolved:
                self.resolved.append(qid)

    @staticmethod
    def _qids(item: PlanItem) -> list[tuple[str, str | None]]:
        """questions.md ids with the doubt each settles (None: ticked when answered)."""
        doubts = list(item.doubts) + [None] * (len(item.question_ids) - len(item.doubts))
        out: list[tuple[str, str | None]] = []
        for qid, key in zip(item.question_ids, doubts, strict=False):
            d = Doubt.parse(key)
            out.append((qid, key if d and d.settles else None))
        return out

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
            about="\n".join(f"- {b}" for b in item.brief) or "- (nothing)",
            known=known,
            context=context,
            said="\n".join(self.said_lines(item)) or "- (nothing)",
            declined="\n".join(self.declined_lines()) or "- (nothing)",
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
        start = time.perf_counter()
        try:
            out = self.llm.structured(
                self.build_messages(block),
                InterviewMessage,
                temperature=TEMPERATURE,
                max_tokens=MAX_MESSAGE_TOKENS,
            )
            message = one_question(out.message)
            self.timings.append(("compose", time.perf_counter() - start))
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
        owner_said = " ".join(e.answer for _, e in self._visible())
        if claim := invented_claim(message, owner_said):
            log.info("invented provenance %r in %r; using the template", claim, message)
            self.trace.append((item.key, f"fallback: provenance {claim!r} in {message!r}"))
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
        message = self.compose(item, self.mode)
        if self.mode == "follow_up" and message != fallback(item, "follow_up"):
            status = self.link([message])[0]
            if status != "open":  # the owner already told it, or would not
                self.trace.append((item.key, f"follow-up dropped: {status} earlier: {message!r}"))
                return self._advance()
        self._record("system", self._marker(item, self.mode))
        self.question = message
        self._say(message)
        self.asked += 1
        if not self.asked_steps or self.asked_steps[-1] != self.idx:
            self.asked_steps.append(self.idx)
        self.trace.append((item.key, self.mode))
        return Reply(message, event=event)

    def _finish(self, event: Event) -> Reply:
        minutes = f"{self.minutes:g} minute" + ("" if self.minutes == 1 else "s")
        message = fill(CLOSING[event], label=self.label.casefold(), minutes=minutes)
        self._say(message)
        self.done = True
        self.trace.append((self.item.key if self.plan else self.region_id, event))
        return Reply(message, done=True, event=event)

    def _advance(self, event: Event = "question") -> Reply:
        previous = self.item if self.plan else None
        self.idx += 1
        self.follow_ups = 0
        self.context_given = False
        self.private = self.private_region
        return self._proceed(event, previous)

    def _proceed(self, event: Event, previous: PlanItem | None) -> Reply:
        """Ask the next step that is still open; at the soft time limit, between blocks, ask
        whether to go on."""
        while True:
            if self.idx >= len(self.plan):
                return self._finish("end")
            item = self.item
            block = _block(item)
            same_block = previous is not None and block is not None and _block(previous) == block
            if self.over_time() and not same_block:
                return self._offer_continue()
            if (
                item.is_doubt
                and not same_block
                and self.elapsed_minutes() >= DOUBT_SHARE * self.minutes
            ):
                self.trace.append((item.key, "skipped: doubt budget"))
                self.idx += 1
                continue
            status = self.status(self.idx)
            if status == "open":
                break
            self._skip_linked(self.idx, status)
            self.idx += 1
        self.mode = item.mode
        return self._ask(event)

    def covered(self) -> list[str]:
        out: list[str] = []
        for idx in self.asked_steps:
            item = self.plan[idx]
            info = self.infos.get(item.entity or "")
            if info:
                name = info.title
            elif ": " in item.topic:
                name = item.topic.rpartition(": ")[2]  # "Projects: Atlas Migration"
            else:
                name = item.topic.casefold()
            if name and name not in out:
                out.append(name)
        return out

    def _offer_continue(self) -> Reply:
        topics = self.covered()[-4:] or [self.label]
        message = fill(CONTINUE, topics=_and(topics))
        self._say(message)
        self.awaiting_continue = True
        self.trace.append((self.item.key, "continue?"))
        return Reply(message, event="continue")

    def _continue(self, text: str) -> Reply:
        self.awaiting_continue = False
        self._record("user", text.strip())
        self.history.append({"role": "user", "content": text.strip()})
        command, _ = parse_command(text)
        if command in ("stop", "skip") or _NO.match(text):
            return self._finish("stop")
        if not wants_more(text):
            return self._finish("time")  # anything unclear stops
        self.deadline = self.elapsed_minutes() + self.minutes * EXTEND
        self.trace.append((self.item.key, "keep going"))
        return self._proceed("question", None)

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
        start = time.perf_counter()
        try:
            label_d, thread_d = self.llm.decide(text, [ANSWER, NEW_THREAD])
            self.timings.append(("classify", time.perf_counter() - start))
        except Exception as err:
            log.warning("answer classification failed (%s)", err)
            return "answered", 0.0
        label = label_d.choice if not label_d.refusal and label_d.choice else "answered"
        if label == "refused" and label_d.probabilities.get(label, 1.0) < REFUSAL_P:
            label = "declined"  # not a clear refusal: nothing to remember as off-limits
        return label, thread_d.p_yes or 0.0

    def reply(self, text: str) -> Reply:
        """Handle the owner's message; return what the brain says next."""
        if self.done:
            return Reply("", done=True, event="end")
        if not text.strip():
            return Reply("", event="empty")
        if self.awaiting_continue:
            return self._continue(text)
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
                self.history.append({"role": "user", "content": text.strip()})
                self._say(PRIVATE_ACK)
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
        shown = PRIVATE_PLACEHOLDER if hidden else answer
        self.history.append({"role": "user", "content": shown})
        self._trim()
        self.answers += 1
        label, p_thread = self.classify(answer)
        item = self.item
        self.trace.append((item.key, f"{label} p_thread={p_thread:.2f}"))
        self.exchanges.append(Exchange(self.idx, self.question, shown, self.private, hidden, label))
        self.evidence.setdefault(self.idx, []).append(len(self.exchanges) - 1)
        self.link_cache.clear()  # a new answer may settle later steps
        if label == "asked":
            if not self.context_given and not hidden:
                self.context_given = True
                return self._context(item)
            return self._advance()  # asked back twice: move on
        self.declines = self.declines + 1 if label in ("declined", "refused") else 0
        if label in ("answered", "partial"):
            for qid, doubt in self._qids(item):
                if not doubt and qid not in self.resolved:
                    self.resolved.append(qid)
        if self.declines >= MAX_DECLINES:
            return self._finish("declined")
        if self.follow_ups < MAX_FOLLOW_UPS and not hidden and not self.over_time():
            if label == "partial" or (label == "answered" and p_thread >= FOLLOW_ABOVE):
                self.follow_ups += 1
                self.mode = "follow_up"
                return self._ask("follow_up")
            if label == "unclear" and self.follow_ups == 0:
                self.follow_ups += 1
                self.mode = "clarify"
                return self._ask("clarify")
        return self._advance()

    def _context(self, item: PlanItem) -> Reply:
        """The owner asked back: what the vault has on the step (its entities' one-liners and
        sources, else the facts it starts from), then the question again. No model call."""
        lines = [b.rstrip(".") for b in item.brief] or [k.rstrip(".") for k in item.known]
        if lines:
            message = fill(CONTEXT, lines="; ".join(lines), question=self.question)
        else:
            message = fill(NO_CONTEXT, question=self.question)
        self._say(message)
        self.trace.append((item.key, "context"))
        return Reply(message, event="context")

    # -- close ------------------------------------------------------------------------------

    def settle(self, result: CloseResult) -> list[str]:
        """Settle the lint doubts the owner answered (``brain.interview.settle``); the
        questions.md ids that are settled."""
        assert self.transcript is not None
        settler = Settler(
            self.vault,
            self.llm,
            transcript=self.transcript.rel,
            today=self.clock().date(),
            now=self.clock(),
            infos=self.infos,
        )
        ids: list[str] = []
        for idx, item in enumerate(self.plan):
            pairs = [(q, d) for q, d in self._qids(item) if d]
            if not pairs:
                continue
            indexes = self.evidence.get(idx, [])  # its own exchanges, or those that answered it
            exchanges = [
                (self.exchanges[n].question, self.exchanges[n].answer)
                for n in indexes
                if not self.exchanges[n].private
            ]
            if not exchanges:
                continue
            outcomes = settler.settle_step([d for _, d in pairs], exchanges, item.brief)
            result.settled += outcomes
            done = {o.doubt for o in outcomes if o.settled}
            ids += [q for q, d in pairs if d in done]
        return ids

    def close(
        self,
        process: ProcessSession | None = None,
        *,
        db: sqlite3.Connection | None = None,
        commit: bool = True,
    ) -> CloseResult:
        """Save the transcript, settle the lint doubts the owner answered, run the memory
        pipeline on it (``channel="interview"``), tick the questions.md items that were answered,
        and rebuild questions.md when ``db`` is given. A pipeline failure is reported, never
        raised: the transcript is saved and can be processed later."""
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
            settled = self.settle(result)
        except Exception as err:  # settling is a bonus: never lose the interview over it
            log.warning("settling lint doubts failed: %s", err)
            settled = []
        for qid in settled:  # the graph changed (or the owner confirmed it): done
            if resolve_question(self.vault, qid):
                result.answered.append(qid)
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
        if result.error is None and not incomplete(result.report):
            # an episode still pending: tick the other items once their answers are in memory,
            # by asking them again rather than losing them
            for qid in self.resolved:
                if resolve_question(self.vault, qid):
                    result.answered.append(qid)
            if db is not None:
                result.questions_changed = refresh_questions(
                    self.vault, db, today=today, commit=False
                )
        if commit and (result.answered or result.questions_changed):
            from brain.vault import gitops

            if gitops.is_dirty(self.vault.root):
                gitops.commit_all(self.vault.root, "brain: update questions.md after interview")
        return result
