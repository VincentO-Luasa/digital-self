"""Save an interview as it goes: each finished block is turned into memory on one worker thread
while the owner reads and types (docs/architecture.md, Flow 2; docs/decisions.md "Interviews
save as they go").

- ``Gate``: the interview holds it busy while it handles an answer (classify, link, compose);
  the worker waits for it before **every** model call, so between two calls of a block (gate,
  segmentation, resolution, each extraction window, checks, corrections, settling) it yields
  to the owner. A call already running when the owner answers is not interrupted: oMLX runs it
  alongside the interview's calls and both slow down (a long prefill delays a short call by
  seconds). So a call starts only after the owner has been idle for a moment (``grace``: an
  owner who answers at once never meets the worker), and a long call (structured output) only
  early in a pause (``latest``), when it is likely to end before the answer comes.
- ``Yielding``: the model client as the worker sees it (each call waits for the gate). A long
  call is streamed with a ``cancel`` event that the interview sets when it gets busy: the worker
  stops generating as soon as the owner answers (the server stops when the stream closes) and
  repeats the call in the next pause (at most ``MAX_INTERRUPTS`` times, then it runs to its
  end). A prefill under way still finishes first.
- ``BlockSaver``: one worker thread, jobs in order. A job settles the block's lint doubts, then
  runs the memory pipeline on the turns written so far (``process_session(..., turns=…,
  refresh=False)``), which segments only what no earlier part covered. The worker is the only
  vault writer while the interview runs (the interview itself only appends to its transcript);
  ``drain()`` waits for it before close writes anything.

Errors never reach the owner: a failed job is logged and kept in ``errors``; what it did not
save is picked up by the run at close (or ``brain memory process --pending``).
"""

from __future__ import annotations

import inspect
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from brain.llm.base import CallCancelled

log = logging.getLogger(__name__)

MODEL_CALLS = frozenset(["structured", "decide", "chat", "embed", "rerank"])
# a long call is stopped at most this often for the owner, then runs to its end. Measured: no
# cap, one extraction window longer than the owner's pauses was restarted 11 times (one block
# saved); a cap of 2 brought the collisions back (+1.5-3 s on 10-12 of 26 turns). A guard
# against endless restarts, not a tuning knob (docs/decisions.md "Interviews save as they go").
MAX_INTERRUPTS = 5


def _accepts_cancel(fn: Callable[..., Any]) -> bool:
    try:
        return "cancel" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


class Gate:
    """Open while the interview is idle (the owner reads or types), closed while it works.

    ``wait(grace, latest)``: return once the interview has been idle for ``grace`` seconds; with
    ``latest``, only while it has been idle for at most that long (a long call started late in
    a pause would still run when the owner answers: it waits for the next pause instead).
    ``release()`` (close) opens it for good."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._busy = False
        self._since = time.monotonic()
        self._released = False
        self._interrupt = threading.Event()  # set when the interview gets busy

    def interrupt_event(self) -> threading.Event:
        """A fresh event that is set as soon as the interview gets busy (already set if it is)."""
        with self._cond:
            self._interrupt = threading.Event()
            if self._busy:
                self._interrupt.set()
            return self._interrupt

    @contextmanager
    def busy(self) -> Iterator[None]:
        with self._cond:
            self._busy = True
            self._interrupt.set()
        try:
            yield
        finally:
            with self._cond:
                self._busy = False
                self._since = time.monotonic()
                self._cond.notify_all()

    def release(self) -> None:
        with self._cond:
            self._released = True
            self._cond.notify_all()

    def wait(self, grace: float = 0.0, latest: float | None = None) -> float:
        """Block until the worker may start a call; the seconds waited."""
        start = time.monotonic()
        with self._cond:
            while not self._released:
                if self._busy:
                    self._cond.wait()
                    continue
                idle = time.monotonic() - self._since
                if idle < grace:
                    self._cond.wait(grace - idle)
                elif latest is not None and idle > latest:
                    self._cond.wait()  # until the next pause begins
                else:
                    break
        return time.monotonic() - start


class Cancelled(Exception):
    """The worker was told to stop before its job made a model call (close does the job)."""


class Yielding:
    """A model client whose calls wait for the gate first (attributes pass through): every call
    after ``grace`` seconds of idleness, long ones (structured output, chat) only up to
    ``latest`` seconds into a pause."""

    LONG = frozenset(["structured", "chat"])

    def __init__(
        self, llm: Any, gate: Gate, *, grace: float = 0.0, latest: float | None = None
    ) -> None:
        self._llm = llm
        self._gate = gate
        self.grace = grace
        self.latest = latest
        self.calls = 0
        self.waited = 0.0  # seconds spent yielding to the interview
        self.seconds = 0.0  # seconds in model calls
        self.active: str | None = None  # the call running now (measurements)
        self.job_calls = 0  # calls made by the current job (reset by the worker)
        self.cancel_unstarted = False  # close: a job that has made no call yet gives up
        self.interrupted = 0  # long calls stopped because the owner answered

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._llm, name)
        if name not in MODEL_CALLS or not callable(attr):
            return attr
        latest = self.latest if name in self.LONG else None
        cancellable = name in self.LONG and _accepts_cancel(attr)

        def call(*args: Any, **kwargs: Any) -> Any:
            tries = 0
            while True:
                self.waited += self._gate.wait(self.grace, latest)
                if self.cancel_unstarted and not self.job_calls:
                    raise Cancelled
                self.calls += 1
                self.job_calls += 1
                self.active = name
                start = time.perf_counter()
                try:
                    if cancellable:  # after MAX_INTERRUPTS, let it finish (no starving)
                        kwargs["cancel"] = (
                            self._gate.interrupt_event() if tries < MAX_INTERRUPTS else None
                        )
                    return attr(*args, **kwargs)
                except CallCancelled:
                    tries += 1
                    self.interrupted += 1  # the owner answered: again in the next pause
                finally:
                    self.active = None
                    self.seconds += time.perf_counter() - start

        return call


@dataclass
class Job:
    """One finished block: what to settle and the turns written so far."""

    turns: list[Any]  # brain.vault.raw.Turn, read by the interview thread
    private_turns: list[int]
    settle: list[Any] = field(default_factory=list)  # session-built settle inputs
    label: str = ""
    now: datetime | None = None  # the interview's clock at submission (the worker never reads it)


@dataclass
class JobResult:
    job: Job
    report: Any = None  # the pipeline's SessionReport
    settled: list[Any] = field(default_factory=list)  # Settlement
    settled_ids: list[str] = field(default_factory=list)  # questions.md ids
    error: str | None = None
    seconds: float = 0.0
    cancelled: bool = False  # stopped at close before any model call: close does it


class BlockSaver:
    """Runs ``work(job) -> JobResult`` for each submitted block, in order, on one thread
    (``background=False``: at once, on the caller's thread; tests and measurements)."""

    def __init__(self, work: Callable[[Job], JobResult], *, background: bool = True) -> None:
        self.work = work
        self.background = background
        self.results: list[JobResult] = []
        self.errors: list[str] = []
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()  # results / errors
        self._stop = False
        self.dropped: list[Job] = []  # jobs never started (drain with cancel)

    def submit(self, job: Job) -> None:
        if self._stop:
            return
        if not self.background:
            self._run(job)
            return
        if self._thread is None:
            # daemon: an interview killed mid-block leaves a resumable vault, never a hung exit
            self._thread = threading.Thread(target=self._loop, name="interview-saver", daemon=True)
            self._thread.start()
        self._queue.put(job)

    def _loop(self) -> None:
        while (job := self._queue.get()) is not None:
            if self._stop:
                self.dropped.append(job)
            else:
                self._run(job)

    def _run(self, job: Job) -> None:
        start = time.perf_counter()
        try:
            result = self.work(job)
        except Exception as err:  # noqa: BLE001 - the close (or --pending) catches up
            log.warning("saving %s failed: %s", job.label or "a block", err)
            result = JobResult(job, error=f"{type(err).__name__}: {err}")
        result.seconds = time.perf_counter() - start
        with self._lock:
            self.results.append(result)
            if result.error:
                self.errors.append(result.error)

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def drain(self, *, cancel: bool = False) -> list[Job]:
        """Wait for the worker to finish; ``cancel``: only the job it is running. The jobs it
        never started (the caller does their work itself)."""
        if cancel:
            self._stop = True
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
            self._thread = None
        return list(self.dropped)
