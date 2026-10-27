"""``process_session``: a finished transcript → episodes → facts, journal lines and intentions
(docs/architecture.md, Flow 3: capture → segment → resolve → extract → apply / queue).

1. **Segment** once per session (``segment``): the episodes are written ``pending`` in one
   change set (``add_episode``, origin ``extract``, source the transcript). A conversation with
   nothing personal writes no episode and a log line instead (``log_event``), so it is not asked
   again.
2. Per pending episode: **resolve** its mentions (``resolve``), **extract** (``extract``), then
   one change set (source the episode) with the facts, confirmations, supersessions, journal
   lines, intentions and the ``update_episode`` that marks it ``extracted``, applied by the
   writer with the owner corrections the typed decision confirmed. Facts and the status change
   commit together, so an interrupted run resumes at the first pending episode.
3. Regenerate core.md / index.md / the denylist, then reindex.

Idempotent: a session with episodes, or with a "no episode" log line, is not segmented again;
extracted episodes are not extracted again.

**Parts of a running session** (``turns=``: an interview saves each finished block): only the
turns not segmented yet are segmented, and a ``segmented turns A-B`` log line, committed with the
part's episodes, records how far segmentation got. The next part, the run at close or
``brain memory process --pending`` after a crash starts after it. A session with such lines is
processed once no episode is pending and every owner turn after the last line is segmented.
Without them, a session with episodes or a "no episode" line is segmented as a whole (as before).
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from brain.index.reindex import ReindexStats, reindex
from brain.memory.conversation import in_range, marker_hints, owner_turns, render
from brain.memory.conversation import private_turns as find_private_turns
from brain.memory.extract import ERRORS, extract_episode
from brain.memory.resolve import Resolution, resolve_mentions
from brain.memory.segment import Segment, SegmentReport, segment
from brain.vault.generated import refresh_generated
from brain.vault.models import SELF_ID, Episode
from brain.vault.ops import AddEpisode, ChangeSet, UpdateEpisode
from brain.vault.raw import Transcript, Turn, slugify
from brain.vault.store import SELF_PAGE, Vault
from brain.vault.writer import ApplyResult, apply, log_event, log_has, log_messages

log = logging.getLogger(__name__)

SessionChannel = Literal["chat", "interview", "import"]
NO_EPISODE = "no episode"  # log.md message of a session with nothing to remember
SEGMENTED = "segmented turns"  # log.md message of a segmented part: "segmented turns 1-23: …"
_SEGMENTED = re.compile(rf"^{SEGMENTED} (\d+)-(\d+)\b")
TOPIC_CHARS = 32  # the slug of an episode id


@dataclass
class EpisodeReport:
    id: str
    gist: str
    importance: int
    regions: list[str]
    facts_added: int = 0
    superseded: int = 0
    corrected: int = 0  # supersessions auto-applied as owner corrections
    confirmed: int = 0
    queued: int = 0
    skipped: int = 0
    observations: int = 0
    intentions: int = 0
    created: list[str] = field(default_factory=list)
    review_id: str | None = None
    seconds: float = 0.0
    error: str | None = None
    notes: list[str] = field(default_factory=list)
    result: ApplyResult | None = None


@dataclass
class SessionReport:
    session: str
    transcript: str  # path relative to the vault
    channel: str
    status: str = "processed"  # processed | already processed | nothing said | no episode | failed
    episodes: list[EpisodeReport] = field(default_factory=list)
    segment: SegmentReport | None = None
    seconds: float = 0.0
    segment_seconds: float = 0.0
    generated: list[Path] = field(default_factory=list)
    reindexed: ReindexStats | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def total(self, name: str) -> int:
        return sum(getattr(e, name) for e in self.episodes)

    @property
    def review_ids(self) -> list[str]:
        return [e.review_id for e in self.episodes if e.review_id]

    def summary(self) -> str:
        """One line for the CLI: episodes, +facts, superseded, queued."""
        if self.status != "processed":
            return f"memory: {self.status}" + (f" ({self.error})" if self.error else "")
        parts = [
            f"{len(self.episodes)} episode{'s' if len(self.episodes) != 1 else ''}",
            f"+{self.total('facts_added')} facts",
            f"{self.total('superseded')} superseded",
        ]
        if corrected := self.total("corrected"):
            parts.append(f"{corrected} owner correction{'s' if corrected > 1 else ''}")
        if confirmed := self.total("confirmed"):
            parts.append(f"{confirmed} confirmed")
        parts.append(f"{self.total('queued')} queued")
        return "memory: " + " · ".join(parts) + f" ({self.seconds:.0f}s)"


# --- helpers -----------------------------------------------------------------------------------


def owner_names(vault: Vault) -> list[str]:
    page = vault.page(SELF_PAGE)
    names = [page.title, *page.aliases] if page else []
    out: list[str] = []
    for name in names:
        if name and name != "Profile" and name.casefold() not in map(str.casefold, out):
            out.append(name)
    return out


def session_episodes(vault: Vault, session: str) -> list[Episode]:
    return [e for e in vault.episodes() if e.session == session]


def segmented_through(vault: Vault, transcript: Transcript) -> float:
    """The last turn segmented so far: the end of the latest ``segmented turns A-B`` line;
    ``inf`` for a session segmented as a whole (episodes or a "no episode" line, no part
    lines); 0 when nothing is segmented yet."""
    ends = [
        int(m.group(2))
        for message in log_messages(vault, "extract", transcript.rel)
        if (m := _SEGMENTED.match(message))
    ]
    if ends:
        return float(max(ends))
    if session_episodes(vault, transcript.session_id):
        return math.inf
    return math.inf if log_has(vault, "extract", transcript.rel, NO_EPISODE) else 0.0


def is_processed(vault: Vault, transcript: Transcript) -> bool:
    """Segmented (episodes exist, nothing to remember was logged, or every owner turn is in a
    segmented part) and nothing pending."""
    episodes = session_episodes(vault, transcript.session_id)
    if any(e.status == "pending" for e in episodes):
        return False
    covered = segmented_through(vault, transcript)
    if covered in (0, math.inf):
        return covered == math.inf
    _, turns = transcript.read()
    return not owner_turns([t for t in turns if t.n > covered])


def transcripts(vault: Vault) -> list[Path]:
    base = vault.raw_dir / "conversations"
    return sorted(p for p in base.rglob("*.md") if p.is_file()) if base.is_dir() else []


def pending_transcripts(vault: Vault) -> list[Transcript]:
    """Finished transcripts not (fully) processed yet, oldest first."""
    out = []
    for path in transcripts(vault):
        try:
            transcript = Transcript.open(vault, path)
        except (OSError, ValueError) as err:
            log.warning("unreadable transcript %s: %s", path, err)
            continue
        if not is_processed(vault, transcript):
            out.append(transcript)
    return out


def _episode_id(started: datetime, topic: str, taken: set[str]) -> str:
    base = f"ep-{started:%Y%m%d-%H%M}-{slugify(topic, TOPIC_CHARS)}"
    n, candidate = 1, base
    while candidate in taken:
        n += 1
        candidate = f"{base}-{n}"
    taken.add(candidate)
    return candidate


def _started(turns: Sequence[Turn], first: int, fallback: datetime) -> datetime:
    at = next((t.at for t in turns if t.n >= first), fallback)
    return at.replace(second=0, microsecond=0, tzinfo=None)


def _today(today: date | None) -> date:
    return today or date.today()


def _now(today: date | None) -> datetime:
    return datetime.combine(today, datetime.now().time()) if today else datetime.now()


# --- stages ------------------------------------------------------------------------------------


def write_episodes(
    vault: Vault,
    transcript: Transcript,
    turns: Sequence[Turn],
    segments: Sequence[Segment],
    resolutions: Sequence[list[Resolution]],
    *,
    channel: str,
    started: datetime,
    now: datetime,
    private: set[int] | frozenset[int] = frozenset(),
    event: str | None = None,
) -> list[Episode]:
    """Write the episodes ``pending``. An episode holding a private answer is ``private``; the
    regions and entities the interview markers in its range name are added to its own.
    ``event``: a log line committed with them (the ``segmented turns`` record of a part)."""
    taken = {e.id for e in vault.episodes()}
    ops = []
    for seg, resolved in zip(segments, resolutions, strict=True):
        at = _started(turns, seg.first, started)
        span = in_range(turns, seg.first, seg.last)
        hinted_regions, hinted_entities = marker_hints(span)
        regions = [*seg.regions, *(r for r in hinted_regions if vault.lifemap.region(r))]
        entities = [r.entity_id for r in resolved if r.entity_id]
        entities += [e for e in hinted_entities if vault.page(e) is not None]
        is_private = any(t.n in private for t in span)
        ops.append(
            AddEpisode(
                id=_episode_id(at, seg.topic, taken),
                session=transcript.session_id,
                turns=(seg.first, seg.last),
                channel=channel,  # type: ignore[arg-type]
                regions=list(dict.fromkeys(regions)),
                entities=list(dict.fromkeys(entities)),
                importance=seg.importance,
                privacy="private" if is_private else "normal",
                gist=seg.gist,
                started=at,
                summary=seg.summary,
            )
        )
    cs = ChangeSet(
        ops=ops,
        origin="extract",
        src=[transcript.rel],
        rationale=f"{len(ops)} episode(s) from {transcript.session_id}.",
    )
    result = apply(cs, vault, now=now, event=event)
    for op, reason in result.skipped:
        log.warning("episode not written (%s): %s", reason, getattr(op, "id", op))
    written = set(result.episodes)
    return [e for e in session_episodes(vault, transcript.session_id) if e.id in written]


def process_episode(
    vault: Vault,
    llm: Any,
    episode: Episode,
    turns: Sequence[Turn],
    *,
    resolutions: Sequence[Resolution],
    owner: str,
    names: Sequence[str],
    today: date,
    now: datetime,
) -> EpisodeReport:
    """Extract one pending episode and apply the result (one change set, one commit).
    ``resolutions``: its mentions, resolved at segmentation (empty when resuming)."""
    start = time.perf_counter()
    report = EpisodeReport(episode.id, episode.gist, episode.importance, list(episode.regions))
    resolutions = list(resolutions)
    known = {r.entity_id for r in resolutions if r.entity_id}
    for entity in episode.entities:  # links written at segmentation (or by hand)
        if entity not in known and entity != SELF_ID and (page := vault.page(entity)):
            resolutions.append(Resolution(page.title, page.type, page.subject, "alias"))
    try:
        extracted = extract_episode(
            vault,
            llm,
            episode,
            turns,
            episode_id=episode.id,
            resolutions=resolutions,
            owner=owner,
            owner_names=names,
            today=today,
            importance=episode.importance,
        )
    except ERRORS as err:
        report.error = f"extraction failed: {err}"
        report.seconds = time.perf_counter() - start
        return report
    report.notes += extracted.notes
    linked = [r.entity_id for r in resolutions if r.entity_id] + extracted.created
    ops = [
        *extracted.ops,
        UpdateEpisode(id=episode.id, status="extracted", add_entities=list(dict.fromkeys(linked))),
    ]
    cs = ChangeSet(
        ops=ops,
        origin="extract",
        src=[episode.id],
        rationale=f"From {episode.id}: {episode.gist}",
        created=episode.started,
    )
    result = apply(cs, vault, now=now, corrections=extracted.corrections)
    report.result = result
    report.facts_added, report.superseded = result.facts_added, result.superseded
    report.corrected, report.confirmed = len(result.corrected), result.confirmed
    report.queued, report.skipped = len(result.queued), len(result.skipped)
    report.created, report.review_id = list(result.created), result.review_id
    report.observations = sum(op.op == "add_observation" for op in result.applied)
    report.intentions = sum(op.op == "add_intention" for op in result.applied)
    report.notes += [f"skipped {op.op}: {reason}" for op, reason in result.skipped]
    report.seconds = time.perf_counter() - start
    return report


def process_session(
    vault: Vault,
    llm: Any,
    transcript: Transcript | Path | str,
    *,
    channel: SessionChannel,
    regions_hint: list[str] | None = None,
    today: date | None = None,
    db: sqlite3.Connection | None = None,
    refresh: bool = True,
    core_max_tokens: int = 1500,
    private_turns: list[int] | None = None,
    turns: Sequence[Turn] | None = None,
) -> SessionReport:
    """Turn a finished transcript into memory (see the module docstring). ``today`` fixes the
    date the model resolves relative dates against (and the record date); ``db`` is the index
    to update (opened from the vault when None; ``refresh=False`` skips regeneration and
    reindexing).

    ``private_turns``: owner turns the owner marked private (an interview also marks them in
    the transcript with ``[interview] privacy:: private``; both are honoured). An episode
    holding one is ``privacy: private`` and every fact, page and journal line from it is
    private (``privacy:: private`` on facts, private new pages), so none reaches the big
    model.

    ``turns``: the turns written so far, read by the caller while the session still runs (an
    interview saving a finished block). Only those not segmented yet are segmented, as a part
    (see the module docstring); None reads the whole transcript."""
    start = time.perf_counter()
    if not isinstance(transcript, Transcript):
        transcript = Transcript.open(vault, transcript)
    report = SessionReport(transcript.session_id, transcript.rel, channel)
    partial = turns is not None
    if turns is not None:  # the session still runs: its turns may be being appended
        fm, turns = transcript.frontmatter(), list(turns)
    else:
        fm, turns = transcript.read()
    today_ = _today(today)
    now = _now(today)
    started = fm.get("started")
    started_at = started if isinstance(started, datetime) else (turns[0].at if turns else now)
    if regions_hint is None and channel == "interview" and fm.get("region"):
        regions_hint = [str(fm["region"])]  # `--pending` on an interview transcript
    names = owner_names(vault)
    owner = names[0] if names else "the owner"
    private = find_private_turns(turns, private_turns or ())

    covered = segmented_through(vault, transcript)
    resolved: dict[str, list[Resolution]] = {}
    tail = [t for t in turns if t.n > covered]
    parts = partial or covered > 0  # record the part (a whole session is recorded as before)
    if covered == math.inf and not session_episodes(vault, transcript.session_id):
        report.status = "already processed"  # "no episode" logged
        return _finish(report, start)
    if not parts or owner_turns(tail):
        span = f"{SEGMENTED} {tail[0].n}-{tail[-1].n}" if tail else SEGMENTED
        try:
            seg = segment(
                llm,
                tail,
                lifemap=vault.lifemap,
                owner=owner,
                channel=channel,
                today=today_,
                regions_hint=regions_hint,
            )
        except ERRORS as err:
            report.status, report.error = "failed", f"segmentation failed: {err}"
            return _finish(report, start)
        report.segment = seg
        report.segment_seconds = seg.gate_seconds + seg.seconds
        report.notes += seg.notes
        if not seg.segments:
            report.status = NO_EPISODE if owner_turns(tail) else "nothing said"
            reason = "; ".join(n for n in seg.notes if n.startswith(("gate", "segment")))
            message = f"{span}: {NO_EPISODE} ({reason})" if parts else f"{NO_EPISODE}: {reason}"
            log_event(vault, "extract", [transcript.rel], message, now=now)
        else:
            resolutions = [
                resolve_mentions(
                    vault,
                    llm,
                    s.mentions,
                    render(in_range(turns, s.first, s.last)),
                    owner_names=names,
                )
                for s in seg.segments
            ]
            for r in resolutions:
                report.notes += r.notes
            written = write_episodes(
                vault,
                transcript,
                turns,
                seg.segments,
                [r.resolved for r in resolutions],
                channel=channel,
                started=started_at,
                now=now,
                private=private,
                event=f"{span}: {len(seg.segments)} episode(s)" if parts else None,
            )
            by_turns = {
                (s.first, s.last): r.resolved
                for s, r in zip(seg.segments, resolutions, strict=True)
            }
            resolved = {e.id: by_turns.get(e.turns, []) for e in written}

    last = turns[-1].n if turns else 0
    pending = [
        e
        for e in session_episodes(vault, transcript.session_id)
        if e.status == "pending" and e.turns[1] <= last
    ]
    if not pending:
        if report.status == "processed":
            report.status = "already processed"
        return _finish(report, start)
    report.status = "processed"  # this part may have had nothing; an earlier episode resumes
    for episode in pending:
        report.episodes.append(
            process_episode(
                vault,
                llm,
                episode,
                turns,
                resolutions=resolved.get(episode.id, []),
                owner=owner,
                names=names,
                today=today_,
                now=now,
            )
        )
    errors = [e.error for e in report.episodes if e.error]
    if errors and len(errors) == len(report.episodes):
        report.status, report.error = "failed", errors[0]
    if refresh:
        report.generated = refresh_generated(vault, max_tokens=core_max_tokens, today=today_)
        report.reindexed = _reindex(vault, llm, db, report)
    return _finish(report, start)


def _reindex(
    vault: Vault, llm: Any, db: sqlite3.Connection | None, report: SessionReport
) -> ReindexStats | None:
    from brain.index.db import open_index
    from brain.interview import refresh_questions

    def sync(conn: sqlite3.Connection) -> ReindexStats:
        stats = reindex(vault, conn, llm)
        refresh_questions(vault, conn)  # coverage follows the index
        return stats

    try:
        if db is not None:
            return sync(db)
        conn = open_index(vault.db_path, embed_model=llm.embed_model, dim=llm.embed_dim)
        try:
            return sync(conn)
        finally:
            conn.close()
    except Exception as err:  # noqa: BLE001 - the vault is updated; `brain reindex` catches up
        report.notes.append(f"reindex failed ({err}): run `brain reindex`")
        return None


def _finish(report: SessionReport, start: float) -> SessionReport:
    report.seconds = time.perf_counter() - start
    return report
