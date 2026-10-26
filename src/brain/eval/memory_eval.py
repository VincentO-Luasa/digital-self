"""``brain eval memory``: run the memory pipeline on fictional transcripts and score it.

Each transcript in ``tests/eval/memory/sessions/`` (persona Alex Rivera, format in
``tests/eval/README.md``) is written into a fresh copy of the fixture vault
(``tests/fixtures/vault``) and processed with ``process_session`` (real model, date fixed to
``EVAL_TODAY``). What the writer applied and queued is compared with the transcript's gold:

- episodes: count within the gold range; each gold episode's regions met by a produced one;
- facts: recall of gold facts among the applied facts (``add_fact`` and the new fact of an
  applied ``supersede_fact``); precision = applied facts matching a gold or ``acceptable`` fact;
- supersessions: gold owner corrections auto-applied with the right value and dates;
- **wrongly auto-applied**: an auto-applied supersession the gold does not expect, or an applied
  fact matching a ``forbid`` entry. Must be 0;
- queued: gold contradictions found in the review queue;
- intentions: gold intentions written (keywords, date window);
- duplicates: current facts that say the same as another current fact of the subject.

Matching reuses ``ingest_eval``'s token similarity (case, accents, stop words, light suffixes).
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from brain.config import REPO_DIR
from brain.eval.ingest_eval import FREE_TEXT, best_similarity, tokens
from brain.vault import gitops
from brain.vault.models import PREDICATES, SELF_ID, link_target
from brain.vault.ops import AddFact, AddIntention, FactSpec, SupersedeFact
from brain.vault.raw import Transcript
from brain.vault.store import Vault

EVAL_DIR = REPO_DIR / "tests" / "eval" / "memory"
SEED_VAULT = REPO_DIR / "tests" / "fixtures" / "vault"
EVAL_TODAY = date(2026, 10, 27)


# --- gold --------------------------------------------------------------------------------------


def _tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(str(v) for v in value) if isinstance(value, list) else (str(value),)


def _date(value: Any) -> str | None:
    return None if value is None else str(value)


@dataclass(frozen=True)
class GoldFact:
    s: str
    ps: tuple[str, ...]
    os: tuple[str, ...]
    start: str | None = None
    end: str | None = None

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> GoldFact:
        return cls(
            str(raw["s"]),
            _tuple(raw["p"]),
            _tuple(raw["o"]),
            _date(raw.get("from")),
            _date(raw.get("to")),
        )

    def describe(self) -> str:
        return f"{self.s} {'/'.join(self.ps)} {'/'.join(self.os)}"


@dataclass(frozen=True)
class GoldSupersede:
    fact: str
    p: str | None
    os: tuple[str, ...]
    start: str | None = None
    end: str | None = None


@dataclass(frozen=True)
class GoldIntention:
    words: tuple[str, ...]
    after: tuple[str, str] | None
    optional: bool = False


@dataclass
class GoldCase:
    path: Path
    started: datetime
    channel: str
    turns: list[tuple[str, str]]
    count: tuple[int, int]
    regions: list[tuple[str, ...]]
    facts: list[GoldFact] = field(default_factory=list)
    acceptable: list[GoldFact] = field(default_factory=list)
    queued: list[GoldFact] = field(default_factory=list)
    forbid: list[GoldFact] = field(default_factory=list)
    supersede: list[GoldSupersede] = field(default_factory=list)
    acceptable_supersede: tuple[str, ...] = ()
    intentions: list[GoldIntention] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.stem


def load_case(path: Path) -> GoldCase:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    gold = data.get("gold") or {}
    episodes = gold.get("episodes") or {}
    count = episodes.get("count", 1)
    lo, hi = (count, count) if isinstance(count, int) else (int(count[0]), int(count[1]))
    turns = []
    for item in data["turns"]:
        [(role, text)] = item.items()
        turns.append(("user" if role == "user" else "assistant", str(text)))
    started = data["started"]
    started = started if isinstance(started, datetime) else datetime.fromisoformat(str(started))
    return GoldCase(
        path=path,
        started=started,
        channel=str(data.get("channel", "chat")),
        turns=turns,
        count=(lo, hi),
        regions=[_tuple(r) for r in episodes.get("regions") or []],
        facts=[GoldFact.parse(f) for f in gold.get("facts") or []],
        acceptable=[GoldFact.parse(f) for f in gold.get("acceptable") or []],
        queued=[GoldFact.parse(f) for f in gold.get("queued") or []],
        forbid=[GoldFact.parse(f) for f in gold.get("forbid") or []],
        supersede=[
            GoldSupersede(
                str(s["fact"]),
                s.get("p"),
                _tuple(s.get("o")),
                _date(s.get("from")),
                _date(s.get("to")),
            )
            for s in gold.get("supersede") or []
        ],
        acceptable_supersede=_tuple(gold.get("acceptable_supersede")),
        intentions=[
            GoldIntention(
                _tuple(i["text"]),
                tuple(map(str, i["after"])) if i.get("after") else None,  # type: ignore[arg-type]
                bool(i.get("optional", False)),
            )
            for i in gold.get("intentions") or []
        ],
    )


def load_set(eval_dir: Path = EVAL_DIR, only: Sequence[str] = ()) -> list[GoldCase]:
    paths = sorted((eval_dir / "sessions").glob("*.yaml"))
    return [load_case(p) for p in paths if not only or any(o in p.stem for o in only)]


def gold_problems(case: GoldCase, seed: Vault | None = None) -> list[str]:
    """Predicates outside the vocabulary, unknown regions, superseded ids not in the seed."""
    seed = seed or Vault(SEED_VAULT)
    out = []
    for fact in (*case.facts, *case.acceptable, *case.queued, *case.forbid):
        out += [f"unknown predicate {p}" for p in fact.ps if p not in PREDICATES]
    for regions in case.regions:
        out += [f"unknown region {r}" for r in regions if not seed.lifemap.region(r)]
    for s in (*[s.fact for s in case.supersede], *case.acceptable_supersede):
        if seed.find_fact(s) is None:
            out.append(f"unknown fact {s} in the seed vault")
    if not any(role == "user" for role, _ in case.turns):
        out.append("no owner turn")
    lo, hi = case.count
    if lo > hi or len(case.regions) > hi:
        out.append("episode count range")
    return out


# --- what the pipeline did -----------------------------------------------------------------------


@dataclass
class GotFact:
    subject: str  # entity id or self
    p: str
    o: str
    start: str | None
    end: str | None
    how: str  # applied | corrected | queued
    fact_id: str | None = None  # the fact a supersession replaces

    def describe(self) -> str:
        when = f" [{self.start or ''}–{self.end or ''}]" if self.start or self.end else ""
        old = f" (replaces ^{self.fact_id})" if self.fact_id else ""
        return f"{self.subject} {self.p} {self.o}{when}{old}"


def _got(subject: str, spec: FactSpec, how: str, fact_id: str | None = None) -> GotFact:
    s = link_target(subject) or subject
    s = SELF_ID if s in (SELF_ID, "self/profile") else s
    return GotFact(s, spec.p, spec.o, spec.valid_from, spec.valid_to, how, fact_id)


def collect(report: Any, vault: Vault) -> tuple[list[GotFact], list[GotFact], list[Any], list[Any]]:
    """(applied facts, queued facts, applied supersessions, applied intentions)."""
    applied: list[GotFact] = []
    queued: list[GotFact] = []
    supersessions: list[Any] = []
    intentions: list[Any] = []
    for ep in report.episodes:
        result = ep.result
        if result is None:
            continue
        corrected = set(result.corrected)
        for op in result.applied:
            if isinstance(op, AddFact):
                applied.append(_got(op.subject, op, "applied"))
            elif isinstance(op, SupersedeFact):
                found = vault.find_fact(op.fact_id)
                subject = found[0].subject if found else SELF_ID
                supersessions.append((op, op.fact_id in corrected))
                if op.new:
                    applied.append(_got(subject, op.new, "corrected", op.fact_id))
            elif isinstance(op, AddIntention):
                intentions.append(op)
        for op, _ in result.queued:
            if isinstance(op, AddFact):
                queued.append(_got(op.subject, op, "queued"))
            elif isinstance(op, SupersedeFact) and op.new:
                found = vault.find_fact(op.fact_id)
                queued.append(
                    _got(found[0].subject if found else SELF_ID, op.new, "queued", op.fact_id)
                )
    return applied, queued, supersessions, intentions


# --- matching ----------------------------------------------------------------------------------


def _names(vault: Vault, entity: str) -> list[str]:
    if entity == SELF_ID:
        return ["self"]
    page = vault.page(entity)
    return [page.title, *page.aliases] if page else [entity]


def _object_names(vault: Vault, o: str) -> list[str]:
    target = link_target(o)
    return _names(vault, target) if target else [o]


def _dates_ok(want: str | None, got: str | None) -> bool:
    return want is None or (got is not None and got.startswith(want))


def matches(gold: GoldFact, got: GotFact, vault: Vault, *, dates: bool = True) -> bool:
    if got.p not in gold.ps:
        return False
    if gold.s == "self":
        if got.subject != SELF_ID:
            return False
    elif got.subject == SELF_ID or not best_similarity([gold.s], _names(vault, got.subject)):
        return False
    golds = ["self"] if gold.os == ("self",) else list(gold.os)
    names = _object_names(vault, got.o)
    if gold.os == ("self",):
        if link_target(got.o) != SELF_ID:
            return False
    elif not best_similarity(golds, names, free_text=got.p in FREE_TEXT):
        return False
    return not dates or (_dates_ok(gold.start, got.start) and _dates_ok(gold.end, got.end))


def _intention_ok(gold: GoldIntention, op: AddIntention) -> bool:
    words = set(tokens(op.text))
    if not all(set(tokens(w)) <= words for w in gold.words):
        return False
    if gold.after is None:
        return True
    return op.after is not None and gold.after[0] <= op.after <= gold.after[1]


def duplicates(vault: Vault) -> list[str]:
    """Current facts of a subject that say the same as another one (same p, value, dates)."""
    seen: dict[tuple[str, str, str, str | None, str | None], str] = {}
    out = []
    for page in vault.pages():
        for f in page.facts:
            value = " ".join(tokens(" ".join(_object_names(vault, f.o))))
            key = (page.subject, f.p, value, f.valid_from, f.valid_to)
            if key in seen:
                out.append(f"^{f.id} repeats ^{seen[key]} ({f.p} {f.o})")
            else:
                seen[key] = f.id
    return out


# --- scoring -----------------------------------------------------------------------------------


@dataclass
class CaseScore:
    gold: GoldCase
    episodes: int = 0
    episode_regions: list[list[str]] = field(default_factory=list)
    regions_matched: int = 0
    facts_matched: int = 0
    applied: int = 0
    applied_matched: int = 0
    supersede_ok: int = 0
    queued_matched: int = 0
    intentions_matched: int = 0
    intentions_expected: int = 0
    wrong_auto: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    applied_facts: list[str] = field(default_factory=list)
    queued_facts: list[str] = field(default_factory=list)
    intentions: list[str] = field(default_factory=list)
    status: str = ""
    seconds: float = 0.0
    segment_seconds: float = 0.0
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def count_ok(self) -> bool:
        lo, hi = self.gold.count
        return lo <= self.episodes <= hi

    def counts(self) -> dict[str, int]:
        g = self.gold
        return {
            "cases": 1,
            "count_ok": int(self.count_ok),
            "regions": len(g.regions),
            "regions_matched": self.regions_matched,
            "facts": len(g.facts),
            "facts_matched": self.facts_matched,
            "applied": self.applied,
            "applied_matched": self.applied_matched,
            "supersede": len(g.supersede),
            "supersede_ok": self.supersede_ok,
            "queued": len(g.queued),
            "queued_matched": self.queued_matched,
            "intentions": self.intentions_expected,
            "intentions_matched": self.intentions_matched,
            "wrong_auto": len(self.wrong_auto),
            "duplicates": len(self.duplicates),
        }


def score_case(
    case: GoldCase,
    report: Any,
    vault: Vault,
    *,
    seconds: float,
) -> CaseScore:
    s = CaseScore(case, seconds=seconds, status=report.status)
    s.segment_seconds = report.segment_seconds
    episodes = [e for e in vault.episodes() if e.session == report.session]
    s.episodes = len(episodes)
    s.episode_regions = [list(e.regions) for e in episodes]
    for regions in case.regions:
        if any(_region_hit(regions, e.regions) for e in episodes):
            s.regions_matched += 1
    applied, queued, supersessions, intentions = collect(report, vault)
    s.applied_facts = [g.describe() + f" ({g.how})" for g in applied]
    s.queued_facts = [g.describe() for g in queued]
    s.intentions = [f"{op.text} (after {op.after})" for op in intentions]

    # facts: greedy one-to-one
    free = list(applied)
    for gold in case.facts:
        hit = next((g for g in free if matches(gold, g, vault)), None)
        if hit is None:
            loose = next((g for g in free if matches(gold, g, vault, dates=False)), None)
            s.misses.append(gold.describe() + (" (dates differ)" if loose else ""))
            continue
        s.facts_matched += 1
        free.remove(hit)
    gold_new = {(sup.fact) for sup in case.supersede}
    for got in free:
        if got.how == "corrected" and got.fact_id in gold_new:
            continue  # scored as a supersession below
        if any(matches(a, got, vault, dates=False) for a in case.acceptable):
            continue
        if got.how == "corrected" and got.fact_id in case.acceptable_supersede:
            continue
        s.unverified.append(got.describe())
    s.applied = len(applied)
    s.applied_matched = s.applied - len(s.unverified)

    # supersessions
    for gold in case.supersede:
        ok = False
        for op, auto in supersessions:
            if op.fact_id != gold.fact or not auto:
                continue
            new = op.new
            if new is None:
                ok = not gold.os
            elif (gold.p is None or new.p == gold.p) and (
                not gold.os or best_similarity(list(gold.os), _object_names(vault, new.o))
            ):
                ok = _dates_ok(gold.start, new.valid_from) and _dates_ok(gold.end, new.valid_to)
        if ok:
            s.supersede_ok += 1
        else:
            s.misses.append(f"supersede ^{gold.fact} → {'/'.join(gold.os) or 'ended'}")
    allowed = {g.fact for g in case.supersede} | set(case.acceptable_supersede)
    for op, auto in supersessions:
        if auto and op.fact_id not in allowed:
            new = f" → {op.new.p} {op.new.o}" if op.new else " (ended)"
            s.wrong_auto.append(f"auto-applied supersession of ^{op.fact_id}{new}")
    for got in applied:
        if any(matches(f, got, vault, dates=False) for f in case.forbid):
            s.wrong_auto.append(f"applied forbidden fact {got.describe()}")

    # queued contradictions
    pool = list(queued)
    for gold in case.queued:
        hit = next((g for g in pool if matches(gold, g, vault, dates=False)), None)
        if hit:
            s.queued_matched += 1
            pool.remove(hit)
        else:
            s.misses.append(f"queued {gold.describe()}")

    # intentions
    for gold in case.intentions:
        hit = next((op for op in intentions if _intention_ok(gold, op)), None)
        if not gold.optional or hit:
            s.intentions_expected += 1
        if hit:
            s.intentions_matched += 1
            intentions.remove(hit)
        elif not gold.optional:
            s.misses.append(f"intention {' '.join(gold.words)}")
    s.duplicates = duplicates(vault)
    s.notes = list(report.notes) + [n for e in report.episodes for n in e.notes]
    if report.error:
        s.error = report.error
    return s


def _region_hit(wanted: Iterable[str], got: Iterable[str]) -> bool:
    """A produced region equals a wanted one, or one is a dotted child of the other."""
    return any(
        x == r or x.startswith(r + ".") or r.startswith(x + ".") for r in wanted for x in got
    )


def _ratio(n: int, d: int) -> float | None:
    return round(n / d, 3) if d else None


def metrics(c: dict[str, int], seconds: float) -> dict[str, float | int | None]:
    return {
        "episode_count_accuracy": _ratio(c["count_ok"], c["cases"]),
        "region_recall": _ratio(c["regions_matched"], c["regions"]),
        "fact_recall": _ratio(c["facts_matched"], c["facts"]),
        "fact_precision": _ratio(c["applied_matched"], c["applied"]),
        "supersessions_correct": _ratio(c["supersede_ok"], c["supersede"]),
        "queued_recall": _ratio(c["queued_matched"], c["queued"]),
        "intention_recall": _ratio(c["intentions_matched"], c["intentions"]),
        "wrongly_auto_applied": c["wrong_auto"],
        "duplicates": c["duplicates"],
        "seconds": round(seconds, 1),
    }


def totals(scores: Sequence[CaseScore]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in scores:
        for k, v in s.counts().items():
            out[k] = out.get(k, 0) + v
    return out


# --- running -----------------------------------------------------------------------------------


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "digital-self" / "eval-memory"


def seed_vault(root: Path) -> Vault:
    shutil.copytree(SEED_VAULT, root)
    gitops.commit_all(root, "eval seed vault (fictional)")
    return Vault(root)


def write_transcript(vault: Vault, case: GoldCase) -> Transcript:
    transcript = Transcript.start(vault, channel=case.channel, now=case.started)
    for n, (role, text) in enumerate(case.turns):
        transcript.append(role, text, at=case.started + timedelta(minutes=n))  # type: ignore[arg-type]
    transcript.commit()
    return transcript


def run_case(case: GoldCase, llm: Any, workdir: Path, *, today: date = EVAL_TODAY) -> CaseScore:
    from brain.memory import process_session

    vault = seed_vault(workdir / case.name / "vault")
    transcript = write_transcript(vault, case)
    start = time.perf_counter()
    try:
        report = process_session(
            vault,
            llm,
            transcript,
            channel=case.channel,
            today=today,
            refresh=False,  # type: ignore[arg-type]
        )
    except Exception as err:  # a crash on one transcript is a result, not the end of the run
        score = CaseScore(case, seconds=time.perf_counter() - start)
        score.error = f"{type(err).__name__}: {err}"
        return score
    return score_case(case, report, vault, seconds=time.perf_counter() - start)


def case_json(s: CaseScore) -> dict[str, Any]:
    return {
        "transcript": s.gold.name,
        "metrics": metrics(s.counts(), s.seconds),
        "status": s.status,
        "episodes": s.episodes,
        "episode_regions": s.episode_regions,
        "segment_seconds": round(s.segment_seconds, 1),
        "applied": s.applied_facts,
        "queued": s.queued_facts,
        "intentions": s.intentions,
        "misses": s.misses,
        "unverified": s.unverified,
        "wrongly_auto_applied": s.wrong_auto,
        "duplicates": s.duplicates,
        "error": s.error,
        "notes": s.notes,
    }


def report_json(scores: Sequence[CaseScore], meta: dict[str, Any]) -> dict[str, Any]:
    seconds = sum(s.seconds for s in scores)
    return {
        "meta": meta,
        "overall": metrics(totals(scores), seconds),
        "counts": totals(scores),
        "transcripts": [case_json(s) for s in scores],
    }


def names(cases: Iterable[GoldCase]) -> list[str]:
    return [c.name for c in cases]
