"""``brain eval lint``: the graph-cleaning pass on a "dirty graph".

A clean fictional vault (``tests/fixtures/vault`` + the richer persona in
``tests/eval/lint/vault``: Alex Rivera, Northwind Labs, Fabrikam Analytics, Westbridge University,
Kestrel Data Cooperative, Sam Okafor and Sam Brennan, Maya Lindqvist, Theo Marchetti, projects
Atlas Migration, Atlas Reporting, Tidepool and Kiln) gets known noise from a seeded script
(``inject``): duplicate pages (a typo, a "University of" prefix, an acronym, an accent or name
order, a project's other name), a topic made into a project page, a project's goal filed on the
owner, a missing ``part_of``, a duty filed under the wrong job (or none), literal employers that
name a page, two current cities, an organisation without its location. The seed picks each
noise's variant. Near misses stay in the clean vault and must not be touched: two different
people called Sam, two projects called Atlas…, a side project with no employer, personal goals,
a university whose location no source states.

Each run: ``run_lint`` on the dirty vault, ``apply_lint`` (questions + one review per finding),
then every lint review is approved in order (the owner saying yes to everything) and the vault
is checked. Scores:

- **fixes**: noise items with a correct operation proposed *and* a vault that is right after
  approval;
- **wrong changes**: proposed operations that match no noise item (must be 0);
- **questions**: questions added; expected ones (two current cities) matched by keywords;
- seconds and model calls.

The clean vault is linted too: any operation there is a wrong change.
"""

from __future__ import annotations

import os
import random
import shutil
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from brain.config import REPO_DIR
from brain.lint import LintOptions, LintReport, apply_lint, run_lint
from brain.lint.names import fold
from brain.vault import gitops
from brain.vault.models import Fact, Page, link_target
from brain.vault.ops import AddFact, DemoteEntity, MergeEntities, Operation, RefileFact
from brain.vault.parse import parse_page
from brain.vault.raw import slugify
from brain.vault.render import render_page
from brain.vault.store import Vault
from brain.vault.writer import approve

EVAL_DIR = REPO_DIR / "tests" / "eval" / "lint"
SEED_VAULT = REPO_DIR / "tests" / "fixtures" / "vault"
OVERLAY = EVAL_DIR / "vault"
EVAL_TODAY = date(2026, 10, 27)
EVAL_NOW = datetime(2026, 10, 27, 12, 0)
NOISE_REC = date(2026, 10, 26)

EP_HISTORY = "ep-20261010-0900-history"
EP_TIDEPOOL = "ep-20261015-1900-tidepool"
EP_GOALS = "ep-20261018-2100-goals"
EP_CAREER = "ep-20261025-1412-career"
CV = "raw/docs/cv.md"
NEAR_MISSES = {
    "two Sams": "people/sam-okafor + people/sam-brennan",
    "two Atlas projects": "projects/atlas-migration + projects/atlas-reporting",
    "side project": "projects/kiln has no employer",
    "personal goals": "self goals ^f-g4h5m6, ^f-g7s2e3",
    "university without a location": "orgs/westbridge-university",
}


# --- gold --------------------------------------------------------------------------------------


@dataclass
class Gold:
    merges: dict[str, frozenset[str]] = field(default_factory=dict)
    demotes: dict[str, str] = field(default_factory=dict)
    refiles: dict[str, tuple[str, str, str]] = field(default_factory=dict)  # fact, field, value
    edges: dict[str, tuple[str, str, str]] = field(default_factory=dict)  # subject, p, object
    asks: dict[str, tuple[str, ...]] = field(default_factory=dict)  # keywords
    variants: dict[str, str] = field(default_factory=dict)

    @property
    def items(self) -> list[str]:
        return [*self.merges, *self.demotes, *self.refiles, *self.edges]


# --- building the vaults -----------------------------------------------------------------------


def clean_vault(root: Path) -> Vault:
    """The fixture vault with the eval persona on top, committed."""
    shutil.copytree(SEED_VAULT, root)
    shutil.copytree(OVERLAY, root, dirs_exist_ok=True)
    gitops.commit_all(root, "eval: clean vault (fictional)")
    return Vault(root)


class _Pages:
    def __init__(self, vault: Vault) -> None:
        self.vault = vault

    def get(self, page_id: str) -> Page:
        path = self.vault.page_path(page_id)
        page, _ = parse_page(path.read_text(encoding="utf-8"), path=path)
        return page

    def put(self, page: Page) -> None:
        path = self.vault.page_path(page.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_page(page), encoding="utf-8")

    def new(self, page_id: str, kind: str, title: str, summary: str, **kw: Any) -> Page:
        page = Page(
            id=page_id,
            type=kind,  # type: ignore[arg-type]
            subject=page_id,
            title=title,
            summary=summary,
            created=NOISE_REC,
            updated=NOISE_REC,
            **kw,
        )
        self.put(page)
        return page

    def add(self, page_id: str, *facts: Fact) -> None:
        page = self.get(page_id)
        page.facts += list(facts)
        self.put(page)

    def edit(self, fact_id: str, **changes: Any) -> None:
        for page in self.vault.pages():
            page = self.get(page.id)
            for section in (page.facts, page.history):
                for n, f in enumerate(section):
                    if f.id == fact_id:
                        section[n] = f.model_copy(update=changes)
                        self.put(page)
                        return
        raise KeyError(fact_id)

    def drop(self, fact_id: str) -> None:
        for page in self.vault.pages():
            page = self.get(page.id)
            if any(f.id == fact_id for f in page.facts):
                page.facts = [f for f in page.facts if f.id != fact_id]
                self.put(page)
                return
        raise KeyError(fact_id)


def _fact(fid: str, text: str, p: str, o: str, src: list[str], **kw: Any) -> Fact:
    extra = kw.pop("extra", {})
    return Fact(
        id=fid,
        text=text,
        p=p,
        o=o,
        rec=NOISE_REC,
        src=src,
        kind=kw.pop("kind", "stated"),
        extra=extra,
        **kw,
    )


def inject(vault: Vault, seed: int) -> Gold:
    """Add the noise (variants chosen by ``seed``) and return what lint should find."""
    rng = random.Random(seed)
    pages = _Pages(vault)
    gold = Gold()

    def pick(name: str, options: list[Any]) -> Any:
        choice = rng.choice(options)
        gold.variants[name] = str(choice)
        return choice

    # 1. an organisation with a typo, which a colleague's job links to
    title = pick("dup-org-typo", ["Nortwind Labs", "Northwind Lab", "Northwnd Labs"])
    dup = f"orgs/{slugify(title)}"
    pages.new(
        dup,
        "org",
        title,
        f"{title} is where Alex works on data pipelines.",
        regions=["career.current_job"],
    )
    pages.add(dup, _fact("f-nb2aaa", f"{title} is based in Porto", "located_in", "Porto", [CV]))
    pages.add(
        "people/sam-brennan",
        _fact("f-nb2aab", f"Works at {title}", "works_at", f"[[{dup}]]", [EP_TIDEPOOL]),
    )
    gold.merges["dup-org-typo"] = frozenset({"orgs/northwind-labs", dup})

    # 2. the university under another form of its name, holding the degree
    title = pick("dup-school-prefix", ["University of Westbridge", "The Westbridge University"])
    dup = f"orgs/{slugify(title)}"
    pages.new(
        dup, "org", title, "Where Alex took a computer science degree.", regions=["education"]
    )
    pages.edit("f-d2g3b4", extra={"at": f"[[{dup}]]"})
    gold.merges["dup-school-prefix"] = frozenset({"orgs/westbridge-university", dup})

    # 3. an acronym, with a second membership
    title = pick("dup-acronym", ["KDC", "K.D.C."])
    pages.new("orgs/kdc", "org", title, f"{title} is a data community Alex goes to.")
    pages.add(
        "self/profile",
        _fact("f-nb2aac", f"Member of {title}", "member_of", "[[orgs/kdc]]", [EP_GOALS]),
    )
    gold.merges["dup-acronym"] = frozenset({"orgs/kestrel-data-cooperative", "orgs/kdc"})

    # 4. a person with an accent or the name the other way round
    title = pick("dup-person", ["Théo Marchetti", "Marchetti, Theo"])
    dup = "people/theo-marchetti-fabrikam"
    pages.new(
        dup,
        "person",
        title,
        "A colleague of Alex's from the Fabrikam years.",
        regions=["relationships.work"],
    )
    pages.add(
        dup,
        _fact(
            "f-nb2aad",
            "Worked with Alex at Fabrikam Analytics",
            "colleague_of",
            "[[self]]",
            [EP_HISTORY],
            extra={"rel": "former colleague"},
            valid_from="2019-06",
            valid_to="2023-01",
        ),
    )
    gold.merges["dup-person"] = frozenset({"people/theo-marchetti", dup})

    # 5. a project under another name
    title = pick("dup-project", ["Project Atlas", "Atlas Migration project", "The Atlas migration"])
    dup = f"projects/{slugify(title)}"
    pages.new(
        dup,
        "project",
        title,
        "Moving the nightly batch pipelines to a managed warehouse.",
        regions=["projects"],
    )
    pages.add(
        dup,
        _fact(
            "f-nb2aae",
            f"{title} is in the testing phase",
            "status",
            "testing",
            [EP_CAREER],
            valid_from="2026-10",
        ),
    )
    gold.merges["dup-project"] = frozenset({"projects/atlas-migration", dup})

    # 6. a topic made into a project page
    title = pick("topic-page", ["Stream Processing", "Data Contracts", "Event Sourcing"])
    topic = f"projects/{slugify(title)}"
    pages.new(
        topic, "project", title, f"Alex is interested in {title.casefold()}.", regions=["projects"]
    )
    pages.add(
        "self/interests",
        _fact("f-nb2aaf", f"Is interested in {title}", "interest", f"[[{topic}]]", [EP_GOALS]),
    )
    gold.demotes["topic-page"] = topic

    # 7. a project's target filed as the owner's goal
    text = pick(
        "goal-on-self",
        [
            "Wants to ship Tidepool's real-time alerting by March 2027",
            "Tidepool should ship real-time alerting by March 2027",
        ],
    )
    pages.add(
        "self/goals",
        _fact("f-nb2aag", text, "goal", "Ship real-time alerting by March 2027", [EP_TIDEPOOL]),
    )
    gold.refiles["goal-on-self"] = ("f-nb2aag", "subject", "projects/tidepool")

    # 8. a missing part_of
    pages.drop("f-t4p5w6")
    gold.edges["missing-part-of"] = ("projects/tidepool", "part_of", "orgs/northwind-labs")

    # 9. a duty under the wrong job, or under none
    at = pick("duty-job", ["northwind", "none"])
    extra = {"at": "[[orgs/northwind-labs]]"} if at == "northwind" else {}
    pages.add(
        "self/profile",
        _fact(
            "f-nb2aah",
            "Built demand forecasting dashboards",
            "responsibility",
            "Demand forecasting dashboards",
            [CV, EP_HISTORY],
            extra=extra,
        ),
    )
    gold.refiles["duty-job"] = ("f-nb2aah", "at", "orgs/fabrikam-analytics")

    # 10. literal employers that name a page
    pages.edit("f-f6b7a2", o="Fabrikam Analytics")
    gold.refiles["literal-self-job"] = ("f-f6b7a2", "o", "orgs/fabrikam-analytics")
    which = pick("literal-colleague", ["theo", "maya"])
    fid = "f-t5w6f7" if which == "theo" else "f-m5w6n7"
    name = "Fabrikam Analytics" if which == "theo" else "Northwind Labs"
    pages.edit(fid, o=name)
    target = "orgs/fabrikam-analytics" if which == "theo" else "orgs/northwind-labs"
    gold.refiles["literal-colleague"] = (fid, "o", target)

    # 11. two current cities
    city = pick("two-cities", ["Braga", "Coimbra"])
    pages.add(
        "self/profile",
        _fact("f-nb2aai", f"Lives in {city}", "lives_in", city, [EP_GOALS], valid_from="2025-06"),
    )
    gold.asks["two-cities"] = ("Porto", city)

    # 12. an organisation that lost its location
    pages.drop("f-f2l3b4")
    gold.edges["org-location"] = ("orgs/fabrikam-analytics", "located_in", "lisbon")

    gitops.commit_all(vault.root, f"eval: noise, seed {seed} (fictional)")
    return gold


# --- scoring -----------------------------------------------------------------------------------


def classify(op: Operation, gold: Gold, report: LintReport) -> str | None:
    """The gold item an operation fixes, or None (a wrong change)."""
    match op:
        case MergeEntities():
            pair = frozenset({op.keep, op.merge})
            return next((k for k, v in gold.merges.items() if v == pair), None)
        case DemoteEntity():
            return next((k for k, v in gold.demotes.items() if v == op.id), None)
        case RefileFact():
            for name, (fid, what, value) in gold.refiles.items():
                if op.fact_id != fid:
                    continue
                got = {"subject": op.subject, "at": op.at, "o": op.o}[what]
                if got is not None and (link_target(got) or got) == value:
                    return name
            for demoted in gold.demotes.values():  # a fact that used the topic as a page
                finding = next((f for f in report.findings if op in f.ops), None)
                if finding and any(
                    isinstance(x, DemoteEntity) and x.id == demoted for x in finding.ops
                ):
                    return "topic-page"
            return None
        case AddFact():
            target = link_target(op.o) or fold(op.o)
            for name, (s, p, o) in gold.edges.items():
                if op.subject == s and op.p == p and target == o:
                    return name
            return None
    return None


def verify(vault: Vault, gold: Gold, name: str) -> bool:
    """Whether the vault is right for gold item ``name`` (after approval)."""
    v = Vault(vault.root)
    if name in gold.merges:
        a, b = sorted(gold.merges[name])
        pa, pb = v.page(a), v.page(b)
        return (pa is None) != (pb is None)
    if name in gold.demotes:
        gone = gold.demotes[name]
        links = [f for p in v.pages() for f in (*p.facts, *p.history) if link_target(f.o) == gone]
        return v.page(gone) is None and not links
    if name in gold.refiles:
        fid, what, value = gold.refiles[name]
        found = v.find_fact(fid)
        if found is None:
            return False
        page, fact, _ = found
        if what == "subject":
            return page.subject == value
        if what == "at":
            return link_target(fact.extra.get("at", "")) == value
        return link_target(fact.o) == value
    if name in gold.edges:
        s, p, o = gold.edges[name]
        facts = [f for pg in v.pages_about(s) for f in pg.facts if f.p == p]
        return any((link_target(f.o) or fold(f.o)) == o for f in facts)
    return False


@dataclass
class RunScore:
    name: str
    gold: Gold | None
    report: LintReport | None = None
    fixed: list[str] = field(default_factory=list)
    proposed_not_verified: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    wrong: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    asks_matched: list[str] = field(default_factory=list)
    asks_missed: list[str] = field(default_factory=list)
    clash_reviews: int = 0
    seconds: float = 0.0
    error: str | None = None

    def counts(self) -> dict[str, int]:
        g = self.gold
        costs = self.report.costs if self.report else None
        return {
            "noise": len(g.items) if g else 0,
            "fixed": len(self.fixed),
            "wrong": len(self.wrong),
            "questions": len(self.questions),
            "asks": len(g.asks) if g else 0,
            "asks_matched": len(self.asks_matched),
            "ops": sum(len(f.ops) for f in self.report.findings) if self.report else 0,
            "decisions": costs.decisions if costs else 0,
            "embedded": costs.embedded if costs else 0,
            "reranked": costs.reranked if costs else 0,
            "pages": self.report.pages if self.report else 0,
        }


def score(
    name: str, vault: Vault, report: LintReport, gold: Gold | None, seconds: float
) -> RunScore:
    s = RunScore(name, gold, report, seconds=seconds)
    from brain.interview.questions import read_questions

    lint_questions = [q for q in read_questions(vault) if q.origin == "lint"]
    s.questions = [q.text for q in lint_questions]
    proposed: set[str] = set()
    for finding in report.findings:
        for op in finding.ops:
            hit = classify(op, gold, report) if gold else None
            if hit is None:
                s.wrong.append(
                    f"{op.op} {op.model_dump(exclude={'op'}, exclude_none=True)} "
                    f"[{finding.summary}]"
                )
            else:
                proposed.add(hit)
    # the owner approves every lint review, in order
    pending_before = {r for r in (res.review_id for res in report.applied) if r}
    for result in report.applied:
        if result.review_id:
            approve(result.review_id, vault, now=EVAL_NOW)
    after = (
        {p.stem for p in vault.reviews_pending_dir.glob("*.md")}
        if vault.reviews_pending_dir.is_dir()
        else set()
    )
    s.clash_reviews = len(after - pending_before)
    if gold:
        for item in gold.items:
            if item in proposed and verify(vault, gold, item):
                s.fixed.append(item)
            elif item in proposed:
                s.proposed_not_verified.append(item)
            else:
                s.missed.append(item)
        for item, words in gold.asks.items():
            if any(all(w.casefold() in q.casefold() for w in words) for q in s.questions):
                s.asks_matched.append(item)
            else:
                s.asks_missed.append(item)
    return s


# --- running -----------------------------------------------------------------------------------


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "digital-self" / "eval-lint"


def run_one(name: str, llm: Any, workdir: Path, seed: int | None) -> RunScore:
    vault = clean_vault(workdir / name / "vault")
    gold = inject(vault, seed) if seed is not None else None
    start = time.perf_counter()
    try:
        report = run_lint(vault, llm, LintOptions(full=True, today=EVAL_TODAY))
        apply_lint(vault, report, now=EVAL_NOW)
    except Exception as err:  # a crash is a result, not the end of the run
        s = RunScore(name, gold, seconds=time.perf_counter() - start)
        s.error = f"{type(err).__name__}: {err}"
        return s
    seconds = time.perf_counter() - start
    return score(name, vault, report, gold, seconds)


def run_json(s: RunScore) -> dict[str, Any]:
    r = s.report
    return {
        "run": s.name,
        "counts": s.counts(),
        "seconds": round(s.seconds, 1),
        "seconds_by_check": {k: round(v, 1) for k, v in (r.costs.seconds.items() if r else [])},
        "variants": s.gold.variants if s.gold else {},
        "fixed": s.fixed,
        "proposed_not_verified": s.proposed_not_verified,
        "missed": s.missed,
        "wrong": s.wrong,
        "questions": s.questions,
        "asks_matched": s.asks_matched,
        "asks_missed": s.asks_missed,
        "clash_reviews": s.clash_reviews,
        "findings": [
            {
                "check": f.check,
                "key": f.key,
                "summary": f.summary,
                "action": f.action,
                "decision": f.decision,
                "p": None if f.p is None else round(f.p, 3),
                "question": f.ask.text if f.ask else None,
            }
            for f in (r.findings if r else [])
        ],
        "notes": r.notes if r else [],
        "error": s.error,
    }


def report_json(scores: list[RunScore], meta: dict[str, Any]) -> dict[str, Any]:
    dirty = [s for s in scores if s.gold]
    total: dict[str, int] = {}
    for s in dirty:
        for k, v in s.counts().items():
            total[k] = total.get(k, 0) + v
    clean = [s for s in scores if not s.gold]
    return {
        "meta": {**meta, "near_misses": NEAR_MISSES},
        "overall": {
            **total,
            "wrong_on_clean": sum(len(s.wrong) for s in clean),
            "questions_on_clean": sum(len(s.questions) for s in clean),
            "seconds": round(sum(s.seconds for s in scores), 1),
        },
        "runs": [run_json(s) for s in scores],
    }
