"""Check 4: missing structural edges, as multiple-choice questions over existing pages.

- **Project → job** (``part_of``): a project page with no ``part_of``, when the owner has jobs
  with an organisation page. Options: those jobs (the ones active while the project ran first),
  "a side project", "unclear". A project no source sentence names is asked about directly (the
  model, given nothing, says "side project"). An employer at ``ACT_P`` adds ``part_of``
  (``kind:: inferred``, ``evidence::`` = the sources that name the project, and the ``works_at``
  fact); a confident
  side project adds nothing; anything else asks the owner ("Is Tidepool part of your work at
  Northwind Labs, or a side project?").
- **Duty → job** (``at::``): an undated duty with no ``at::`` while the owner had more than one
  job (or only past ones), and a duty whose ``at::`` job its sources never name while they name
  another job. Options: the jobs, "unclear". At ``ACT_P``: ``refile_fact`` with ``at``.
- **Organisation → place** (``located_in``), only from source text: places the vault knows (place
  pages, values of ``lives_in`` / ``located_in`` / ``hometown``) that the sentences naming the
  organisation also name. Options: those places, "another place" (the true one may not be a
  place the vault knows: never a forced choice), "not stated". At ``ACT_P``: ``located_in``,
  ``kind:: inferred`` with the sentence as evidence.

Edges are inferred, so every one carries evidence and goes to review.
"""

from __future__ import annotations

from brain.lint.common import ASK_P, Context
from brain.lint.graph import Job, Ref, names_in
from brain.lint.names import fold
from brain.lint.report import Ask, Finding
from brain.lint.subjects import related_sentences
from brain.llm.decide import Choice, ChoiceQuestion
from brain.prompts import load_prompt
from brain.vault.models import SELF_ID, Page, link_target
from brain.vault.ops import AddFact, RefileFact

ACT_P = 0.8
SIDE, UNCLEAR, NOT_STATED, ELSEWHERE = "a side project", "unclear", "not stated", "another place"
MAX_JOBS = 4
PLACE_PREDICATES = ("lives_in", "located_in", "hometown")
QUOTE_CHARS = 120


def _distinct(jobs: list[Job]) -> list[Job]:
    """One option per employer (the most recent stint), with or without a page."""
    seen: set[str] = set()
    out = []
    for job in jobs:
        if fold(job.name) not in seen:
            seen.add(fold(job.name))
            out.append(job)
    return out


def _jobs_listing(jobs: list[Job]) -> str:
    return "\n".join(f"- {j.name} ({j.period()})" for j in jobs)


def _quote(text: str) -> str:
    text = " ".join(text.split()).replace('"', "'")
    return text if len(text) <= QUOTE_CHARS else text[: QUOTE_CHARS - 1].rstrip() + "…"


def _overlap(job: Job, start: str | None, end: str | None) -> bool:
    if not (start or end) or not (job.start or job.end):
        return True
    return (job.start or "0000") <= (end or "9999~") and (start or "0000") <= (job.end or "9999~")


def _period(ctx: Context, page: Page) -> tuple[str | None, str | None]:
    dates = []
    for ref in [
        *ctx.graph.facts_of(page.subject, history=True),
        *ctx.graph.incoming.get(page.subject, []),
    ]:
        dates += [d for d in (ref.fact.valid_from, ref.fact.valid_to) if d]
    return (min(dates), max(dates)) if dates else (None, None)


# --- project → job -----------------------------------------------------------------------------


def part_of_question(jobs: list[Job]) -> ChoiceQuestion:
    return ChoiceQuestion(
        name="part_of",
        instructions="According to the text, where does the owner do this project?",
        choices=[  # the answers that are not a job come first: a listed name draws the eye
            Choice(value=UNCLEAR, description="the text does not say"),
            Choice(value=SIDE, description="on their own, as a side project outside any job"),
            *(Choice(value=j.name, description=f"at {j.name}") for j in jobs),
        ],
    )


def _part_of(ctx: Context, findings: list[Finding]) -> None:
    g = ctx.graph
    jobs = _distinct(g.jobs)  # employers without a page are options too: never a forced choice
    if not any(j.org for j in jobs):
        return
    for page in sorted(g.entities.values(), key=lambda p: p.id):
        if page.type != "project" or not ctx.in_scope(page.id) or not ctx.free(page.id):
            continue
        if any(r.fact.p == "part_of" for r in g.facts_of(page.subject, history=True)):
            continue
        start, end = _period(ctx, page)
        options = sorted(jobs, key=lambda j: not _overlap(j, start, end))[:MAX_JOBS]
        excerpt, used = g.excerpt(g.sources_about(page.subject), g.names(page))
        if not excerpt:  # no source names it: the model would guess "side project"; ask the owner
            findings.append(_part_of_ask(ctx, page, options[0].name, "no source names it"))
            continue
        text = load_prompt("lint_part_of").format(
            project=g.describe(page), sources=excerpt, jobs=_jobs_listing(options)
        )
        v = ctx.decide(text, part_of_question(options), f"edges part_of {page.id}")
        if v is None:
            continue
        job = next((j for j in options if j.name == v.choice), None)
        if v.choice == SIDE and v.acts(ACT_P):
            ctx.note(f"edges: {page.id} is a side project ({v.summary()})")
            continue
        finding = Finding(
            check="edges",
            key=f"part_of {page.id}",
            summary=f"{page.title} ({page.id}) has no part_of",
            about=[page.id],
            p=v.p,
            decision=v.summary(),
            src=used,
        )
        if job and job.org and v.acts(ACT_P):
            lines = [ln.lstrip("- ") for ln in excerpt.split("\n") if ln.strip()]
            line = next((ln for ln in lines if names_in(ln, [job.name])), lines[0] if lines else "")
            quote = _quote(line) if line else ""
            finding.summary += f": part of the work at {job.name}"
            finding.ops = [
                AddFact(
                    subject=page.id,
                    p="part_of",
                    o=f"[[{job.org}]]",
                    text=f"{page.title} is part of the owner's work at {job.name}",
                    kind="inferred",
                    evidence=[*used, job.fact_id, *([quote] if quote else [])],
                )
            ]
        else:
            likely = job.name if job else options[0].name
            finding.ask = _part_of_ask(ctx, page, likely, f"{v.choice} p {v.p:.2f}").ask
        findings.append(finding)


def _part_of_ask(ctx: Context, page: Page, likely: str, why: str) -> Finding:
    return Finding(
        check="edges",
        key=f"part_of {page.id}",
        summary=f"{page.title} ({page.id}) has no part_of ({why})",
        about=[page.id],
        ask=Ask(
            text=f"Is {page.title} part of your work at {likely}, or a side project?",
            reason=f"lint part_of {page.id}: {why}",
            about=page.id,
            region=ctx.region(page.id, "projects"),
            private=ctx.private(page.id),
        ),
    )


# --- duty → job --------------------------------------------------------------------------------


def job_question(jobs: list[Job]) -> ChoiceQuestion:
    return ChoiceQuestion(
        name="job",
        instructions="At which job did the owner have this duty?",
        choices=[
            Choice(value=UNCLEAR, description="the text does not say"),
            *(Choice(value=j.name, description=f"at {j.name} ({j.period()})") for j in jobs),
        ],
    )


def _duty_doubt(ctx: Context, ref: Ref, jobs: list[Job]) -> str | None:
    f = ref.fact
    at = link_target(f.extra.get("at", ""))
    if at is None:
        if f.valid_from or f.valid_to:
            return None  # its dates place it
        if len(ctx.graph.jobs) > 1 or not any(j.current for j in ctx.graph.jobs):
            return "undated, no job"
        return None
    sentences = " ".join(related_sentences(ctx, ref, []))  # the ones about the duty
    named = {j.org for j in jobs if j.org and names_in(sentences, [j.name, *_aliases(ctx, j)])}
    if named and at not in named:
        return f"filed at {ctx.graph.title(at)}, its source names another job"
    return None


def _aliases(ctx: Context, job: Job) -> list[str]:
    page = ctx.graph.pages.get(job.org or "")
    return list(page.aliases) if page else []


def _duties(ctx: Context, findings: list[Finding]) -> None:
    g = ctx.graph
    jobs = _distinct(g.jobs)[:MAX_JOBS]
    if not any(j.org for j in jobs) or not ctx.in_scope(SELF_ID, *(j.org for j in jobs if j.org)):
        return
    for ref in g.facts_of(SELF_ID):
        if ref.fact.p != "responsibility" or not (why := _duty_doubt(ctx, ref, jobs)):
            continue
        sentences = related_sentences(ctx, ref, [])
        text = load_prompt("lint_duty_job").format(
            duty=g.readable(ref.fact.text),
            sources="\n".join(f"- {s}" for s in sentences) or "(nothing more)",
            jobs=_jobs_listing(jobs),
        )
        v = ctx.decide(text, job_question(jobs), f"edges job ^{ref.fact.id}")
        if v is None or v.p < ASK_P:
            continue
        job = next((j for j in jobs if j.name == v.choice), None)
        current_at = link_target(ref.fact.extra.get("at", ""))
        if job and job.org == current_at:
            continue  # the duty is where it is filed
        finding = Finding(
            check="edges",
            key=f"job {ref.fact.id}",
            summary=f"duty ^{ref.fact.id} {ref.fact.text!r}: {why}",
            about=[ref.fact.id],
            p=v.p,
            decision=v.summary(),
        )
        if job and job.org and v.acts(ACT_P):
            finding.summary += f": at {job.name}"
            finding.ops = [RefileFact(fact_id=ref.fact.id, at=f"[[{job.org}]]")]
        else:
            options = " or ".join(j.name for j in jobs[:2])
            finding.ask = Ask(
                text=f"“{g.readable(ref.fact.text).rstrip('.')}”: was that at {options}?",
                reason=f"lint job {ref.fact.id}: {v.choice} p {v.p:.2f}",
                region="career.history",
            )
        findings.append(finding)


# --- organisation → place ----------------------------------------------------------------------


def _known_places(ctx: Context) -> dict[str, str]:
    """Folded place name → the object to write (a place page link or the literal)."""
    g = ctx.graph
    out: dict[str, str] = {}
    for page in g.entities.values():
        if page.type == "place":
            for name in g.names(page):
                out.setdefault(fold(name), f"[[{page.id}]]")
    for ref in g.refs:
        if ref.fact.p in PLACE_PREDICATES and not link_target(ref.fact.o):
            out.setdefault(fold(ref.fact.o), ref.fact.o)
    return {k: v for k, v in out.items() if k}


def located_question(org: str, places: list[str]) -> ChoiceQuestion:
    return ChoiceQuestion(
        name="located",
        instructions=f"According to the text, where is {org} based?",
        choices=[  # first: with them last, a listed city won over the one the text named
            Choice(value=ELSEWHERE, description="a place these options do not list"),
            Choice(value=NOT_STATED, description="the text does not say where it is based"),
            *(Choice(value=p, description=f"{org} is based in {p}") for p in places),
        ],
    )


def _located(ctx: Context, findings: list[Finding]) -> None:
    g = ctx.graph
    places = _known_places(ctx)
    if not places:
        return
    for page in sorted(g.entities.values(), key=lambda p: p.id):
        if page.type != "org" or not ctx.in_scope(page.id) or not ctx.free(page.id):
            continue
        if any(r.fact.p == "located_in" for r in g.facts_of(page.subject, history=True)):
            continue
        excerpt, used = g.excerpt(g.sources_about(page.subject), g.names(page))
        lines = [ln for ln in excerpt.split("\n") if ln.strip()]
        found = {
            g.readable(obj): obj
            for key, obj in places.items()
            if any(fold(key) and f" {key} " in f" {fold(ln)} " for ln in lines)
        }
        found = {k: v for k, v in found.items() if fold(k) not in {fold(n) for n in g.names(page)}}
        if not found:
            continue
        options = sorted(found)[:MAX_JOBS]
        text = load_prompt("lint_located_in").format(org=g.describe(page), sources=excerpt)
        v = ctx.decide(text, located_question(page.title, options), f"edges located_in {page.id}")
        if v is None or v.choice in (NOT_STATED, ELSEWHERE) or v.p < ASK_P:
            continue
        quote = next(
            (_quote(ln.lstrip("- ")) for ln in lines if v.choice.casefold() in ln.casefold()), ""
        )
        finding = Finding(
            check="edges",
            key=f"located_in {page.id}",
            summary=f"{page.title} ({page.id}) has no located_in; its sources say {v.choice}",
            about=[page.id],
            p=v.p,
            decision=v.summary(),
            src=used,
        )
        if v.acts(ACT_P):
            finding.ops = [
                AddFact(
                    subject=page.id,
                    p="located_in",
                    o=found[v.choice],
                    text=f"{page.title} is based in {v.choice}",
                    kind="inferred",
                    evidence=[*used, *([quote] if quote else [])],
                )
            ]
        else:
            finding.ask = Ask(
                text=f"Is {page.title} based in {v.choice}?",
                reason=f"lint located_in {page.id}: {v.choice} p {v.p:.2f}",
                about=page.id,
                region=ctx.region(page.id),
                private=ctx.private(page.id),
            )
        findings.append(finding)


def check_edges(ctx: Context) -> list[Finding]:
    findings: list[Finding] = []
    _part_of(ctx, findings)
    _duties(ctx, findings)
    _located(ctx, findings)
    return findings
