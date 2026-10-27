"""A fictional interview about lint doubts (Alex Rivera persona), shared by the unit tests and the
live end-to-end check.

The vault: the lint eval's clean persona (jobs at Northwind Labs, current, and Fabrikam
Analytics, past) plus ``tests/fixtures/interview/doubts``: a homepage listing projects without
employers, and the pages ingest made from it: Stream processing, Ingestion (a topic made into a
project page), Fraud alerts and Transaction monitoring (one project under two names).
questions.md holds the doubts lint would add about them.

``Owner`` plays Alex in the live check: it answers by what the brain's message is about, knows
what it already told and what it declined, and counts the brain's repeats.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from interview_helpers import Env  # type: ignore[import-not-found]

from brain.eval.lint_eval import clean_vault
from brain.index.db import open_index
from brain.index.reindex import reindex
from brain.interview.doubts import entity_infos
from brain.interview.planner import PlanItem, build_plan
from brain.interview.questions import add_question, doubt_key, read_questions
from brain.lint.edges import part_of_text
from brain.llm.fake import FakeLLM
from brain.vault import gitops
from brain.vault.generated import refresh_generated
from brain.vault.store import Vault

OVERLAY = Path(__file__).resolve().parents[1] / "fixtures" / "interview" / "doubts"
TODAY = date(2026, 10, 27)
REGION = "projects"
SP, ING = "projects/stream-processing", "projects/ingestion"
FRAUD, TM = "projects/fraud-alerts", "projects/transaction-monitoring"
HOMEPAGE = "raw/docs/2026-10-20-homepage.md"
DOUBTS = [  # (text as lint words it, doubt key, about, priority)
    (part_of_text("Stream processing", []), f"part_of {SP}", SP, 4),
    ("Is Stream processing a project, or a topic?", f"type {SP}", SP, 4),
    ("Is Ingestion a project, or a topic?", f"type {ING}", ING, 3),
    (
        "Are Transaction monitoring and Fraud alerts the same project?",
        f"merge {FRAUD} {TM}",
        FRAUD,
        3,
    ),
]


def build(root: Path) -> Env:
    vault = clean_vault(root)
    shutil.copytree(OVERLAY, root, dirs_exist_ok=True)
    for text, doubt, about, priority in DOUBTS:
        add_question(
            vault,
            text,
            region=REGION,
            reason=f"lint {doubt}: scenario",
            priority=priority,
            origin="lint",
            about=about,
            doubt=doubt,
        )
    gitops.commit_all(root, "interview doubts scenario (fictional)")
    vault = Vault(root)
    refresh_generated(vault, today=TODAY)
    llm = FakeLLM()
    db = open_index(vault.db_path, embed_model=llm.embed_model, dim=llm.embed_dim)
    reindex(vault, db, llm)
    return Env(vault, db)


def plan_for(env: Env) -> tuple[list[PlanItem], dict]:
    items = read_questions(env.vault)
    infos = entity_infos(env.vault, [doubt_key(i) for i in items if i.open], today=TODAY)
    return build_plan(env.report(TODAY), REGION, items, infos=infos), infos


# --- the owner, for the live check -------------------------------------------------------------


SP_ANSWER = (
    "It was a project I built at Fabrikam for a media client, a streaming pipeline for their "
    "click data. Ingestion is different: not a project, just what I work on day to day at "
    "Northwind."
)
MERGE_ANSWER = (
    "Yes, same thing: fraud alerts was the old name of transaction monitoring, a rules engine "
    "I built at Fabrikam."
)
ASK_BACK = "It depends, can you tell me more about these?"
DECLINE = "I'd rather not say, it's confidential."
SCRAPER = "It fed a reporting scraper, one of the tools I built for that media client."
GENERIC = {
    "atlas reporting": "That one is finished, it was a dashboard suite at Fabrikam.",
    "atlas": "We went live in December, a few weeks late. Now I'm cleaning up the old cluster.",
    "kiln": "Still working on Kiln on weekends, slowly.",
    "tidepool": "Tidepool is still a prototype.",
}


def _has(text: str, *words: str) -> bool:
    return all(re.search(rf"\b{re.escape(w)}", text) for w in words)


@dataclass
class Owner:
    """Answers like Alex would; counts what the brain should not have asked."""

    told: set[str] = field(default_factory=set)  # sp, ingestion, merge, scraper
    declined_client: bool = False
    asked_back: bool = False
    repeats: list[str] = field(default_factory=list)
    declined_reasked: list[str] = field(default_factory=list)
    context_replies: list[str] = field(default_factory=list)

    def answer(self, message: str) -> str:
        m = message.casefold()
        if "client" in m and re.search(r"\b(name|which|who|what client|for whom)\b", m):
            if self.declined_client:
                self.declined_reasked.append(message)
                return "Like I said, I can't name clients."
            self.declined_client = True
            return DECLINE
        if (
            "scraper" in self.told
            and "media client" in m
            and re.search(r"what (was|is) it\b(?!s)|what did you build|which tool|what tool", m)
        ):
            self.repeats.append(message)
            return "The reporting scraper, as I said."
        if "fraud" in m or "transaction monitoring" in m:
            if "merge" in self.told:
                if _has(m, "same") or _has(m, "one and the same") or _has(m, "different"):
                    self.repeats.append(message)
                return "It was a rules engine for card payments."
            if not self.asked_back:
                self.asked_back = True
                return ASK_BACK
            self.told.add("merge")
            return MERGE_ANSWER
        if "stream" in m:
            if "sp" in self.told and re.search(
                r"\b(topic|tool|side project|on your own|part of your work|which (of your )?jobs?)\b",
                m,
            ):
                self.repeats.append(message)
                return "As I said, a project at Fabrikam."
            if "sp" not in self.told:
                self.told |= {"sp", "ingestion"}
                return SP_ANSWER
            self.told.add("scraper")
            return SCRAPER
        if "ingestion" in m:
            if "ingestion" in self.told and re.search(r"\b(project|topic|tool)\b", m):
                self.repeats.append(message)
                return "Like I said, it's not a project."
            self.told.add("ingestion")
            return "It's not a project, it's what I work on at Northwind."
        for key, text in GENERIC.items():
            if key in m:
                return text
        if re.search(r"keep going|stop here", m):
            return "Let's stop here."
        return "Nothing more on that for now."
