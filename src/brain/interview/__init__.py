"""Interview mode: coverage of the life map, region choice, questions.md, the conversation
(docs/architecture.md, Flow 2)."""

from brain.interview.coverage import CoverageReport, RegionCoverage, SlotCoverage, coverage_report
from brain.interview.planner import PlanItem, RegionError, build_plan, choose_region
from brain.interview.questions import (
    GapItem,
    add_question,
    read_questions,
    refresh_questions,
    resolve_question,
    write_questions,
)
from brain.interview.session import CloseResult, InterviewSession, Reply

__all__ = [
    "CloseResult",
    "CoverageReport",
    "GapItem",
    "InterviewSession",
    "PlanItem",
    "RegionCoverage",
    "RegionError",
    "Reply",
    "SlotCoverage",
    "add_question",
    "build_plan",
    "choose_region",
    "coverage_report",
    "read_questions",
    "refresh_questions",
    "resolve_question",
    "write_questions",
]
