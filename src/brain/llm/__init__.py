"""Model access. Only this package talks to the model server."""

from brain.llm.base import (
    ChatResult,
    OutputTruncated,
    RetrievalModel,
    StructuredOutputError,
    ToolCall,
)
from brain.llm.client import LLM
from brain.llm.decide import ChoiceQuestion, Decision, PredicateQuestion
from brain.llm.fake import FakeLLM

__all__ = [
    "LLM",
    "ChatResult",
    "ChoiceQuestion",
    "Decision",
    "FakeLLM",
    "OutputTruncated",
    "PredicateQuestion",
    "RetrievalModel",
    "StructuredOutputError",
    "ToolCall",
]
