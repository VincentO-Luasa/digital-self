"""Model access. Only this package talks to the model server."""

from brain.llm.base import ChatResult, RetrievalModel, StructuredOutputError, ToolCall
from brain.llm.client import LLM
from brain.llm.fake import FakeLLM

__all__ = ["LLM", "ChatResult", "FakeLLM", "RetrievalModel", "StructuredOutputError", "ToolCall"]
