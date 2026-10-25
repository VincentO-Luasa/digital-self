"""Settings, loaded from ~/.config/local-brain/config.toml with environment overrides."""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

CONFIG_PATH = Path.home() / ".config" / "local-brain" / "config.toml"
REPO_DIR = Path(__file__).resolve().parents[2]


class LLMSettings(BaseModel):
    base_url: str = "http://127.0.0.1:8000/v1"
    chat_model: str = "Qwen3.6-35B-A3B-4bit"
    embed_model: str = "Qwen3-Embedding-0.6B-8bit"
    rerank_model: str = "Qwen3-Reranker-0.6B-mxfp8"
    chat_temperature: float = 0.7
    chat_top_p: float = 0.8
    chat_top_k: int = 20
    extract_temperature: float = 0.2
    timeout_s: float = 300.0


class RetrievalSettings(BaseModel):
    budget_tokens: int = 1000
    candidates: int = 40
    rerank_top: int = 12


class EscalationSettings(BaseModel):
    """Where hard questions go. The author uses Claude Code; any big model works."""

    provider: Literal["claude-code", "clipboard", "openai-compatible"] = "clipboard"
    display_name: str = "the big model"  # banner: "→ Ask <display_name>"
    base_url: str | None = None  # openai-compatible only
    model: str | None = None
    api_key_env: str | None = None


class Settings(BaseModel):
    vault: Path = Path.home() / "brain-vault"
    llm: LLMSettings = LLMSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    escalation: EscalationSettings = EscalationSettings()
    core_max_tokens: int = 1500

    @property
    def db_path(self) -> Path:
        return self.vault / ".brain" / "index.sqlite"


@lru_cache
def load_settings(path: Path | None = None) -> Settings:
    path = path or CONFIG_PATH
    data: dict = {}
    if path.exists():
        data = tomllib.loads(path.read_text())
    if vault := os.environ.get("BRAIN_VAULT"):
        data["vault"] = vault
    if base_url := os.environ.get("BRAIN_LLM_BASE_URL"):
        data.setdefault("llm", {})["base_url"] = base_url
    settings = Settings.model_validate(data)
    settings.vault = settings.vault.expanduser()
    return settings
