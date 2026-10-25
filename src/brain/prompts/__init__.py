"""Pinned prompt templates. Each template is a markdown file in this package.

Templates are versioned with the code: changing one changes the request, so recorded LLM
fixtures that replay it must be re-recorded (the replay tests compare the prompts).
"""

from __future__ import annotations

from functools import cache
from importlib import resources


@cache
def load_prompt(name: str) -> str:
    """The template ``<name>.md`` with its leading ``<!-- ... -->`` comment removed."""
    text = resources.files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8")
    if text.startswith("<!--"):
        text = text[text.index("-->") + 3 :]
    return text.strip() + "\n"
