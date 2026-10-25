"""Cheap token estimates for prompt budgeting (no tokenizer dependency).

Qwen tokenizers average ~4 characters per token on English prose; we divide by 3.5 so budgets
err on the safe side (markdown, ids and numbers tokenize worse than prose).
"""

from __future__ import annotations

import math

CHARS_PER_TOKEN = 3.5


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN) if text else 0
