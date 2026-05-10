"""Load LLM system prompts from packaged files or env override (no model IDs)."""

from __future__ import annotations

import os
from pathlib import Path

_PKG = Path(__file__).resolve().parent


def load_llm_prompt(kind: str) -> str:
    """kind: ``scoring`` | ``verifier`` → ``prompts/llm_*.txt`` next to this module."""
    env_key = f"BIAS_LLM_PROMPT_{kind.upper()}_FILE"
    override = os.environ.get(env_key, "").strip()
    if override:
        p = Path(override).expanduser()
        if p.is_file():
            return p.read_text(encoding="utf-8")
        raise FileNotFoundError(f"{env_key}={override} is not a readable file")

    fname = {
        "scoring": "llm_bias_scoring.txt",
        "verifier": "llm_verifier.txt",
    }.get(kind, f"llm_{kind}.txt")
    path = _PKG / "prompts" / fname
    if not path.is_file():
        raise FileNotFoundError(f"Missing packaged prompt: {path}")
    return path.read_text(encoding="utf-8")
