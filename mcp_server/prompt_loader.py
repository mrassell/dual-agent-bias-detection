"""Backward-compatible import path. Prefer ``mcp_server.providers.prompt_loader``."""

from mcp_server.providers.prompt_loader import load_llm_prompt

__all__ = ["load_llm_prompt"]
