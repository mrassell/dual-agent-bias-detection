"""Backward-compatible import path. Prefer ``mcp_server.auditing.invocation``."""

from mcp_server.auditing.invocation import (
    basil_dual_agent_nli_invocation,
    llm_slot_invocation,
    merge_invocation,
)

__all__ = [
    "basil_dual_agent_nli_invocation",
    "llm_slot_invocation",
    "merge_invocation",
]
