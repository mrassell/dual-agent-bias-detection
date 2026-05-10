"""Metadata merged into audit rows (transport, LLM slot, orchestration)."""

from __future__ import annotations

from typing import Any, Mapping


def merge_invocation(
    arguments: dict[str, Any],
    invocation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not invocation:
        return dict(arguments)
    return {"__audit": dict(invocation), **dict(arguments)}


def llm_slot_invocation(
    *,
    slot: str,
    vendor: str,
    model: str,
    purpose: str,
    transport: str = "capability_surface",
) -> dict[str, str]:
    return {
        "transport": transport,
        "llm_slot": slot.upper(),
        "llm_vendor": vendor,
        "llm_model": model,
        "purpose": purpose,
    }


def basil_dual_agent_nli_invocation(hypothesis_excerpt: str) -> dict[str, str]:
    return {
        "transport": "capability_surface",
        "orchestration": "basil_dual_agent_nli",
        "hypothesis_excerpt": hypothesis_excerpt[:200],
    }
