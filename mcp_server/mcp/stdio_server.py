"""MCP **transport** only: FastMCP stdio adapter.

This module must not import domain evaluators directly. It delegates to
``mcp_server.bias_surface`` (capability API) and applies MCP-level ``log_call``
wrapping for each tool invocation.

**Abstraction:** swapping LLM vendors/models is env-driven via ``mcp_server.providers``
(``BIAS_LLM_*``); this file stays unchanged.
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_server import bias_surface as surface
from mcp_server.auditing import init_db, log_call

mcp = FastMCP(
    "bias-detection",
    instructions=(
        "All tools delegate to mcp_server.bias_surface (capability layer). "
        "Model-agnostic LLMs: BIAS_LLM_<SLOT>_VENDOR, _MODEL, keys — see mcp_server.providers. "
        "Set BASIL_DATA_DIR. Required for RoBERTa/NLI tools: AUDITOR_MODEL_ID, NLI_MODEL_NAME. "
        "Optional: MCP_AUDIT_DB. "
        "Heavy tools write under outputs/; use run_basil_dual_agent_eval(max_sentences=200) for smoke tests."
    ),
)


def _wrap_log(name: str, arguments: dict[str, Any], result: Any) -> Any:
    if os.environ.get("MCP_DISABLE_AUDIT", "").lower() not in ("1", "true", "yes"):
        if name != "audit_recent":
            log_call(name, arguments, result)
    return result


@mcp.tool()
def detect_bias(text: str) -> dict[str, Any]:
    """Lexical (+ mirrored informational) scores as fixed JSON."""
    arguments = {"text": text[:2000]}
    return _wrap_log("detect_bias", arguments, surface.run_detect_bias(text))


@mcp.tool()
def nli_check(premise: str, hypothesis: str) -> dict[str, Any]:
    """BART-MNLI: premise vs hypothesis, or leave hypothesis blank for explicit bias templates."""
    arguments = {"premise": premise[:4000], "hypothesis": hypothesis[:2000]}
    return _wrap_log("nli_check", arguments, surface.run_nli_check(premise, hypothesis))


@mcp.tool()
def evaluate_basil(max_sentences: int = 0, threshold: float = 0.5) -> dict[str, Any]:
    """BASIL test split (event holdout); gold = any span. max_sentences 0 = all."""
    arguments = {"max_sentences": max_sentences, "threshold": threshold}
    return _wrap_log(
        "evaluate_basil",
        arguments,
        surface.run_evaluate_basil(max_sentences=max_sentences, threshold=threshold),
    )


@mcp.tool()
def basil_outlet_drift() -> dict[str, Any]:
    """Outlet drift (Jensen–Shannon) on BASIL test buckets."""
    arguments: dict[str, Any] = {}
    return _wrap_log("basil_outlet_drift", arguments, surface.run_basil_outlet_drift())


@mcp.tool()
def temporal_drift_analysis(buckets_json: str) -> dict[str, Any]:
    """Temporal / generic bucket drift (JSON array of buckets)."""
    arguments = {"buckets_json": buckets_json[:20000]}
    return _wrap_log(
        "temporal_drift_analysis",
        arguments,
        surface.run_temporal_drift_analysis(buckets_json),
    )


@mcp.tool()
def audit_recent(limit: int = 30) -> dict[str, Any]:
    """Recent rows from the SQLite audit log (not re-logged)."""
    return surface.run_audit_recent(limit)


@mcp.tool()
def sweep_auditor_thresholds() -> dict[str, Any]:
    """Nine-threshold sweep + plots + threshold_choice.json."""
    arguments: dict[str, Any] = {}
    return _wrap_log(
        "sweep_auditor_thresholds",
        arguments,
        surface.run_sweep_auditor_thresholds(),
    )


@mcp.tool()
def export_audit_artifacts() -> dict[str, Any]:
    """Export audit DB to outputs/ CSV + Markdown sample."""
    arguments: dict[str, Any] = {}
    return _wrap_log(
        "export_audit_artifacts",
        arguments,
        surface.run_export_audit_artifacts(),
    )


@mcp.tool()
def run_basil_dual_agent_eval(max_sentences: int = 0) -> dict[str, Any]:
    """RoBERTa + BART-NLI verifier on BASIL test (full split if max_sentences=0)."""
    arguments = {"max_sentences": max_sentences}
    return _wrap_log(
        "run_basil_dual_agent_eval",
        arguments,
        surface.run_basil_dual_agent_eval(max_sentences=max_sentences),
    )


@mcp.tool()
def llm_bias_score(sentence: str, slot: str = "A") -> dict[str, Any]:
    """Cloud LLM bias score. Slot = env group BIAS_LLM_<SLOT>_VENDOR / _MODEL."""
    arguments = {"sentence": sentence[:4000], "slot": slot.strip().upper()[:16]}
    return _wrap_log(
        "llm_bias_score",
        arguments,
        surface.run_llm_bias_score(sentence, slot.strip(), internal_audit=False),
    )


@mcp.tool()
def llm_verify(sentence: str, auditor_note: str, slot: str = "B") -> dict[str, Any]:
    """Cloud LLM verifier (verified / confidence / reasoning) for a flagged sentence."""
    arguments = {
        "sentence": sentence[:4000],
        "auditor_note": auditor_note[:2000],
        "slot": slot.strip().upper()[:16],
    }
    return _wrap_log(
        "llm_verify",
        arguments,
        surface.run_llm_verify(sentence, auditor_note, slot.strip(), internal_audit=False),
    )


def main() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    init_db()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
