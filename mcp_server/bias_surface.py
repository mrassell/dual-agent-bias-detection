"""Strict **capability surface** for bias / BASIL / NLI / drift / audit / jobs.

**Rule:** All MCP tools and matching CLIs should go through these callables.
``mcp_server.mcp`` is **transport only** (FastMCP stdio): it maps tool names
→ ``run_*`` here and applies MCP-level auditing. This module may compose
``pipeline_jobs`` and domain modules; the server must not import them directly.

Notebooks and experiments may import this module instead of the MCP adapter.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from mcp_server.auditing import log_call, recent_events
from mcp_server.auditing.invocation import llm_slot_invocation
from mcp_server.auditor import get_auditor
from mcp_server.providers import build_chat_backend, resolve_slot_config
from mcp_server.basil_eval import run_basil_evaluation
from mcp_server.basil_paths import resolve_basil_data_dir
from mcp_server.bias import build_detection_result
from mcp_server.drift import temporal_drift_analysis as run_drift
from mcp_server.nli import nli_check as run_nli
from mcp_server.providers.prompt_loader import load_llm_prompt
from mcp_server.pipeline_jobs import (
    run_audit_export as _pipeline_run_audit_export,
    run_full_basil_auditor_nli_eval as _pipeline_run_full_basil_auditor_nli_eval,
    run_threshold_sweep as _pipeline_run_threshold_sweep,
)


def _parse_llm_json(raw: str) -> dict[str, Any]:
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    return json.loads(raw)


def run_llm_bias_score(
    sentence: str,
    slot: str,
    *,
    internal_audit: bool = True,
) -> dict[str, Any]:
    """Cloud LLM bias score for ``sentence`` using env-backed slot ``A`` / ``B`` / …

    When ``internal_audit`` is True, appends one SQLite row (same schema as MCP tools).
    Set False when the MCP transport layer already logs the enclosing tool call.
    """
    cfg = resolve_slot_config(slot)
    backend = build_chat_backend(cfg)
    system = load_llm_prompt("scoring")
    want_json = cfg.vendor in ("openai", "openai_compatible")
    raw = backend.complete(
        system,
        f"Sentence: {sentence}",
        max_tokens=400,
        json_object=want_json,
    )
    data = _parse_llm_json(raw)
    result: dict[str, Any] = {
        "bias_score": float(data.get("bias_score", 0.5)),
        "bias_type": str(data.get("bias_type", "none")),
        "reasoning": str(data.get("reasoning", "")),
        "demo_title": "LLM bias score",
        "demo_readout": (
            f"slot={cfg.slot} {cfg.vendor}:{cfg.model} "
            f"score={float(data.get('bias_score', 0.5)):.3f}"
        ),
        "llm_slot": cfg.slot,
        "llm_vendor": cfg.vendor,
        "llm_model": cfg.model,
    }
    if internal_audit:
        log_call(
            "llm_bias_score",
            {"sentence": sentence[:500], "slot": cfg.slot},
            result,
            invocation=llm_slot_invocation(
                slot=cfg.slot,
                vendor=cfg.vendor,
                model=cfg.model,
                purpose="bias_scoring",
            ),
        )
    return result


def run_llm_verify(
    sentence: str,
    auditor_note: str,
    slot: str,
    *,
    internal_audit: bool = True,
) -> dict[str, Any]:
    """Second-agent verification (JSON verified/confidence/reasoning)."""
    cfg = resolve_slot_config(slot)
    backend = build_chat_backend(cfg)
    system = load_llm_prompt("verifier")
    want_json = cfg.vendor in ("openai", "openai_compatible")
    user_msg = f"Sentence: {sentence}\n\nAuditor note: {auditor_note}"
    raw = backend.complete(
        system,
        user_msg,
        max_tokens=400,
        json_object=want_json,
    )
    data = _parse_llm_json(raw)
    result: dict[str, Any] = {
        "verified": bool(data.get("verified", False)),
        "confidence": float(data.get("confidence", 0.0)),
        "reasoning": str(data.get("reasoning", "")),
        "demo_title": "LLM verify",
        "demo_readout": f"slot={cfg.slot} verified={data.get('verified')} conf={data.get('confidence')}",
        "llm_slot": cfg.slot,
        "llm_vendor": cfg.vendor,
        "llm_model": cfg.model,
    }
    if internal_audit:
        log_call(
            "llm_verify",
            {"sentence": sentence[:500], "slot": cfg.slot},
            result,
            invocation=llm_slot_invocation(
                slot=cfg.slot,
                vendor=cfg.vendor,
                model=cfg.model,
                purpose="bias_verify",
            ),
        )
    return result


def run_detect_bias(text: str) -> dict[str, Any]:
    """Lexical (+ mirrored informational) scores as fixed JSON (primary auditor path)."""
    auditor = get_auditor()
    pred = auditor.predict_one(text)
    result = build_detection_result(text, pred, auditor.model_id).model_dump()
    result["demo_title"] = "Bias scores"
    result["demo_readout"] = f"lexical_score={result['lexical_score']:.3f}"
    return result


def run_nli_check(premise: str, hypothesis: str) -> dict[str, Any]:
    """NLI (BART-MNLI). Empty ``hypothesis`` triggers the explicit bias-template pack."""
    return run_nli(premise, hypothesis)


def run_evaluate_basil(max_sentences: int = 0, threshold: float = 0.5) -> dict[str, Any]:
    """Test split (grouped by event_id); gold = any BASIL span on sentence. max_sentences 0 = all."""
    try:
        data_dir = resolve_basil_data_dir()
    except FileNotFoundError as e:
        return {
            "demo_title": "BASIL evaluation",
            "demo_readout": str(e),
            "error": "basil_not_found",
        }

    limit = None if max_sentences <= 0 else max_sentences
    metrics = run_basil_evaluation(data_dir, max_sentences=limit, threshold=threshold)
    n = metrics["sentence_count"]
    f1 = metrics["f1_macro"]
    acc = metrics["accuracy"]
    metrics["demo_title"] = "BASIL test split (real labels)"
    metrics["demo_readout"] = (
        f"n={n} (event_id holdout). acc={acc:.3f} macro_f1={f1:.3f} "
        f"mae_vs_gold={metrics['mae_score_vs_gold_binary']:.3f}"
    )
    return metrics


def run_basil_outlet_drift() -> dict[str, Any]:
    """Outlet buckets from the same test split as evaluate_basil; Jensen–Shannon between low/high mean score."""
    try:
        data_dir = resolve_basil_data_dir()
    except FileNotFoundError as e:
        return {
            "demo_title": "BASIL outlet drift",
            "demo_readout": str(e),
            "error": "basil_not_found",
        }

    threshold = float(os.environ.get("AUDITOR_THRESHOLD", "0.5"))
    metrics = run_basil_evaluation(data_dir, max_sentences=None, threshold=threshold)
    buckets = metrics.get("outlet_buckets_for_drift_tool", [])
    buckets = sorted(
        buckets,
        key=lambda b: float(b.get("bias_tag_counts", {}).get("mean_lexical_score", 0.0)),
    )
    if len(buckets) < 2:
        return {
            "demo_title": "BASIL outlet drift",
            "demo_readout": "Need at least two outlets in the test split.",
            "error": "insufficient_buckets",
        }

    result = run_drift(buckets)
    result["demo_title"] = "BASIL outlets: bias-mix divergence"
    js = result.get("jensen_shannon_divergence", "n/a")
    interp = result.get("interpretation", "n/a")
    lo = buckets[0]["period"]
    hi = buckets[-1]["period"]
    result["demo_readout"] = (
        f"outlets {lo} → {hi} by mean lexical score; JS={js} ({interp})"
    )
    result["buckets_used"] = buckets
    return result


def run_temporal_drift_analysis(buckets_json: str) -> dict[str, Any]:
    """Compare bias tag counts (or means) between time buckets — Jensen–Shannon divergence."""
    buckets = json.loads(buckets_json)
    if not isinstance(buckets, list):
        return {
            "demo_title": "Temporal drift",
            "demo_readout": "Invalid input: need a JSON array of time buckets.",
            "error": "buckets_json_must_be_a_json_array",
        }
    result = run_drift(buckets)
    if "error" not in result:
        result["demo_title"] = "Temporal drift (bias mix)"
        interp = result.get("interpretation", "n/a")
        js = result.get("jensen_shannon_divergence", "n/a")
        result["demo_readout"] = f"first vs last bucket: JS={js} ({interp})"
    return result


def run_audit_recent(limit: int = 30) -> dict[str, Any]:
    """SQLite audit tail (same data as MCP ``audit_recent`` tool)."""
    events = recent_events(min(200, max(1, limit)))
    n = len(events)
    return {
        "demo_title": "Audit log (replay)",
        "demo_readout": f"{n} recent audited tool calls",
        "count": n,
        "events": events,
    }


def run_sweep_auditor_thresholds() -> dict[str, Any]:
    """Nine-threshold sweep on BASIL test; writes CSV, JSON, plots under ``outputs/``."""
    try:
        return _pipeline_run_threshold_sweep()
    except FileNotFoundError as e:
        return {
            "demo_title": "Threshold sweep",
            "demo_readout": str(e),
            "error": "basil_not_found",
        }
    except Exception as e:
        return {
            "demo_title": "Threshold sweep",
            "demo_readout": str(e),
            "error": "sweep_failed",
        }


def run_export_audit_artifacts() -> dict[str, Any]:
    """Export audit SQLite → ``outputs/audit_log_export.csv`` + ``audit_log_sample.md``."""
    return _pipeline_run_audit_export()


def run_basil_dual_agent_eval(max_sentences: int = 0) -> dict[str, Any]:
    """RoBERTa auditor + BART-MNLI verifier on BASIL test; ``max_sentences`` 0 = full split."""
    lim = None if max_sentences <= 0 else max_sentences
    try:
        return _pipeline_run_full_basil_auditor_nli_eval(max_sentences=lim)
    except FileNotFoundError as e:
        return {
            "demo_title": "BASIL dual-agent eval",
            "demo_readout": str(e),
            "error": "basil_not_found",
        }
    except Exception as e:
        return {
            "demo_title": "BASIL dual-agent eval",
            "demo_readout": str(e),
            "error": "eval_failed",
        }
