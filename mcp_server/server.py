"""MCP server: BASIL eval, bias scores, NLI, drift, audit."""

from __future__ import annotations

import json
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_server.audit import init_db, log_call, recent_events
from mcp_server.auditor import get_auditor
from mcp_server.basil_eval import run_basil_evaluation
from mcp_server.basil_paths import resolve_basil_data_dir
from mcp_server.bias import build_detection_result
from mcp_server.drift import temporal_drift_analysis as run_drift
from mcp_server.nli import nli_check as run_nli

mcp = FastMCP(
    "bias-detection",
    instructions=(
        "Set BASIL_DATA_DIR to the folder of BASIL *.json articles. "
        "Optional: AUDITOR_MODEL_ID, NLI_MODEL_NAME."
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
    auditor = get_auditor()
    pred = auditor.predict_one(text)
    result = build_detection_result(text, pred, auditor.model_id).model_dump()
    result["demo_title"] = "Bias scores"
    result["demo_readout"] = f"lexical_score={result['lexical_score']:.3f}"
    return _wrap_log("detect_bias", arguments, result)


@mcp.tool()
def nli_check(premise: str, hypothesis: str) -> dict[str, Any]:
    """Natural language inference: is the hypothesis supported by the premise?"""
    arguments = {"premise": premise[:4000], "hypothesis": hypothesis[:2000]}
    result = run_nli(premise, hypothesis)
    return _wrap_log("nli_check", arguments, result)


@mcp.tool()
def evaluate_basil(max_sentences: int = 0, threshold: float = 0.5) -> dict[str, Any]:
    """Test split (grouped by event_id); gold = any BASIL span on sentence. max_sentences 0 = all."""
    arguments = {"max_sentences": max_sentences, "threshold": threshold}
    try:
        data_dir = resolve_basil_data_dir()
    except FileNotFoundError as e:
        err: dict[str, Any] = {
            "demo_title": "BASIL evaluation",
            "demo_readout": str(e),
            "error": "basil_not_found",
        }
        return _wrap_log("evaluate_basil", arguments, err)

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
    return _wrap_log("evaluate_basil", arguments, metrics)


@mcp.tool()
def basil_outlet_drift() -> dict[str, Any]:
    """Outlet buckets from the same test split as evaluate_basil; Jensen–Shannon between low/high mean score."""
    arguments: dict[str, Any] = {}
    try:
        data_dir = resolve_basil_data_dir()
    except FileNotFoundError as e:
        out = {
            "demo_title": "BASIL outlet drift",
            "demo_readout": str(e),
            "error": "basil_not_found",
        }
        return _wrap_log("basil_outlet_drift", arguments, out)

    threshold = float(os.environ.get("AUDITOR_THRESHOLD", "0.5"))
    metrics = run_basil_evaluation(data_dir, max_sentences=None, threshold=threshold)
    buckets = metrics.get("outlet_buckets_for_drift_tool", [])
    buckets = sorted(
        buckets,
        key=lambda b: float(b.get("bias_tag_counts", {}).get("mean_lexical_score", 0.0)),
    )
    if len(buckets) < 2:
        out = {
            "demo_title": "BASIL outlet drift",
            "demo_readout": "Need at least two outlets in the test split.",
            "error": "insufficient_buckets",
        }
        return _wrap_log("basil_outlet_drift", arguments, out)

    result = run_drift(buckets)
    result["demo_title"] = "BASIL outlets: bias-mix divergence"
    js = result.get("jensen_shannon_divergence", "n/a")
    interp = result.get("interpretation", "n/a")
    lo = buckets[0]["period"]
    hi = buckets[-1]["period"]
    result["demo_readout"] = f"outlets {lo} → {hi} by mean lexical score; JS={js} ({interp})"
    result["buckets_used"] = buckets
    return _wrap_log("basil_outlet_drift", arguments, result)


@mcp.tool()
def temporal_drift_analysis(buckets_json: str) -> dict[str, Any]:
    """Compare bias tag counts (or means) between time buckets — Jensen–Shannon divergence."""
    arguments = {"buckets_json": buckets_json[:20000]}
    buckets = json.loads(buckets_json)
    if not isinstance(buckets, list):
        result: dict[str, Any] = {
            "demo_title": "Temporal drift",
            "demo_readout": "Invalid input: need a JSON array of time buckets.",
            "error": "buckets_json_must_be_a_json_array",
        }
    else:
        result = run_drift(buckets)
        if "error" not in result:
            result["demo_title"] = "Temporal drift (bias mix)"
            interp = result.get("interpretation", "n/a")
            js = result.get("jensen_shannon_divergence", "n/a")
            result["demo_readout"] = f"first vs last bucket: JS={js} ({interp})"
    return _wrap_log("temporal_drift_analysis", arguments, result)


@mcp.tool()
def audit_recent(limit: int = 30) -> dict[str, Any]:
    events = recent_events(min(200, max(1, limit)))
    n = len(events)
    return {
        "demo_title": "Audit log (replay)",
        "demo_readout": f"{n} recent audited tool calls",
        "count": n,
        "events": events,
    }


def main() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    init_db()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
