"""Long-running evaluation jobs callable from MCP tools or CLI scripts.

Keeps heavy logic importable so ``python -m mcp_server`` can expose the same
workflows that used to be script-only (threshold sweep, audit export, full
dual-agent BASIL eval).
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from mcp_server.auditing import audit_db_path, log_call
from mcp_server.auditing.invocation import basil_dual_agent_nli_invocation
from mcp_server.auditor import get_auditor
from mcp_server.basil_dataset import load_basil_sentences, split_sentence_frame
from mcp_server.basil_paths import resolve_basil_data_dir
from mcp_server.nli import BIAS_VERIFY_HYPOTHESES, nli_check

THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def _default_outputs_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs"


def _metrics_at(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (scores >= threshold).astype(np.int64)
    f1_bin = float(
        f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    )
    return {
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "f1_binary_biased": round(f1_bin, 6),
        "mae": round(float(mean_absolute_error(y_true.astype(np.float64), scores)), 6),
        "predicted_positive_rate": round(float(y_pred.mean()), 6),
        "gold_positive_rate": round(float(y_true.mean()), 6),
    }


def run_threshold_sweep(*, outputs_dir: Path | None = None) -> dict[str, Any]:
    """Sweep nine thresholds on BASIL test; write CSV, JSON bundle, PR + F1 plots."""
    out = (outputs_dir or _default_outputs_dir()).resolve()
    out.mkdir(parents=True, exist_ok=True)

    data_dir = resolve_basil_data_dir()
    frame = load_basil_sentences(data_dir)
    _, test = split_sentence_frame(frame, test_size=0.2, random_state=42)
    test = test.reset_index(drop=True)

    auditor = get_auditor()
    texts = test["sentence_text"].astype(str).tolist()
    y_true = test["label"].to_numpy(dtype=np.int64)

    batch_size = 32
    preds: list[dict] = []
    for start in range(0, len(texts), batch_size):
        preds.extend(auditor.predict_batch(texts[start : start + batch_size]))
    scores = np.array([p["lexical_probability"] for p in preds], dtype=np.float64)

    rows = [_metrics_at(y_true, scores, t) for t in THRESHOLDS]
    best = max(rows, key=lambda r: r["f1_macro"])
    fixed_050 = next(r for r in rows if r["threshold"] == 0.5)

    csv_path = out / "threshold_sweep.csv"
    header = list(rows[0].keys())
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r[k]) for k in header))
    csv_path.write_text("\n".join(lines) + "\n")

    bundle = {
        "f1_optimal": best,
        "fixed_threshold_0_50": fixed_050,
        "sweep_thresholds": THRESHOLDS,
        "artifacts": {
            "threshold_sweep_csv": str(csv_path),
            "precision_recall_curve_png": str(out / "precision_recall_curve.png"),
            "threshold_vs_f1_macro_png": str(out / "threshold_vs_f1_macro.png"),
        },
        "_notes": (
            "Macro-F1 is the selection objective. Re-run after AUDITOR_MODEL_ID changes."
        ),
    }
    choice_path = out / "threshold_choice.json"
    choice_path.write_text(json.dumps(bundle, indent=2))

    plot_errors: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pr_prec, pr_rec, _pr_thresh = precision_recall_curve(y_true, scores)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(pr_rec, pr_prec, color="steelblue", lw=2, label="PR curve")
        for r in rows:
            t = r["threshold"]
            ax.scatter(
                r["recall"],
                r["precision"],
                zorder=5,
                s=60,
                color="tomato" if t == best["threshold"] else "gray",
            )
            ax.annotate(
                f"{t:.2f}",
                (r["recall"], r["precision"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(
            f"Precision-Recall — BASIL test (event holdout)\n"
            f"F1-optimal τ={best['threshold']:.2f}  "
            f"(P={best['precision']:.3f}, R={best['recall']:.3f}, F1={best['f1_macro']:.3f})"
        )
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        pr_path = out / "precision_recall_curve.png"
        fig.savefig(pr_path, dpi=150)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        xs = [r["threshold"] for r in rows]
        ys = [r["f1_macro"] for r in rows]
        ax2.plot(xs, ys, marker="o", color="steelblue", lw=2)
        ax2.axvline(
            best["threshold"],
            color="tomato",
            ls="--",
            label=f"F1-best ({best['threshold']:.2f})",
        )
        ax2.set_xlabel("Score threshold")
        ax2.set_ylabel("Macro-F1")
        ax2.set_title("Threshold sweep — lexical probability")
        ax2.legend()
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        tv_path = out / "threshold_vs_f1_macro.png"
        fig2.savefig(tv_path, dpi=150)
        plt.close(fig2)
    except ImportError as e:
        plot_errors.append(str(e))

    return {
        "demo_title": "Threshold sweep",
        "demo_readout": (
            f"F1-optimal τ={best['threshold']:.2f} macro_f1={best['f1_macro']:.3f} "
            f"(n={len(test)}). Wrote {csv_path.name}, threshold_choice.json, plots."
        ),
        "sentence_count": len(test),
        "auditor_model_id": auditor.model_id,
        "f1_optimal": best,
        "fixed_threshold_0_50": fixed_050,
        "sweep_rows": rows,
        "artifacts": bundle["artifacts"],
        "plot_errors": plot_errors,
    }


def _ts_to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _truncate(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def run_audit_export(*, outputs_dir: Path | None = None, sample_rows: int = 20) -> dict[str, Any]:
    """Export MCP audit SQLite → CSV + Markdown sample (same as ``export_audit_log.py``)."""
    out = (outputs_dir or _default_outputs_dir()).resolve()
    out.mkdir(parents=True, exist_ok=True)
    db_path = audit_db_path()

    if not db_path.exists():
        return {
            "demo_title": "Audit export",
            "demo_readout": f"No database at {db_path}",
            "error": "audit_db_not_found",
            "db_path": str(db_path),
        }

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "SELECT id, ts, tool_name, arguments_json, result_json "
            "FROM tool_calls ORDER BY ts ASC"
        )
        all_rows = cur.fetchall()
    finally:
        conn.close()

    if not all_rows:
        return {
            "demo_title": "Audit export",
            "demo_readout": "Audit DB exists but has zero rows.",
            "db_path": str(db_path),
            "total_calls": 0,
        }

    csv_path = out / "audit_log_export.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "timestamp_utc", "tool_name", "arguments", "result"]
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "timestamp_utc": _ts_to_iso(row["ts"]),
                    "tool_name": row["tool_name"],
                    "arguments": row["arguments_json"],
                    "result": row["result_json"],
                }
            )

    tool_counts: dict[str, int] = {}
    for row in all_rows:
        tool_counts[row["tool_name"]] = tool_counts.get(row["tool_name"], 0) + 1

    sample = list(all_rows)[-sample_rows:]
    md_lines = [
        f"# MCP Audit Log — Sample ({sample_rows} most recent calls)",
        "",
        f"**Database**: `{db_path}`  ",
        f"**Total logged calls**: {len(all_rows)}  ",
        f"**Export date**: {_ts_to_iso(time.time())}",
        "",
        "## Per-tool call counts",
        "",
        "| Tool | Calls |",
        "|------|------:|",
    ]
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        md_lines.append(f"| `{tool}` | {count} |")

    md_lines += [
        "",
        f"## Last {sample_rows} calls",
        "",
        "| # | Timestamp (UTC) | Tool | Key Arguments | Result (excerpt) |",
        "|---|-----------------|------|---------------|-----------------|",
    ]
    for i, row in enumerate(sample, 1):
        ts = _ts_to_iso(row["ts"])
        try:
            args = json.loads(row["arguments_json"])
        except Exception:
            args = {}
        try:
            result = json.loads(row["result_json"])
        except Exception:
            result = {}
        arg_display = ""
        for key in ("sentence", "query", "text", "tool_name", "outlet"):
            if key in args:
                val = str(args[key])
                arg_display = f"`{key}`: {_truncate(val, 60)}"
                break
        if not arg_display and args:
            first_key = next(iter(args))
            arg_display = f"`{first_key}`: {_truncate(str(args[first_key]), 60)}"
        result_display = ""
        if isinstance(result, dict):
            for rkey in (
                "bias_score",
                "label",
                "entailment_probability",
                "claim_follows_from_premise",
                "f1_macro",
                "accuracy",
            ):
                if rkey in result:
                    result_display = f"`{rkey}={result[rkey]}`"
                    break
        if not result_display:
            result_display = _truncate(str(result), 60)
        tool_name = row["tool_name"]
        md_lines.append(
            f"| {i} | {ts} | `{tool_name}` | {arg_display} | {result_display} |"
        )

    md_path = out / "audit_log_sample.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "demo_title": "Audit export",
        "demo_readout": f"Exported {len(all_rows)} calls to {csv_path.name} + {md_path.name}",
        "db_path": str(db_path),
        "total_calls": len(all_rows),
        "tool_counts": tool_counts,
        "csv_path": str(csv_path),
        "markdown_sample_path": str(md_path),
        "time_range_utc": {
            "first": _ts_to_iso(all_rows[0]["ts"]),
            "last": _ts_to_iso(all_rows[-1]["ts"]),
        },
    }


def _load_auditor_threshold(outputs_dir: Path) -> float:
    env_val = os.environ.get("AUDITOR_THRESHOLD", "").strip()
    if env_val:
        return float(env_val)
    choice_path = outputs_dir / "threshold_choice.json"
    if choice_path.exists():
        data = json.loads(choice_path.read_text())
        return float(data["f1_optimal"]["threshold"])
    return 0.5


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    return {
        "sentence_count": int(len(y_true)),
        "gold_positive_rate": round(float(y_true.mean()), 6),
        "predicted_positive_rate": round(float(y_pred.mean()), 6),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "mae": round(float(mean_absolute_error(y_true.astype(float), scores)), 6),
    }


def _outlet_breakdown(df: Any, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    breakdown: dict[str, Any] = {}
    for outlet, grp in df.groupby("source"):
        idx = grp.index.tolist()
        gt = y_true[idx]
        gp = y_pred[idx]
        gs = scores[idx]
        breakdown[str(outlet)] = _metrics_dict(gt, gp, gs)
    return breakdown


def _nli_grounded(sentence: str) -> tuple[bool, float]:
    max_entailment = 0.0
    any_supported = False
    for hyp in BIAS_VERIFY_HYPOTHESES:
        result = nli_check(sentence, hyp)
        log_call(
            "nli_check",
            {"premise": sentence[:500], "hypothesis": hyp[:500]},
            result,
            invocation=basil_dual_agent_nli_invocation(hyp),
        )
        p_ent = float(result.get("entailment_probability", 0.0))
        max_entailment = max(max_entailment, p_ent)
        if result.get("claim_follows_from_premise", False):
            any_supported = True
    return any_supported, max_entailment


def run_full_basil_auditor_nli_eval(
    *,
    outputs_dir: Path | None = None,
    max_sentences: int | None = None,
) -> dict[str, Any]:
    """Auditor-only + NLI verifier pass on BASIL test split; writes JSON + CSV under outputs/."""
    out = (outputs_dir or _default_outputs_dir()).resolve()
    out.mkdir(parents=True, exist_ok=True)

    data_dir = resolve_basil_data_dir()
    threshold = _load_auditor_threshold(out)
    auditor = get_auditor()

    frame = load_basil_sentences(data_dir)
    _, test = split_sentence_frame(frame, test_size=0.2, random_state=42)
    test = test.reset_index(drop=True)
    if max_sentences is not None and max_sentences > 0:
        test = test.iloc[:max_sentences].reset_index(drop=True)

    texts = test["sentence_text"].astype(str).tolist()
    y_true = test["label"].to_numpy(dtype=np.int64)

    raw_preds: list[dict] = []
    batch_size = 32
    for start in range(0, len(texts), batch_size):
        raw_preds.extend(auditor.predict_batch(texts[start : start + batch_size]))
    scores = np.array([p["lexical_probability"] for p in raw_preds], dtype=np.float64)
    auditor_pred = (scores >= threshold).astype(np.int64)

    for pos in range(len(texts)):
        log_call(
            "roberta_auditor_infer",
            {"sentence": texts[pos][:500], "index": pos},
            {
                "lexical_probability": round(float(scores[pos]), 6),
                "predicted_biased": bool(int(auditor_pred[pos])),
                "threshold": threshold,
                "auditor_model_id": auditor.model_id,
            },
            invocation={
                "orchestration": "basil_dual_agent_eval",
                "stage": "roberta_auditor",
            },
        )

    auditor_metrics = _metrics_dict(y_true, auditor_pred, scores)
    auditor_metrics["threshold"] = threshold
    auditor_metrics["auditor_model"] = auditor.model_id
    auditor_metrics["outlet_breakdown"] = _outlet_breakdown(test, y_true, auditor_pred, scores)
    (out / "eval_full_auditor_only.json").write_text(json.dumps(auditor_metrics, indent=2))

    verifier_verdicts: list[str] = []
    final_pred = auditor_pred.copy()
    revised = 0
    for i, (sentence, aud_pred) in enumerate(zip(texts, auditor_pred)):
        if aud_pred == 1:
            grounded, _me = _nli_grounded(sentence)
            if not grounded:
                verifier_verdicts.append("downgraded")
                final_pred[i] = 0
                revised += 1
            else:
                verifier_verdicts.append("confirmed")
        else:
            verifier_verdicts.append("skipped")

    revision_rate = revised / len(texts) if texts else 0.0
    verifier_metrics = _metrics_dict(y_true, final_pred, scores)
    verifier_metrics["threshold"] = threshold
    verifier_metrics["auditor_model"] = auditor.model_id
    verifier_metrics["nli_model"] = os.environ.get("NLI_MODEL_NAME", "").strip()
    verifier_metrics["revision_rate"] = round(revision_rate, 6)
    verifier_metrics["sentences_revised"] = revised
    verifier_metrics["outlet_breakdown"] = _outlet_breakdown(test, y_true, final_pred, scores)
    (out / "eval_full_verifier_corrected.json").write_text(json.dumps(verifier_metrics, indent=2))

    csv_path = out / "eval_full_predictions.csv"
    fieldnames = [
        "event_id",
        "outlet",
        "sentence",
        "gold",
        "auditor_score",
        "auditor_pred",
        "verifier_verdict",
        "final_pred",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pos, (_, row) in enumerate(test.iterrows()):
            writer.writerow(
                {
                    "event_id": row["event_id"],
                    "outlet": row["source"],
                    "sentence": row["sentence_text"],
                    "gold": int(y_true[pos]),
                    "auditor_score": round(float(scores[pos]), 6),
                    "auditor_pred": int(auditor_pred[pos]),
                    "verifier_verdict": verifier_verdicts[pos],
                    "final_pred": int(final_pred[pos]),
                }
            )

    return {
        "demo_title": "BASIL dual-agent eval",
        "demo_readout": (
            f"n={len(test)} auditor F1={auditor_metrics['f1_macro']:.3f} → "
            f"verifier F1={verifier_metrics['f1_macro']:.3f} "
            f"(revised {revised}, τ={threshold}). Files: eval_full_*.json, {csv_path.name}"
        ),
        "sentence_count": len(test),
        "threshold": threshold,
        "auditor_only": {k: auditor_metrics[k] for k in ("f1_macro", "accuracy", "precision", "recall", "mae")},
        "verifier_corrected": {k: verifier_metrics[k] for k in ("f1_macro", "accuracy", "precision", "recall", "mae")},
        "revision_rate": round(revision_rate, 6),
        "sentences_revised": revised,
        "paths": {
            "auditor_json": str(out / "eval_full_auditor_only.json"),
            "verifier_json": str(out / "eval_full_verifier_corrected.json"),
            "predictions_csv": str(csv_path),
        },
    }
