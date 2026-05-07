#!/usr/bin/env python3
"""Sweep decision thresholds on the BASIL test split and pick the F1-optimal one.

Outputs
-------
outputs/threshold_sweep.csv      one row per threshold
outputs/precision_recall_curve.png
outputs/threshold_choice.json    F1-optimal threshold + its metrics

Usage
-----
    python scripts/threshold_sweep.py

Env vars
--------
    BASIL_DATA_DIR   path to BASIL *.json articles (auto-detected if not set)
    AUDITOR_MODEL_ID override the RoBERTa checkpoint
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from mcp_server.audit import init_db
from mcp_server.auditor import get_auditor
from mcp_server.basil_dataset import load_basil_sentences, split_sentence_frame
from mcp_server.basil_paths import resolve_basil_data_dir

THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
OUTPUTS = ROOT / "outputs"


def _metrics_at(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(np.int64)
    return {
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "mae": round(float(mean_absolute_error(y_true.astype(np.float64), scores)), 6),
        "predicted_positive_rate": round(float(y_pred.mean()), 6),
        "gold_positive_rate": round(float(y_true.mean()), 6),
    }


def main() -> None:
    init_db()

    try:
        data_dir = resolve_basil_data_dir()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading BASIL from {data_dir} ...")
    frame = load_basil_sentences(data_dir)
    _, test = split_sentence_frame(frame, test_size=0.2, random_state=42)
    test = test.reset_index(drop=True)
    print(f"Test split: {len(test)} sentences  (gold positive rate: {test['label'].mean():.3f})")

    # Run auditor once and cache raw scores
    print("Running auditor (scores cached for all thresholds) ...")
    auditor = get_auditor()
    texts = test["sentence_text"].astype(str).tolist()
    y_true = test["label"].to_numpy(dtype=np.int64)

    batch_size = 32
    preds: list[dict] = []
    for start in range(0, len(texts), batch_size):
        preds.extend(auditor.predict_batch(texts[start : start + batch_size]))
        if (start // batch_size + 1) % 20 == 0:
            print(f"  {start + batch_size}/{len(texts)}")
    scores = np.array([p["lexical_probability"] for p in preds], dtype=np.float64)
    print("Auditor done.")

    # Sweep thresholds
    rows = [_metrics_at(y_true, scores, t) for t in THRESHOLDS]

    best = max(rows, key=lambda r: r["f1_macro"])
    print(f"\nThreshold sweep results:")
    print(f"{'Threshold':>10}  {'Precision':>10}  {'Recall':>10}  {'F1_macro':>10}  {'MAE':>8}")
    for r in rows:
        marker = " <-- best F1" if r["threshold"] == best["threshold"] else ""
        print(
            f"{r['threshold']:>10.2f}  {r['precision']:>10.4f}  "
            f"{r['recall']:>10.4f}  {r['f1_macro']:>10.4f}  {r['mae']:>8.4f}{marker}"
        )

    OUTPUTS.mkdir(exist_ok=True)

    # Save CSV
    csv_path = OUTPUTS / "threshold_sweep.csv"
    header = list(rows[0].keys())
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r[k]) for k in header))
    csv_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved {csv_path}")

    # Save F1-optimal choice
    choice_path = OUTPUTS / "threshold_choice.json"
    choice_path.write_text(json.dumps({"f1_optimal": best}, indent=2))
    print(f"Saved {choice_path}")

    # Precision-recall curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_true, scores)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(pr_rec, pr_prec, color="steelblue", lw=2, label="PR curve")

        # Mark each swept threshold on the curve
        for r in rows:
            t = r["threshold"]
            ax.scatter(r["recall"], r["precision"], zorder=5, s=60,
                       color="tomato" if t == best["threshold"] else "gray")
            ax.annotate(f"{t:.2f}", (r["recall"], r["precision"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve — RoBERTa-BABE on BASIL test split\n"
                     f"F1-optimal threshold = {best['threshold']:.2f}  "
                     f"(P={best['precision']:.3f}, R={best['recall']:.3f}, F1={best['f1_macro']:.3f})")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        pr_path = OUTPUTS / "precision_recall_curve.png"
        fig.savefig(pr_path, dpi=150)
        plt.close(fig)
        print(f"Saved {pr_path}")
    except ImportError:
        print("matplotlib not installed — skipping PR curve plot.")

    print(f"\nF1-optimal threshold: {best['threshold']:.2f}")
    print(f"  Precision : {best['precision']:.4f}")
    print(f"  Recall    : {best['recall']:.4f}")
    print(f"  F1 macro  : {best['f1_macro']:.4f}")
    print(f"  MAE       : {best['mae']:.4f}")
    print(f"\nTo use this threshold in subsequent runs:")
    print(f"  export AUDITOR_THRESHOLD={best['threshold']}")


if __name__ == "__main__":
    main()
