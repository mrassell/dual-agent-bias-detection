#!/usr/bin/env python3
"""Sweep decision thresholds on the BASIL test split and pick the F1-optimal one.

CLI wrapper around ``mcp_server.bias_surface.run_sweep_auditor_thresholds``
(same as MCP tool ``sweep_auditor_thresholds``).

Outputs
-------
outputs/threshold_sweep.csv
outputs/precision_recall_curve.png
outputs/threshold_vs_f1_macro.png
outputs/threshold_choice.json

Usage
-----
    python scripts/threshold_sweep.py

Env vars
--------
    BASIL_DATA_DIR   path to BASIL *.json articles (auto-detected if not set)
    AUDITOR_MODEL_ID   required — sentence classifier checkpoint (HF id or path)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from mcp_server.auditing import init_db
from mcp_server.bias_surface import run_sweep_auditor_thresholds


def main() -> None:
    if not os.environ.get("AUDITOR_MODEL_ID", "").strip():
        print("ERROR: set AUDITOR_MODEL_ID", file=sys.stderr)
        sys.exit(1)
    init_db()
    try:
        result = run_sweep_auditor_thresholds()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    best = result["f1_optimal"]
    rows = result["sweep_rows"]
    print(f"\nThreshold sweep results (n={result['sentence_count']}):")
    hdr = f"{'Thr':>6}  {'P':>8}  {'R':>8}  {'F1_macro':>10}  {'F1_bin+':>9}  {'MAE':>8}"
    print(hdr)
    for r in rows:
        marker = " <-- best macro-F1" if r["threshold"] == best["threshold"] else ""
        print(
            f"{r['threshold']:>6.2f}  {r['precision']:>8.4f}  {r['recall']:>8.4f}  "
            f"{r['f1_macro']:>10.4f}  {r['f1_binary_biased']:>9.4f}  {r['mae']:>8.4f}{marker}"
        )

    print("\n" + result.get("demo_readout", ""))
    if result.get("plot_errors"):
        print("Plot warnings:", result["plot_errors"])
    print(f"\nF1-optimal threshold: {best['threshold']:.2f}")
    print(f"  export AUDITOR_THRESHOLD={best['threshold']}")


if __name__ == "__main__":
    main()
