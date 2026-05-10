#!/usr/bin/env python3
"""Full BASIL test-set evaluation: auditor-only and verifier-corrected.

Thin CLI around ``mcp_server.bias_surface.run_basil_dual_agent_eval`` —
the same logic is exposed as the MCP tool ``run_basil_dual_agent_eval``.

Outputs
-------
outputs/eval_full_auditor_only.json
outputs/eval_full_verifier_corrected.json
outputs/eval_full_predictions.csv

Usage
-----
    export AUDITOR_MODEL_ID=/path/to/checkpoint   # or HF hub id
    export NLI_MODEL_NAME=facebook/bart-large-mnli
    python scripts/run_full_eval.py

Env vars
--------
    BASIL_DATA_DIR       path to BASIL *.json articles
    AUDITOR_MODEL_ID     required — Hugging Face id or local classifier directory
    NLI_MODEL_NAME       required — NLI sequence-classification checkpoint
    AUDITOR_THRESHOLD    decision threshold; falls back to outputs/threshold_choice.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from mcp_server.auditing import init_db
from mcp_server.bias_surface import run_basil_dual_agent_eval


def main() -> None:
    if not os.environ.get("AUDITOR_MODEL_ID", "").strip():
        print(
            "ERROR: set AUDITOR_MODEL_ID (Hugging Face hub id or local checkpoint).",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.environ.get("NLI_MODEL_NAME", "").strip():
        print(
            "ERROR: set NLI_MODEL_NAME (NLI model id, e.g. facebook/bart-large-mnli).",
            file=sys.stderr,
        )
        sys.exit(1)

    init_db()
    try:
        result = run_basil_dual_agent_eval(max_sentences=0)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(result.get("demo_readout", ""))
    print("\nAuditor-only:", json.dumps(result.get("auditor_only"), indent=2))
    print("\nVerifier-corrected:", json.dumps(result.get("verifier_corrected"), indent=2))
    print("\nPaths:", json.dumps(result.get("paths"), indent=2))
    print("\nAll outputs saved to outputs/")


if __name__ == "__main__":
    main()
