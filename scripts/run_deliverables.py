#!/usr/bin/env python3
"""Print MCP tool outputs against local BASIL (needs BASIL_DATA_DIR or default path)."""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_server.audit import init_db
from mcp_server.basil_dataset import load_basil_sentences, split_sentence_frame
from mcp_server.basil_paths import resolve_basil_data_dir
from mcp_server.auditor import get_auditor
from mcp_server.server import basil_outlet_drift, detect_bias, evaluate_basil, nli_check


def _slide(title: str, readout: str, payload: dict) -> None:
    print()
    print("=" * 76)
    print(title.upper())
    print("=" * 76)
    print(readout)
    print("-" * 76)
    print(json.dumps(payload, indent=2))


def main() -> None:
    init_db()
    _ = get_auditor()

    try:
        data_dir = resolve_basil_data_dir()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    print()
    print("run_deliverables — BASIL", data_dir)

    frame = load_basil_sentences(data_dir)
    _, test = split_sentence_frame(frame)
    for _, row in test.head(3).iterrows():
        out = detect_bias(str(row["sentence_text"]))
        _slide(
            out["demo_title"],
            out["demo_readout"],
            {k: v for k, v in out.items() if k not in ("demo_title", "demo_readout")},
        )

    pos = test[test["label"] == 1]
    neg = test[test["label"] == 0]
    if len(pos) > 0 and len(neg) > 0:
        claim = "Expert annotators marked at least one media-bias span on this BASIL sentence."
        for name, row in [("gold=biased", pos.iloc[0]), ("gold=neutral", neg.iloc[0])]:
            sent = str(row["sentence_text"])[:800]
            out = nli_check(sent, claim)
            _slide(
                f'{out["demo_title"]} — {name}',
                out["demo_readout"],
                {
                    "premise_excerpt": sent[:140] + ("…" if len(sent) > 140 else ""),
                    "hypothesis": claim,
                    **{k: v for k, v in out.items() if k not in ("demo_title", "demo_readout")},
                },
            )

    cap = int(os.environ.get("BASIL_EVAL_CAP", "500"))
    out = evaluate_basil(max_sentences=cap if cap > 0 else 0, threshold=0.5)
    _slide(
        out.get("demo_title", "BASIL evaluation"),
        out.get("demo_readout", ""),
        {k: v for k, v in out.items() if k not in ("demo_title", "demo_readout")},
    )

    out = basil_outlet_drift()
    _slide(
        out.get("demo_title", "Drift"),
        out.get("demo_readout", ""),
        {k: v for k, v in out.items() if k not in ("demo_title", "demo_readout")},
    )

    print()
    print("done.")


if __name__ == "__main__":
    main()
