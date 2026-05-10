#!/usr/bin/env python3
"""Multi-provider auditor experiment — **no hard-coded vendors or models**.

Both scorers call ``mcp_server.bias_surface.run_llm_bias_score`` (same code path as MCP
tool ``llm_bias_score``). Each call logs via ``internal_audit=True`` to the same
SQLite schema as MCP tools.

Configure two env-backed slots (default slot ids ``A`` and ``B``):

    BIAS_LLM_A_VENDOR, BIAS_LLM_A_MODEL, (+ API key)
    BIAS_LLM_B_VENDOR, BIAS_LLM_B_MODEL, (+ API key)

Optional: ``MULTI_PROVIDER_SLOT_A``, ``MULTI_PROVIDER_SLOT_B`` to use other slot ids
(e.g. ``AUDITOR`` if you defined ``BIAS_LLM_AUDITOR_*``).

Outputs
-------
outputs/multi_provider_predictions.csv
outputs/multi_provider_metrics.json

Usage
-----
    BASIL_DATA_DIR=... python scripts/run_multi_provider.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

from mcp_server.auditing import init_db
from mcp_server.basil_dataset import load_basil_sentences, split_sentence_frame
from mcp_server.basil_paths import resolve_basil_data_dir
from mcp_server.bias_surface import run_llm_bias_score
from mcp_server.providers import resolve_slot_config

OUTPUTS = ROOT / "outputs"
SAMPLE_PER_OUTLET = int(os.environ.get("SAMPLE_SIZE", "67"))
SLOT_A = os.environ.get("MULTI_PROVIDER_SLOT_A", "A").strip()
SLOT_B = os.environ.get("MULTI_PROVIDER_SLOT_B", "B").strip()


def _load_threshold() -> float:
    env_val = os.environ.get("AUDITOR_THRESHOLD", "").strip()
    if env_val:
        return float(env_val)
    choice = OUTPUTS / "threshold_choice.json"
    if choice.exists():
        data = json.loads(choice.read_text())
        t = float(data["f1_optimal"]["threshold"])
        print(f"Threshold {t} loaded from threshold_choice.json")
        return t
    return 0.5


def _compute_metrics(
    rows: list[dict],
    threshold: float,
    label_a: str,
    label_b: str,
    meta_a: dict,
    meta_b: dict,
) -> dict:
    sa = [r["score_a"] for r in rows]
    sb = [r["score_b"] for r in rows]
    pa = [r["pred_a"] for r in rows]
    pb = [r["pred_b"] for r in rows]
    outlets = [r["outlet"] for r in rows]

    agreement_rate = sum(x == y for x, y in zip(pa, pb)) / len(rows)
    kappa = float(cohen_kappa_score(pa, pb))
    r, p = pearsonr(sa, sb)

    outlet_breakdown: dict[str, dict] = {}
    for outlet in sorted(set(outlets)):
        idx = [i for i, o in enumerate(outlets) if o == outlet]
        gp = [pa[i] for i in idx]
        cp = [pb[i] for i in idx]
        disagree = sum(x != y for x, y in zip(gp, cp))
        outlet_breakdown[outlet] = {
            "sentence_count": len(idx),
            "agreement_rate": round(1 - disagree / len(idx), 4),
            "disagreement_count": disagree,
            "slot_a_positive_rate": round(sum(gp) / len(gp), 4),
            "slot_b_positive_rate": round(sum(cp) / len(cp), 4),
        }

    return {
        "slot_a_id": meta_a["slot"],
        "slot_a_vendor": meta_a["vendor"],
        "slot_a_model": meta_a["model"],
        "slot_b_id": meta_b["slot"],
        "slot_b_vendor": meta_b["vendor"],
        "slot_b_model": meta_b["model"],
        "threshold": threshold,
        "sample_size": len(rows),
        "agreement_rate": round(agreement_rate, 6),
        "disagreement_rate": round(1 - agreement_rate, 6),
        "disagreement_count": int(sum(x != y for x, y in zip(pa, pb))),
        "cohen_kappa": round(kappa, 6),
        "pearson_r": round(float(r), 6),
        "pearson_p_value": round(float(p), 6),
        "slot_a_positive_rate": round(sum(pa) / len(pa), 6),
        "slot_b_positive_rate": round(sum(pb) / len(pb), 6),
        "outlet_breakdown": outlet_breakdown,
        "column_labels": {"score_a": label_a, "score_b": label_b},
    }


def main() -> None:
    try:
        cfg_a = resolve_slot_config(SLOT_A)
        cfg_b = resolve_slot_config(SLOT_B)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Define BIAS_LLM_<SLOT>_VENDOR and BIAS_LLM_<SLOT>_MODEL for both slots.", file=sys.stderr)
        sys.exit(1)

    init_db()
    OUTPUTS.mkdir(exist_ok=True)

    try:
        data_dir = resolve_basil_data_dir()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    threshold = _load_threshold()

    import pandas as pd

    print(f"Loading BASIL from {data_dir} ...")
    frame = load_basil_sentences(data_dir)
    _, test = split_sentence_frame(frame, test_size=0.2, random_state=42)
    test = test.reset_index(drop=True)

    parts = []
    for outlet, grp in test.groupby("source"):
        n = min(SAMPLE_PER_OUTLET, len(grp))
        parts.append(grp.sample(n=n, random_state=42))
    sample = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

    la = f"{cfg_a.vendor}:{cfg_a.model}"
    lb = f"{cfg_b.vendor}:{cfg_b.model}"
    print(f"Sample    : {len(sample)} sentences — {dict(sample['source'].value_counts())}")
    print(f"Threshold : {threshold}")
    print(f"Slot {cfg_a.slot} : {la}")
    print(f"Slot {cfg_b.slot} : {lb}\n")

    rows: list[dict] = []

    for i, (_, row) in enumerate(sample.iterrows()):
        sentence = str(row["sentence_text"])
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sample)}")

        try:
            ra = run_llm_bias_score(sentence, SLOT_A, internal_audit=True)
        except Exception as e:
            print(f"  [warn] slot {SLOT_A} error at {i}: {e}")
            ra = {"bias_score": 0.5, "bias_type": "none", "reasoning": str(e)}
        time.sleep(0.1)

        try:
            rb = run_llm_bias_score(sentence, SLOT_B, internal_audit=True)
        except Exception as e:
            print(f"  [warn] slot {SLOT_B} error at {i}: {e}")
            rb = {"bias_score": 0.5, "bias_type": "none", "reasoning": str(e)}
        time.sleep(0.1)

        sa = float(ra["bias_score"])
        sb = float(rb["bias_score"])
        pa = int(sa >= threshold)
        pb = int(sb >= threshold)

        rows.append(
            {
                "event_id": row["event_id"],
                "outlet": row["source"],
                "sentence": sentence,
                "gold": int(row["label"]),
                "score_a": round(sa, 6),
                "score_b": round(sb, 6),
                "pred_a": pa,
                "pred_b": pb,
                "agreement": int(pa == pb),
                "type_a": ra["bias_type"],
                "type_b": rb["bias_type"],
                "reasoning_a": ra["reasoning"],
                "reasoning_b": rb["reasoning"],
            }
        )

    meta_a = {"slot": cfg_a.slot, "vendor": cfg_a.vendor, "model": cfg_a.model}
    meta_b = {"slot": cfg_b.slot, "vendor": cfg_b.vendor, "model": cfg_b.model}
    metrics = _compute_metrics(rows, threshold, la, lb, meta_a, meta_b)

    csv_path = OUTPUTS / "multi_provider_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}  ({len(rows)} rows)")

    metrics_path = OUTPUTS / "multi_provider_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved {metrics_path}")

    print("\n" + "=" * 60)
    print("MULTI-PROVIDER SUMMARY (capability surface: run_llm_bias_score)")
    print("=" * 60)
    print(f"Slot A ({cfg_a.slot}): {la}")
    print(f"Slot B ({cfg_b.slot}): {lb}")
    print(f"Agreement rate : {metrics['agreement_rate']:.4f}")
    print(f"Cohen's kappa  : {metrics['cohen_kappa']:.4f}")
    print(f"Pearson r      : {metrics['pearson_r']:.4f}")
    print("\nSwapping providers is env-only (BIAS_LLM_*); tool code path is unchanged.")


if __name__ == "__main__":
    main()
