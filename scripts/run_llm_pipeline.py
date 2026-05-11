#!/usr/bin/env python3
"""Dual-agent cloud LLM pipeline — **no hard-coded model IDs or vendors**.

Uses ``mcp_server.bias_surface.run_llm_bias_score`` and ``run_llm_verify`` (same as MCP
tools ``llm_bias_score`` / ``llm_verify``). Each inference logs to SQLite via ``internal_audit=True``.

Env
---
    BIAS_LLM_AUDITOR_SLOT     slot id for scorer (default ``A``) → ``BIAS_LLM_<SLOT>_VENDOR``, ``_MODEL``, keys
    BIAS_LLM_VERIFIER_SLOT    if set (e.g. ``B``), run verifier on auditor-positive sentences
    BASIL_DATA_DIR, SAMPLE_SIZE, AUDITOR_THRESHOLD, VERIFY_THRESHOLD

Outputs
-------
outputs/llm_pipeline_predictions.csv
outputs/llm_pipeline_metrics.json
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

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

from mcp_server.auditing import init_db
from mcp_server.basil_dataset import load_basil_sentences, split_sentence_frame
from mcp_server.basil_paths import resolve_basil_data_dir
from mcp_server.bias_surface import run_llm_bias_score, run_llm_verify
from mcp_server.providers import resolve_slot_config

OUTPUTS = ROOT / "outputs"
SAMPLE_PER_OUTLET = int(os.environ.get("SAMPLE_SIZE", "67"))
VERIFY_THRESHOLD = float(os.environ.get("VERIFY_THRESHOLD", "0.5"))
AUDITOR_SLOT = os.environ.get("BIAS_LLM_AUDITOR_SLOT", "A").strip()
VERIFIER_SLOT = os.environ.get("BIAS_LLM_VERIFIER_SLOT", "").strip()


def _load_threshold() -> float:
    env_val = os.environ.get("AUDITOR_THRESHOLD", "").strip()
    if env_val:
        return float(env_val)
    choice = OUTPUTS / "threshold_choice.json"
    if choice.exists():
        t = float(json.loads(choice.read_text())["f1_optimal"]["threshold"])
        print(f"Threshold {t} loaded from threshold_choice.json")
        return t
    return 0.5


def _metrics(y_true, y_pred, label: str) -> dict:
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    return {
        "label": label,
        "sentence_count": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1_binary": round(float(f1_score(y_true, y_pred, average="binary", zero_division=0)), 6),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "positive_rate": round(float(y_pred.mean()), 6),
    }


def main() -> None:
    try:
        aud_cfg = resolve_slot_config(AUDITOR_SLOT)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    ver_cfg = None
    if VERIFIER_SLOT:
        try:
            ver_cfg = resolve_slot_config(VERIFIER_SLOT)
        except ValueError as e:
            print(f"ERROR (verifier slot): {e}", file=sys.stderr)
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

    if SAMPLE_PER_OUTLET > 0:
        parts = []
        for outlet, grp in test.groupby("source"):
            n = min(SAMPLE_PER_OUTLET, len(grp))
            parts.append(grp.sample(n=n, random_state=42))
        sample = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        sample = test

    aud_label = f"{aud_cfg.vendor}:{aud_cfg.model}"
    ver_label = f"{ver_cfg.vendor}:{ver_cfg.model}" if ver_cfg else "(no verifier)"
    print(f"Auditor slot {AUDITOR_SLOT} : {aud_label}")
    print(f"Verifier slot {VERIFIER_SLOT or '—'} : {ver_label}")
    print(f"Sample    : {len(sample)} sentences")
    print(f"Threshold : {threshold}\n")

    rows: list[dict] = []
    auditor_only_preds: list[int] = []
    final_preds: list[int] = []
    gold_labels: list[int] = []
    verifier_calls = 0
    verifier_downgrades = 0
    auditor_positive_count = 0

    for i, (_, row) in enumerate(sample.iterrows()):
        sentence = str(row["sentence_text"])
        gold = int(row["label"])
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sample)}")

        try:
            aud = run_llm_bias_score(sentence, AUDITOR_SLOT, internal_audit=True)
        except Exception as e:
            print(f"  [warn] Auditor error at {i}: {e}")
            aud = {"bias_score": 0.5, "bias_type": "none", "reasoning": str(e)}
        time.sleep(0.1)

        aud_score = float(aud.get("bias_score", 0.5))
        aud_pred = int(aud_score >= threshold)
        if aud_pred == 1:
            auditor_positive_count += 1

        ver_verdict = "skipped"
        ver_verified = None
        ver_confidence = None
        ver_reasoning = ""
        final_pred = aud_pred

        if aud_pred == 1 and ver_cfg is not None:
            verifier_calls += 1
            try:
                ver = run_llm_verify(
                    sentence,
                    str(aud.get("reasoning", "")),
                    VERIFIER_SLOT,
                    internal_audit=True,
                )
                ver_verified = bool(ver.get("verified", False))
                ver_confidence = float(ver.get("confidence", 0.0))
                ver_reasoning = str(ver.get("reasoning", ""))
                if not ver_verified or ver_confidence < VERIFY_THRESHOLD:
                    final_pred = 0
                    ver_verdict = "downgraded"
                    verifier_downgrades += 1
                else:
                    ver_verdict = "confirmed"
            except Exception as e:
                print(f"  [warn] Verifier error at {i}: {e}")
                ver_verdict = "error"
            time.sleep(0.1)

        auditor_only_preds.append(aud_pred)
        final_preds.append(final_pred)
        gold_labels.append(gold)

        rows.append(
            {
                "event_id": row["event_id"],
                "outlet": row["source"],
                "sentence": sentence,
                "gold": gold,
                "auditor_score": round(aud_score, 6),
                "auditor_pred": aud_pred,
                "auditor_type": aud["bias_type"],
                "auditor_reasoning": aud["reasoning"],
                "verifier_verdict": ver_verdict,
                "verifier_verified": str(ver_verified),
                "verifier_confidence": str(ver_confidence),
                "verifier_reasoning": ver_reasoning,
                "final_pred": final_pred,
                "correct": int(final_pred == gold),
            }
        )

    print("\nComputing metrics ...")
    pipeline_desc = "auditor+verifier" if ver_cfg else "auditor_only"
    aud_metrics = _metrics(gold_labels, auditor_only_preds, f"auditor ({aud_label})")
    final_metrics = _metrics(gold_labels, final_preds, f"pipeline ({pipeline_desc})")
    revision_rate = verifier_downgrades / len(rows) if rows else 0.0

    outlet_breakdown: dict[str, dict] = {}
    result_df = pd.DataFrame(rows)
    for outlet, grp in result_df.groupby("outlet"):
        outlet_breakdown[str(outlet)] = {
            "sentence_count": len(grp),
            "auditor_f1": round(
                float(f1_score(grp["gold"], grp["auditor_pred"], average="macro", zero_division=0)), 4
            ),
            "auditor_f1_binary": round(
                float(f1_score(grp["gold"], grp["auditor_pred"], average="binary", zero_division=0)), 4
            ),
            "pipeline_f1": round(
                float(f1_score(grp["gold"], grp["final_pred"], average="macro", zero_division=0)), 4
            ),
            "pipeline_f1_binary": round(
                float(f1_score(grp["gold"], grp["final_pred"], average="binary", zero_division=0)), 4
            ),
            "verifier_calls": int((grp["verifier_verdict"] != "skipped").sum()),
            "downgrades": int((grp["verifier_verdict"] == "downgraded").sum()),
        }

    full_metrics = {
        "auditor_slot": AUDITOR_SLOT,
        "verifier_slot": VERIFIER_SLOT or None,
        "auditor_model": aud_label,
        "verifier_model": ver_label,
        "threshold": threshold,
        "verify_threshold": VERIFY_THRESHOLD,
        "sample_size": len(rows),
        "auditor_only": aud_metrics,
        "pipeline": final_metrics,
        "verifier_calls": verifier_calls,
        "auditor_positive_count": auditor_positive_count,
        "verifier_downgrades": verifier_downgrades,
        "revision_rate": round(revision_rate, 6),
        "outlet_breakdown": outlet_breakdown,
    }

    csv_path = OUTPUTS / "llm_pipeline_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")

    metrics_path = OUTPUTS / "llm_pipeline_metrics.json"
    metrics_path.write_text(json.dumps(full_metrics, indent=2))
    print(f"Saved {metrics_path}")

    print("\n" + "=" * 60)
    print("LLM PIPELINE (bias_surface + SQLite audit per call)")
    print("=" * 60)
    for key in ["accuracy", "precision", "recall", "f1_binary", "f1_macro"]:
        print(f"  {key}: auditor {aud_metrics[key]:.4f}  pipeline {final_metrics[key]:.4f}")
    print(f"\nVerifier downgrades: {verifier_downgrades} / {len(rows)}")


if __name__ == "__main__":
    main()
