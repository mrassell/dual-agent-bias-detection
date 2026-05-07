#!/usr/bin/env python3
"""Full BASIL test-set evaluation: auditor-only and verifier-corrected.

Runs the complete test split (no sentence cap) at the F1-optimal threshold.
Produces per-outlet breakdowns and a sentence-level predictions CSV.

Outputs
-------
outputs/eval_full_auditor_only.json
outputs/eval_full_verifier_corrected.json
outputs/eval_full_predictions.csv

Usage
-----
    python scripts/run_full_eval.py

Env vars
--------
    BASIL_DATA_DIR       path to BASIL *.json articles
    AUDITOR_MODEL_ID     fine-tuned checkpoint (default: outputs/roberta-babe-basil-ft)
    AUDITOR_THRESHOLD    decision threshold; falls back to outputs/threshold_choice.json
    NLI_MODEL_NAME       NLI model for verifier (default: facebook/bart-large-mnli)
"""

from __future__ import annotations

import csv
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

# Default to the fine-tuned model if available
_ft_model = str(ROOT / "outputs" / "roberta-babe-basil-ft")
os.environ.setdefault("AUDITOR_MODEL_ID", _ft_model if Path(_ft_model).is_dir() else "mediabiasgroup/roberta-babe-ft")
os.environ.setdefault("NLI_MODEL_NAME", "facebook/bart-large-mnli")

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)

from mcp_server.audit import init_db
from mcp_server.auditor import get_auditor
from mcp_server.basil_dataset import load_basil_sentences, split_sentence_frame
from mcp_server.basil_paths import resolve_basil_data_dir
from mcp_server.nli import nli_check

OUTPUTS = ROOT / "outputs"

# Verifier thresholds (match notebook)
NLI_DOWNGRADE = 0.35   # auditor says biased, NLI support < this → flip
NLI_UPGRADE   = 0.65   # auditor says not_biased, NLI support > this → flip

# Hypotheses used for bias verification
HYPOTHESES = [
    "This sentence uses biased or loaded language.",
    "This sentence presents information in a one-sided or misleading way.",
]


def _load_threshold() -> float:
    """Read from env, fall back to saved threshold_choice.json."""
    env_val = os.environ.get("AUDITOR_THRESHOLD", "").strip()
    if env_val:
        return float(env_val)
    choice_path = OUTPUTS / "threshold_choice.json"
    if choice_path.exists():
        data = json.loads(choice_path.read_text())
        t = float(data["f1_optimal"]["threshold"])
        print(f"Loaded threshold {t} from {choice_path}")
        return t
    print("No threshold found — defaulting to 0.5")
    return 0.5


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
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


def _outlet_breakdown(df, y_true, y_pred, scores) -> dict:
    breakdown = {}
    for outlet, grp in df.groupby("source"):
        idx = grp.index.tolist()
        gt = y_true[idx]
        gp = y_pred[idx]
        gs = scores[idx]
        breakdown[str(outlet)] = _metrics_dict(gt, gp, gs)
    return breakdown


def _nli_grounded(sentence: str) -> tuple[bool, float]:
    """Return (grounded, max_entailment) across both bias hypotheses via MCP nli_check.

    A sentence is considered grounded if AT LEAST ONE hypothesis is supported
    (P(entailment) > P(contradiction)) by the NLI model.
    """
    max_entailment = 0.0
    any_supported = False
    for hyp in HYPOTHESES:
        result = nli_check(sentence, hyp)
        p_ent = result.get("entailment_probability", 0.0)
        max_entailment = max(max_entailment, p_ent)
        if result.get("claim_follows_from_premise", False):
            any_supported = True
    return any_supported, max_entailment


def main() -> None:
    init_db()
    OUTPUTS.mkdir(exist_ok=True)

    try:
        data_dir = resolve_basil_data_dir()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    threshold = _load_threshold()
    auditor = get_auditor()
    print(f"Auditor model : {auditor.model_id}")
    print(f"NLI model     : {os.environ.get('NLI_MODEL_NAME')}")
    print(f"Threshold     : {threshold}")

    print(f"\nLoading BASIL from {data_dir} ...")
    frame = load_basil_sentences(data_dir)
    _, test = split_sentence_frame(frame, test_size=0.2, random_state=42)
    test = test.reset_index(drop=True)
    print(f"Test split    : {len(test)} sentences (gold pos rate: {test['label'].mean():.3f})")

    # ------------------------------------------------------------------ #
    # 1. Run auditor on every sentence (one pass, scores cached)          #
    # ------------------------------------------------------------------ #
    print("\nRunning auditor ...")
    texts = test["sentence_text"].astype(str).tolist()
    y_true = test["label"].to_numpy(dtype=np.int64)

    raw_preds: list[dict] = []
    batch_size = 32
    for start in range(0, len(texts), batch_size):
        raw_preds.extend(auditor.predict_batch(texts[start : start + batch_size]))
        if (start // batch_size + 1) % 20 == 0:
            print(f"  {min(start + batch_size, len(texts))}/{len(texts)}")

    scores = np.array([p["lexical_probability"] for p in raw_preds], dtype=np.float64)
    auditor_pred = (scores >= threshold).astype(np.int64)
    print("Auditor done.")

    # ------------------------------------------------------------------ #
    # 2. Auditor-only metrics                                             #
    # ------------------------------------------------------------------ #
    auditor_metrics = _metrics_dict(y_true, auditor_pred, scores)
    auditor_metrics["threshold"] = threshold
    auditor_metrics["auditor_model"] = auditor.model_id
    auditor_metrics["outlet_breakdown"] = _outlet_breakdown(test, y_true, auditor_pred, scores)

    path = OUTPUTS / "eval_full_auditor_only.json"
    path.write_text(json.dumps(auditor_metrics, indent=2))
    print(f"\nAuditor-only  → {path.name}")
    print(f"  accuracy={auditor_metrics['accuracy']:.4f}  "
          f"precision={auditor_metrics['precision']:.4f}  "
          f"recall={auditor_metrics['recall']:.4f}  "
          f"f1={auditor_metrics['f1_macro']:.4f}")

    # ------------------------------------------------------------------ #
    # 3. Verifier-corrected pass                                          #
    # ------------------------------------------------------------------ #
    print(f"\nRunning NLI verifier on {len(texts)} sentences ...")
    print(f"  (downgrade if support < {NLI_DOWNGRADE}, upgrade if support > {NLI_UPGRADE})")

    verifier_verdicts: list[str] = []
    final_pred = auditor_pred.copy()
    revised = 0

    for i, (sentence, aud_pred, aud_score) in enumerate(zip(texts, auditor_pred, scores)):
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(texts)}")

        # Only verify auditor-positive sentences; downgrade if NLI finds no support.
        # The verifier acts as a grounding check, not a second classifier.
        # Downgrade if the sentence fails BOTH bias hypotheses (neither has entailment > contradiction).
        if aud_pred == 1:
            grounded, max_ent = _nli_grounded(sentence)
            if not grounded:
                verdict = "downgraded"
                final_pred[i] = 0
                revised += 1
            else:
                verdict = "confirmed"
        else:
            verdict = "skipped"

        verifier_verdicts.append(verdict)

    revision_rate = revised / len(texts)
    print(f"Verifier done. Revised {revised}/{len(texts)} ({revision_rate:.3f} revision rate)")

    # ------------------------------------------------------------------ #
    # 4. Verifier-corrected metrics                                       #
    # ------------------------------------------------------------------ #
    verifier_metrics = _metrics_dict(y_true, final_pred, scores)
    verifier_metrics["threshold"] = threshold
    verifier_metrics["auditor_model"] = auditor.model_id
    verifier_metrics["nli_model"] = os.environ.get("NLI_MODEL_NAME")
    verifier_metrics["revision_rate"] = round(revision_rate, 6)
    verifier_metrics["sentences_revised"] = revised
    verifier_metrics["outlet_breakdown"] = _outlet_breakdown(test, y_true, final_pred, scores)

    path = OUTPUTS / "eval_full_verifier_corrected.json"
    path.write_text(json.dumps(verifier_metrics, indent=2))
    print(f"\nVerifier-corrected → {path.name}")
    print(f"  accuracy={verifier_metrics['accuracy']:.4f}  "
          f"precision={verifier_metrics['precision']:.4f}  "
          f"recall={verifier_metrics['recall']:.4f}  "
          f"f1={verifier_metrics['f1_macro']:.4f}")

    # ------------------------------------------------------------------ #
    # 5. Sentence-level predictions CSV                                   #
    # ------------------------------------------------------------------ #
    csv_path = OUTPUTS / "eval_full_predictions.csv"
    fieldnames = [
        "event_id", "outlet", "sentence", "gold",
        "auditor_score", "auditor_pred", "verifier_verdict", "final_pred",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in test.iterrows():
            writer.writerow({
                "event_id": row["event_id"],
                "outlet": row["source"],
                "sentence": row["sentence_text"],
                "gold": int(y_true[i]),
                "auditor_score": round(float(scores[i]), 6),
                "auditor_pred": int(auditor_pred[i]),
                "verifier_verdict": verifier_verdicts[i],
                "final_pred": int(final_pred[i]),
            })
    print(f"\nPredictions CSV → {csv_path.name}  ({len(test)} rows)")

    # ------------------------------------------------------------------ #
    # 6. Summary comparison                                               #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<22} {'Auditor-only':>14} {'Verifier-corrected':>20}")
    print("-" * 60)
    for metric in ["accuracy", "precision", "recall", "f1_macro", "mae"]:
        a = auditor_metrics[metric]
        v = verifier_metrics[metric]
        diff = v - a
        sign = "+" if diff >= 0 else ""
        print(f"{metric:<22} {a:>14.4f} {v:>14.4f}  ({sign}{diff:.4f})")
    print(f"\n{'Revision rate':<22} {'—':>14} {revision_rate:>14.4f}")

    print("\nPer-outlet breakdown (F1 macro):")
    for outlet in sorted(auditor_metrics["outlet_breakdown"]):
        a_f1 = auditor_metrics["outlet_breakdown"][outlet]["f1_macro"]
        v_f1 = verifier_metrics["outlet_breakdown"][outlet]["f1_macro"]
        print(f"  {outlet:<8} auditor={a_f1:.4f}  verifier={v_f1:.4f}")

    print("\nAll outputs saved to outputs/")


if __name__ == "__main__":
    main()
