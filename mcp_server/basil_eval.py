"""Evaluate the MCP auditor on real BASIL sentences (gold = any span on sentence)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)

from mcp_server.auditor import LexicalBiasAuditor, get_auditor
from mcp_server.basil_dataset import load_basil_sentences, split_sentence_frame


def run_basil_evaluation(
    dataset_dir: Path,
    *,
    max_sentences: int | None = None,
    threshold: float = 0.5,
    random_state: int = 42,
    predictor: LexicalBiasAuditor | None = None,
) -> dict[str, Any]:
    """Grouped holdout by event_id; test split only. Gold = any annotation; pred = score >= threshold."""
    frame = load_basil_sentences(dataset_dir)
    _, test_frame = split_sentence_frame(frame, test_size=0.2, random_state=random_state)
    test_frame = test_frame.reset_index(drop=True)

    if max_sentences is not None and max_sentences > 0:
        test_frame = test_frame.iloc[:max_sentences].reset_index(drop=True)

    texts = test_frame["sentence_text"].astype(str).tolist()
    y_true = test_frame["label"].to_numpy(dtype=np.int64)

    auditor = predictor or get_auditor()
    batch_size = 32
    preds: list[dict[str, Any]] = []
    for start in range(0, len(texts), batch_size):
        preds.extend(auditor.predict_batch(texts[start : start + batch_size]))
    scores = np.array([p["lexical_probability"] for p in preds], dtype=np.float64)
    y_pred = (scores >= threshold).astype(np.int64)

    mae = mean_absolute_error(y_true.astype(np.float64), scores)

    out: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "split": "test",
        "group_shuffle_by": "event_id",
        "sentence_count": len(test_frame),
        "threshold": threshold,
        "auditor_model_id": auditor.model_id,
        "gold_positive_rate": float(y_true.mean()),
        "predicted_positive_rate": float(y_pred.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mae_score_vs_gold_binary": float(mae),
    }

    # Per-outlet metrics and drift buckets
    tmp = test_frame.copy()
    tmp["_score"] = scores
    tmp["_pred"] = y_pred
    outlet_metrics: dict[str, Any] = {}
    bucket_rows: list[dict[str, Any]] = []
    for source, grp in tmp.groupby("source", dropna=False):
        g_true = grp["label"].to_numpy(dtype=np.int64)
        g_pred = grp["_pred"].to_numpy(dtype=np.int64)
        g_score = grp["_score"].to_numpy(dtype=np.float64)
        outlet_metrics[str(source)] = {
            "sentence_count": len(grp),
            "gold_positive_rate": float(g_true.mean()),
            "predicted_positive_rate": float(g_pred.mean()),
            "accuracy": float(accuracy_score(g_true, g_pred)),
            "precision": float(precision_score(g_true, g_pred, zero_division=0)),
            "recall": float(recall_score(g_true, g_pred, zero_division=0)),
            "f1_macro": float(f1_score(g_true, g_pred, average="macro", zero_division=0)),
            "mae_score_vs_gold_binary": float(mean_absolute_error(g_true.astype(np.float64), g_score)),
        }
        bucket_rows.append(
            {
                "period": str(source),
                "bias_tag_counts": {
                    "gold_biased_sentences": int(g_true.sum()),
                    "pred_biased_sentences": int(g_pred.sum()),
                    "mean_lexical_score": float(g_score.mean()),
                },
            }
        )
    out["outlet_metrics"] = outlet_metrics
    out["outlet_buckets_for_drift_tool"] = bucket_rows

    # Sample rows (prefer mistakes)
    examples: list[dict[str, Any]] = []
    wrong_idx = np.where(y_true != y_pred)[0]
    right_idx = np.where(y_true == y_pred)[0]
    pick = list(wrong_idx[:3]) + list(right_idx[: max(0, 3 - len(wrong_idx[:3]))])
    for i in pick[:5]:
        row = test_frame.iloc[int(i)]
        examples.append(
            {
                "example_id": row["example_id"],
                "source": row["source"],
                "gold_label": int(row["label"]),
                "lexical_score": float(scores[int(i)]),
                "predicted_biased": bool(y_pred[int(i)]),
                "sentence_excerpt": (row["sentence_text"][:160] + "…")
                if len(str(row["sentence_text"])) > 160
                else row["sentence_text"],
            }
        )
    out["examples"] = examples

    return out
