"""Load BASIL article JSON into a sentence-level DataFrame (aligned with the project notebook)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def load_basil_sentences(dataset_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(dataset_dir.glob("*.json")):
        article = json.loads(path.read_text())
        article_id = path.stem
        event_id, source = article_id.split("_", 1)
        for sentence_obj in article["body"]:
            sentence = (sentence_obj.get("sentence") or "").strip()
            annotations = sentence_obj.get("annotations", [])
            bias_types = sorted(
                {
                    annotation.get("bias", "").strip().lower()
                    for annotation in annotations
                    if annotation.get("bias")
                }
            )
            targets = sorted(
                {
                    annotation.get("target", "").strip()
                    for annotation in annotations
                    if annotation.get("target")
                }
            )
            rows.append(
                {
                    "example_id": f"{article_id}::{sentence_obj['sentence-index']}",
                    "event_id": event_id,
                    "article_id": article_id,
                    "source": source.lower(),
                    "date": article.get("date"),
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "main_event": article.get("main-event"),
                    "sentence_index": sentence_obj["sentence-index"],
                    "sentence_text": sentence,
                    "label": int(bool(annotations)),
                    "gold_annotation_count": len(annotations),
                    "gold_bias_types": bias_types,
                    "gold_targets": targets,
                }
            )

    frame = pd.DataFrame(rows)
    frame["sentence_text"] = frame["sentence_text"].fillna("")
    return frame


def split_sentence_frame(
    frame: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(frame, frame["label"], groups=frame["event_id"]))
    train_frame = frame.iloc[train_idx].reset_index(drop=True)
    test_frame = frame.iloc[test_idx].reset_index(drop=True)
    train_frame["split"] = "train"
    test_frame["split"] = "test"
    return train_frame, test_frame
