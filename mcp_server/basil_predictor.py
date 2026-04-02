"""Optional DistilBERT checkpoint trained on BASIL (notebook-style layout)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LENGTH = 256


def _softmax2d(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def resolve_basil_model_dir() -> Path:
    """Directory with config.json; BASIL_MODEL_PATH or ~/basil_workspace/distilbert_basil_model."""
    raw = os.environ.get("BASIL_MODEL_PATH", "").strip()
    if raw:
        root = Path(raw).expanduser().resolve()
    else:
        root = (Path.home() / "basil_workspace" / "distilbert_basil_model").resolve()

    if not root.exists():
        raise FileNotFoundError(
            "Fine-tuned BASIL DistilBERT not found. Train with BASIL_Bias_Baseline.ipynb (OUTPUT_DIR) "
            f"or set BASIL_MODEL_PATH to that folder. Expected: {root}"
        )

    if (root / "config.json").is_file():
        return root

    checkpoints = sorted(
        root.glob("checkpoint-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]) if p.name.rsplit("-", 1)[-1].isdigit() else -1,
    )
    for cp in reversed(checkpoints):
        if (cp / "config.json").is_file():
            return cp

    raise FileNotFoundError(
        f"No config.json in {root} or checkpoint-* subfolders. "
        "Save the Trainer output from the notebook, or set BASIL_MODEL_PATH to the checkpoint directory."
    )


def _read_base_model_name(model_dir: Path) -> str:
    cfg_path = model_dir / "config.json"
    data = json.loads(cfg_path.read_text())
    return str(data.get("_name_or_path") or data.get("model_type") or "distilbert-base-uncased")


class BasilPredictor:
    """Binary sentence classifier from a local BASIL-tuned checkpoint."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir or resolve_basil_model_dir()
        self.base_model_name = _read_base_model_name(self.model_dir)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.eval()
        if self.model.config.num_labels != 2:
            raise ValueError(f"Expected num_labels=2, got {self.model.config.num_labels}")

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device)

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        if not texts:
            return []
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits.float().cpu().numpy()
        probs = _softmax2d(logits)
        out: list[dict[str, Any]] = []
        for i in range(len(texts)):
            p_bias = float(probs[i, 1])
            pred_idx = int(np.argmax(probs[i]))
            conf = float(np.max(probs[i]))
            out.append(
                {
                    "bias_probability": p_bias,
                    "predicted_has_bias": pred_idx,
                    "predicted_label": "biased" if pred_idx == 1 else "not_biased",
                    "prediction_confidence": conf,
                }
            )
        return out

    def predict_one(self, text: str) -> dict[str, Any]:
        return self.predict_batch([text])[0]


_predictor: BasilPredictor | None = None


def get_predictor() -> BasilPredictor:
    global _predictor
    if _predictor is None:
        _predictor = BasilPredictor()
    return _predictor


def reset_predictor_for_tests() -> None:
    global _predictor
    _predictor = None
