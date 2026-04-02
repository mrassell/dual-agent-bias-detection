"""RoBERTa fine-tuned on BABE for sentence-level lexical bias (binary logits)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_AUDITOR_MODEL = "mediabiasgroup/roberta-babe-ft"
MAX_LENGTH = 256


def _softmax2d(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


class LexicalBiasAuditor:
    """Binary classifier: neutral vs lexical bias (BABE checkpoint)."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = (model_id or os.environ.get("AUDITOR_MODEL_ID") or DEFAULT_AUDITOR_MODEL).strip()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.model.eval()
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device)
        if self.model.config.num_labels != 2:
            raise ValueError(f"Expected 2 labels, got {self.model.config.num_labels}")

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
            p_lex = float(probs[i, 1])
            p_neu = float(probs[i, 0])
            pred = 1 if p_lex >= p_neu else 0
            conf = float(max(p_lex, p_neu))
            out.append(
                {
                    "lexical_probability": p_lex,
                    "neutral_probability": p_neu,
                    "predicted_lexical_bias": pred == 1,
                    "confidence": conf,
                }
            )
        return out

    def predict_one(self, text: str) -> dict[str, Any]:
        return self.predict_batch([text])[0]


_auditor: LexicalBiasAuditor | None = None


def get_auditor() -> LexicalBiasAuditor:
    global _auditor
    if _auditor is None:
        _auditor = LexicalBiasAuditor()
    return _auditor


def reset_auditor() -> None:
    global _auditor
    _auditor = None
