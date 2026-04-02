"""MNLI: premise vs hypothesis → supported or not (entailment vs contradiction)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_NLI_MODEL_ID = "typeform/distilbert-base-uncased-mnli"

_nli_tokenizer = None
_nli_model = None
_nli_device: torch.device | None = None


def _softmax1d(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


def _label_probs(probs: np.ndarray, id2label: dict[int, str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for i in range(len(probs)):
        name = str(id2label[i]).upper()
        if "ENTAIL" in name:
            out["entailment"] = float(probs[i])
        elif "CONTRAD" in name:
            out["contradiction"] = float(probs[i])
        elif "NEUTRAL" in name:
            out["neutral"] = float(probs[i])
    return out


def _load_nli() -> tuple[Any, Any, torch.device]:
    global _nli_tokenizer, _nli_model, _nli_device
    if _nli_tokenizer is not None and _nli_model is not None and _nli_device is not None:
        return _nli_tokenizer, _nli_model, _nli_device

    name = os.environ.get("NLI_MODEL_NAME", DEFAULT_NLI_MODEL_ID).strip() or DEFAULT_NLI_MODEL_ID
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)
    _nli_tokenizer = tok
    _nli_model = model
    _nli_device = device
    return tok, model, device


def nli_check(premise: str, hypothesis: str) -> dict[str, Any]:
    if not hypothesis.strip():
        return {
            "demo_title": "NLI: claim vs source",
            "demo_readout": "No hypothesis text — nothing to check.",
            "verdict": "N/A",
            "rule": "Support if P(entailment) > P(contradiction).",
            "claim_follows_from_premise": True,
            "entailment_probability": 0.5,
            "contradiction_probability": 0.0,
            "neutral_probability": 0.5,
            "predicted_label": "neutral",
            "model": os.environ.get("NLI_MODEL_NAME", DEFAULT_NLI_MODEL_ID),
        }

    tok, model, device = _load_nli()
    enc = tok(
        premise,
        hypothesis,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits.float().cpu().numpy()[0]
    probs = _softmax1d(logits)
    raw = model.config.id2label
    id2label: dict[int, str] = {int(k): str(v) for k, v in raw.items()}
    lp = _label_probs(probs, id2label)
    p_ent = lp.get("entailment", 0.0)
    p_con = lp.get("contradiction", 0.0)
    p_neu = lp.get("neutral", max(0.0, 1.0 - p_ent - p_con))

    pred_idx = int(np.argmax(probs))
    pred_raw = id2label[pred_idx].upper()
    if "ENTAIL" in pred_raw:
        label = "entailment"
    elif "CONTRAD" in pred_raw:
        label = "contradiction"
    else:
        label = "neutral"

    follows = p_ent > p_con
    verdict = "supported (p_entail > p_contrad)" if follows else "not supported"

    return {
        "demo_title": "NLI: claim vs source",
        "demo_readout": verdict,
        "verdict": verdict,
        "rule": "Support if P(entailment) > P(contradiction).",
        "claim_follows_from_premise": follows,
        "entailment_probability": round(p_ent, 4),
        "contradiction_probability": round(p_con, 4),
        "neutral_probability": round(p_neu, 4),
        "predicted_label": label,
        "model": os.environ.get("NLI_MODEL_NAME", DEFAULT_NLI_MODEL_ID),
    }


def reset_nli_for_tests() -> None:
    global _nli_tokenizer, _nli_model, _nli_device
    _nli_tokenizer = None
    _nli_model = None
    _nli_device = None
