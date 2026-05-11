"""NLI verifier for MCP ``nli_check`` (model id from ``NLI_MODEL_NAME`` only).

When ``hypothesis`` is empty/whitespace, ``nli_check`` runs a **fixed template
pack** (lexical bias, informational bias, neutral reference) and aggregates
entailment scores — verifier path for batch jobs that need an automatic bias read.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# Explicit hypotheses for automatic bias verification (premise = sentence).
EXPLICIT_BIAS_NLI_TEMPLATES: list[tuple[str, str]] = [
    (
        "lexical",
        "The sentence uses loaded, slanted, or emotionally charged wording typical of biased news coverage.",
    ),
    (
        "informational",
        "The sentence omits important context or frames the story in a one-sided or misleading way.",
    ),
    (
        "neutral_reference",
        "The sentence is balanced, factual reporting without media bias or partisan framing.",
    ),
]

# Shorter strings for scripts that loop pairwise (``run_full_eval``, etc.)
BIAS_VERIFY_HYPOTHESES: list[str] = [t[1] for t in EXPLICIT_BIAS_NLI_TEMPLATES if t[0] != "neutral_reference"]

_nli_tokenizer = None
_nli_model = None
_nli_device = None


def _require_nli_model_id() -> str:
    name = os.environ.get("NLI_MODEL_NAME", "").strip()
    if not name:
        raise ValueError(
            "Set NLI_MODEL_NAME to a Hugging Face sequence-classification NLI checkpoint "
            "(no default is configured in code)."
        )
    return name


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


def _load_nli() -> tuple[Any, Any, Any]:
    global _nli_tokenizer, _nli_model, _nli_device
    if _nli_tokenizer is not None and _nli_model is not None and _nli_device is not None:
        return _nli_tokenizer, _nli_model, _nli_device

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "torch and transformers are required for NLI. "
            "Install them with: pip install torch transformers"
        ) from e

    name = _require_nli_model_id()
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


def _forward_pair(premise: str, hypothesis: str) -> dict[str, Any]:
    tok, model, device = _load_nli()
    enc = tok(
        premise,
        hypothesis,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    import torch
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
    return {
        "entailment_probability": round(p_ent, 4),
        "contradiction_probability": round(p_con, 4),
        "neutral_probability": round(p_neu, 4),
        "predicted_label": label,
        "claim_follows_from_premise": follows,
    }


def nli_check_explicit_bias_pack(premise: str) -> dict[str, Any]:
    """Run BART-MNLI on each explicit template; aggregate a bias-supported signal."""
    per: dict[str, dict[str, Any]] = {}
    bias_entails: list[float] = []
    neutral_entail = 0.0
    for key, hyp in EXPLICIT_BIAS_NLI_TEMPLATES:
        out = _forward_pair(premise, hyp)
        per[key] = out
        if key == "neutral_reference":
            neutral_entail = float(out["entailment_probability"])
        else:
            bias_entails.append(float(out["entailment_probability"]))

    max_bias_entail = max(bias_entails) if bias_entails else 0.0
    bias_cons = [
        float(per[k]["contradiction_probability"])
        for k in ("lexical", "informational")
        if k in per
    ]
    max_bias_con = max(bias_cons) if bias_cons else 0.0
    # Bias supported if a bias template entails more strongly than neutral reference.
    margin = max_bias_entail - neutral_entail
    bias_supported = margin > 0.02
    verdict = (
        "bias templates favored over neutral (NLI pack)"
        if bias_supported
        else "neutral template comparable or stronger (NLI pack)"
    )

    model_id = _require_nli_model_id()
    return {
        "demo_title": "NLI: explicit bias hypotheses",
        "demo_readout": verdict,
        "verdict": verdict,
        "rule": "Support bias pack if max(entail_bias) > entail(neutral_reference) + 0.02.",
        "claim_follows_from_premise": bias_supported,
        "entailment_probability": round(max_bias_entail, 4),
        "contradiction_probability": round(max_bias_con, 4),
        "neutral_probability": round(neutral_entail, 4),
        "predicted_label": "entailment" if bias_supported else "neutral",
        "model": model_id,
        "explicit_bias_margin": round(margin, 4),
        "per_hypothesis": per,
    }


def nli_check(premise: str, hypothesis: str) -> dict[str, Any]:
    """Premise vs hypothesis NLI, or **automatic explicit bias pack** if hypothesis is empty."""
    if not hypothesis.strip():
        return nli_check_explicit_bias_pack(premise)

    out = _forward_pair(premise, hypothesis)
    p_ent = out["entailment_probability"]
    p_con = out["contradiction_probability"]
    p_neu = out["neutral_probability"]
    follows = bool(out["claim_follows_from_premise"])
    verdict = "supported (p_entail > p_contrad)" if follows else "not supported"
    model_id = _require_nli_model_id()

    return {
        "demo_title": "NLI: claim vs source",
        "demo_readout": verdict,
        "verdict": verdict,
        "rule": "Support if P(entailment) > P(contradiction).",
        "claim_follows_from_premise": follows,
        "entailment_probability": p_ent,
        "contradiction_probability": p_con,
        "neutral_probability": p_neu,
        "predicted_label": out["predicted_label"],
        "model": model_id,
    }


def reset_nli_for_tests() -> None:
    global _nli_tokenizer, _nli_model, _nli_device
    _nli_tokenizer = None
    _nli_model = None
    _nli_device = None
