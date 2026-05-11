#!/usr/bin/env python3
"""Flask demo server for the dual-agent bias detection frontend.

Uses ``mcp_server.bias_surface`` + ``mcp_server.providers`` — same code path as MCP
cloud LLM tools. Model ids come **only** from ``BIAS_LLM_*`` env vars (slots ``A`` / ``B``).

Legacy ``mode`` names (gpt/claude) map to slots: **A** vs **B** — not to fixed vendors.

Endpoints
---------
POST /api/analyze          run cloud auditor + optional verifier on article text
GET  /api/logs             recent audit log entries
GET  /api/metrics/pipeline llm_pipeline_metrics.json
GET  /api/metrics/multi    multi_provider_metrics.json
GET  /api/artifacts        list output files with sizes

Usage
-----
    Configure BIAS_LLM_A_* and BIAS_LLM_B_* then:
    python demo/server.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from mcp_server.auditing import init_db, recent_events
from mcp_server.bias_surface import run_llm_bias_score, run_llm_verify
from mcp_server.providers import resolve_slot_config

app = Flask(__name__)
CORS(app)

OUTPUTS = ROOT / "outputs"
THRESHOLD = float(os.environ.get("AUDITOR_THRESHOLD", "0.4"))

# Legacy API: "gpt" → slot A, "claude" → slot B (configure vendors in env, not here).
SLOT_PRIMARY = os.environ.get("DEMO_LLM_SLOT_A", "A").strip()
SLOT_SECONDARY = os.environ.get("DEMO_LLM_SLOT_B", "B").strip()


def _audit_slot(sentence: str, slot: str) -> dict:
    r = run_llm_bias_score(sentence, slot, internal_audit=True)
    return {
        "bias_score": float(r["bias_score"]),
        "bias_type": str(r["bias_type"]),
        "biased_phrases": [],
        "reasoning": str(r.get("reasoning", "")),
        "_llm_slot": r.get("llm_slot"),
        "_llm_vendor": r.get("llm_vendor"),
        "_llm_model": r.get("llm_model"),
    }


def _verify_slot(sentence: str, auditor_reasoning: str, slot: str) -> dict:
    r = run_llm_verify(sentence, auditor_reasoning, slot, internal_audit=True)
    return {
        "verified": bool(r["verified"]),
        "confidence": float(r["confidence"]),
        "biased_phrases": [],
        "reasoning": str(r.get("reasoning", "")),
        "_llm_slot": r.get("llm_slot"),
        "_llm_vendor": r.get("llm_vendor"),
        "_llm_model": r.get("llm_model"),
    }


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(])', text.strip())
    return [p.strip() for p in parts if p.strip()]


VALID_MODES = {"gpt-only", "claude-only", "gpt-claude", "claude-gpt"}


def _run_pipeline(sent: str, mode: str) -> dict:
    if mode in ("gpt-only", "gpt-claude"):
        aud_slot = SLOT_PRIMARY
    else:
        aud_slot = SLOT_SECONDARY
    try:
        aud = _audit_slot(sent, aud_slot)
    except Exception as e:
        aud = {
            "bias_score": 0.0,
            "bias_type": "none",
            "biased_phrases": [],
            "reasoning": str(e),
        }
    time.sleep(0.05)

    aud_pred = float(aud.get("bias_score", 0.0)) >= THRESHOLD
    ver = None
    if aud_pred and mode in ("gpt-claude", "claude-gpt"):
        vslot = SLOT_SECONDARY if mode == "gpt-claude" else SLOT_PRIMARY
        try:
            ver = _verify_slot(sent, aud.get("reasoning", ""), vslot)
        except Exception as e:
            ver = {"verified": False, "confidence": 0.0, "biased_phrases": [], "reasoning": str(e)}
        time.sleep(0.05)

    if not aud_pred:
        status = "clean"
    elif ver is None:
        status = "auditor_flagged"
    elif ver.get("verified"):
        status = "confirmed"
    else:
        status = "downgraded"

    return {"sentence": sent, "status": status, "auditor": aud, "verifier": ver}


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Body JSON: text (required), mode: gpt-only | claude-only | gpt-claude | claude-gpt."""
    body = request.get_json(force=True)
    text = (body.get("text") or "").strip()
    mode = (body.get("mode") or "gpt-claude").strip().lower()

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if mode not in VALID_MODES:
        return jsonify({"error": f"mode must be one of {sorted(VALID_MODES)}"}), 400

    sentences = _split_sentences(text)
    if not sentences:
        return jsonify({"error": "Could not parse sentences"}), 400

    results = []
    for sent in sentences:
        if len(sent) < 10:
            results.append({"sentence": sent, "status": "skipped", "auditor": None, "verifier": None})
            continue
        results.append(_run_pipeline(sent, mode))

    return jsonify(
        {
            "mode": mode,
            "sentence_count": len(results),
            "threshold": THRESHOLD,
            "slot_primary": SLOT_PRIMARY,
            "slot_secondary": SLOT_SECONDARY,
            "results": results,
        }
    )


@app.route("/api/logs")
def logs():
    limit = int(request.args.get("limit", 50))
    events = recent_events(limit=limit)
    return jsonify(events)


@app.route("/api/metrics/pipeline")
def pipeline_metrics():
    path = OUTPUTS / "llm_pipeline_metrics.json"
    if not path.exists():
        return jsonify({"error": "Run run_llm_pipeline.py first"}), 404
    return jsonify(json.loads(path.read_text()))


@app.route("/api/metrics/multi")
def multi_metrics():
    path = OUTPUTS / "multi_provider_metrics.json"
    if not path.exists():
        return jsonify({"error": "Run run_multi_provider.py first"}), 404
    return jsonify(json.loads(path.read_text()))


@app.route("/api/artifacts")
def artifacts():
    files = []
    for p in sorted(OUTPUTS.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            files.append(
                {"name": p.name, "size_kb": round(p.stat().st_size / 1024, 1), "ext": p.suffix}
            )
    return jsonify(files)


def _slot_label(slot: str) -> str:
    try:
        c = resolve_slot_config(slot)
        return f"{c.vendor}:{c.model}"
    except Exception as e:
        return f"(not configured: {e})"


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "threshold": THRESHOLD,
            "slot_primary": SLOT_PRIMARY,
            "slot_secondary": SLOT_SECONDARY,
            "slot_primary_model": _slot_label(SLOT_PRIMARY),
            "slot_secondary_model": _slot_label(SLOT_SECONDARY),
            "modes": sorted(VALID_MODES),
            "default_mode": "gpt-claude",
            "note": "gpt/claude in mode names map to DEMO_LLM_SLOT_A / B (default A then B).",
        }
    )


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5001))
    print(f"Threshold: {THRESHOLD}")
    print(f"Slot {SLOT_PRIMARY} (gpt / first): {_slot_label(SLOT_PRIMARY)}")
    print(f"Slot {SLOT_SECONDARY} (claude / second): {_slot_label(SLOT_SECONDARY)}")
    print(f"Starting server on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
