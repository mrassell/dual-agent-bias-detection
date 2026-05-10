#!/usr/bin/env python3
"""Flask demo server for the dual-agent bias detection frontend.

Endpoints
---------
POST /api/analyze          run GPT auditor + Claude verifier on article text
GET  /api/logs             recent MCP audit log entries
GET  /api/metrics/pipeline llm_pipeline_metrics.json
GET  /api/metrics/multi    multi_provider_metrics.json
GET  /api/artifacts        list output files with sizes

Usage
-----
    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... python demo/server.py
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

from mcp_server.audit import init_db, log_call, recent_events

import anthropic as anthropic_sdk
import openai as openai_sdk

app = Flask(__name__)
CORS(app)

OUTPUTS   = ROOT / "outputs"
THRESHOLD = float(os.environ.get("AUDITOR_THRESHOLD", "0.4"))
GPT_MODEL    = "gpt-4o-mini"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# ------------------------------------------------------------------ #
# Prompts                                                             #
# ------------------------------------------------------------------ #

AUDITOR_PROMPT = """You are a media bias classifier. Analyze the following news sentence for bias.
Return JSON only — no other text:
{
  "bias_score": <float 0.0 to 1.0>,
  "bias_type": <"lexical", "informational", or "none">,
  "biased_phrases": <array of exact substrings from the sentence that carry bias, empty if none>,
  "reasoning": <one sentence, max 30 words>
}"""

VERIFIER_PROMPT = """You are a media bias verification agent. An AI auditor flagged this news sentence as potentially biased.
Verify whether the bias is real — consider whether loaded language comes from the reporter or is just quoted speech.
Return JSON only — no other text:
{
  "verified": <true or false>,
  "confidence": <float 0.0 to 1.0>,
  "biased_phrases": <array of exact substrings that are genuinely biased, empty if not verified>,
  "reasoning": <one sentence, max 30 words>
}"""


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _extract_json(raw: str) -> dict:
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    return json.loads(raw)


def _split_sentences(text: str) -> list[str]:
    """Simple but robust sentence splitter."""
    # Split on . ! ? followed by whitespace+capital or end of string
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(])', text.strip())
    # Further split very long chunks and filter empties
    sentences = []
    for p in parts:
        p = p.strip()
        if p:
            sentences.append(p)
    return sentences


def _audit_gpt(sentence: str) -> dict:
    client = openai_sdk.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=GPT_MODEL, max_tokens=300,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": AUDITOR_PROMPT},
                  {"role": "user",   "content": f"Sentence: {sentence}"}],
    )
    data = _extract_json(resp.choices[0].message.content.strip())
    result = {"bias_score": float(data.get("bias_score", 0.0)),
              "bias_type":  str(data.get("bias_type", "none")),
              "biased_phrases": list(data.get("biased_phrases", [])),
              "reasoning":  str(data.get("reasoning", ""))}
    log_call("demo_audit_gpt", {"sentence": sentence[:300]}, result)
    return result


def _audit_claude(sentence: str) -> dict:
    client = anthropic_sdk.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=300, system=AUDITOR_PROMPT,
        messages=[{"role": "user", "content": f"Sentence: {sentence}"}],
    )
    raw = resp.content[0].text.strip() if resp.content else "{}"
    data = _extract_json(raw)
    result = {"bias_score": float(data.get("bias_score", 0.0)),
              "bias_type":  str(data.get("bias_type", "none")),
              "biased_phrases": list(data.get("biased_phrases", [])),
              "reasoning":  str(data.get("reasoning", ""))}
    log_call("demo_audit_claude", {"sentence": sentence[:300]}, result)
    return result


def _verify_claude(sentence: str, auditor_reasoning: str) -> dict:
    client = anthropic_sdk.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=300, system=VERIFIER_PROMPT,
        messages=[{"role": "user", "content":
                   f"Sentence: {sentence}\nAuditor note: {auditor_reasoning}"}],
    )
    raw = resp.content[0].text.strip() if resp.content else "{}"
    data = _extract_json(raw)
    result = {"verified":   bool(data.get("verified", False)),
              "confidence": float(data.get("confidence", 0.0)),
              "biased_phrases": list(data.get("biased_phrases", [])),
              "reasoning":  str(data.get("reasoning", ""))}
    log_call("demo_verify_claude", {"sentence": sentence[:300]}, result)
    return result


def _verify_gpt(sentence: str, auditor_reasoning: str) -> dict:
    client = openai_sdk.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    system = VERIFIER_PROMPT + f"\n\nAuditor note: {auditor_reasoning}"
    resp = client.chat.completions.create(
        model=GPT_MODEL, max_tokens=300,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": f"Sentence: {sentence}"}],
    )
    data = _extract_json(resp.choices[0].message.content.strip())
    result = {"verified":   bool(data.get("verified", False)),
              "confidence": float(data.get("confidence", 0.0)),
              "biased_phrases": list(data.get("biased_phrases", [])),
              "reasoning":  str(data.get("reasoning", ""))}
    log_call("demo_verify_gpt", {"sentence": sentence[:300]}, result)
    return result


# ------------------------------------------------------------------ #
# Routes                                                              #
# ------------------------------------------------------------------ #

VALID_MODES = {"gpt-only", "claude-only", "gpt-claude", "claude-gpt"}

def _run_pipeline(sent: str, mode: str) -> dict:
    """Run the chosen pipeline mode on a single sentence."""
    # --- Auditor ---
    try:
        if mode in ("gpt-only", "gpt-claude"):
            aud = _audit_gpt(sent)
        else:
            aud = _audit_claude(sent)
    except Exception as e:
        aud = {"bias_score": 0.0, "bias_type": "none",
               "biased_phrases": [], "reasoning": str(e)}
    time.sleep(0.05)

    aud_pred = float(aud.get("bias_score", 0.0)) >= THRESHOLD

    # --- Verifier (only on positives, only in dual-agent modes) ---
    ver = None
    if aud_pred and mode in ("gpt-claude", "claude-gpt"):
        try:
            if mode == "gpt-claude":
                ver = _verify_claude(sent, aud.get("reasoning", ""))
            else:
                ver = _verify_gpt(sent, aud.get("reasoning", ""))
        except Exception as e:
            ver = {"verified": False, "confidence": 0.0,
                   "biased_phrases": [], "reasoning": str(e)}
        time.sleep(0.05)

    # Status
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
    """Run the chosen pipeline mode on article text.

    Body JSON:
        text  : article text (required)
        mode  : gpt-only | claude-only | gpt-claude | claude-gpt  (default: gpt-claude)
    """
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
            results.append({"sentence": sent, "status": "skipped",
                             "auditor": None, "verifier": None})
            continue
        results.append(_run_pipeline(sent, mode))

    return jsonify({
        "mode":           mode,
        "sentence_count": len(results),
        "threshold":      THRESHOLD,
        "results":        results,
    })


@app.route("/api/logs")
def logs():
    """Return the 50 most recent MCP audit log entries."""
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
            files.append({
                "name": p.name,
                "size_kb": round(p.stat().st_size / 1024, 1),
                "ext": p.suffix,
            })
    return jsonify(files)


@app.route("/api/health")
def health():
    return jsonify({
        "status":        "ok",
        "threshold":     THRESHOLD,
        "gpt_model":     GPT_MODEL,
        "claude_model":  CLAUDE_MODEL,
        "modes":         sorted(VALID_MODES),
        "default_mode":  "gpt-claude",
    })


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    missing = [k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if not os.environ.get(k)]
    if missing:
        print(f"ERROR: set env vars: {', '.join(missing)}")
        sys.exit(1)
    init_db()
    print(f"Auditor : {GPT_MODEL}")
    print(f"Verifier: {CLAUDE_MODEL}")
    print(f"Threshold: {THRESHOLD}")
    print("Starting server on http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
