#!/usr/bin/env python3
"""Dual-agent LLM bias detection pipeline — configurable model setup.

Supports four pipeline modes via PIPELINE_MODE env var:

    gpt-only      GPT-4o-mini scores every sentence, no verifier
    claude-only   Claude-haiku scores every sentence, no verifier
    gpt-claude    GPT auditor → Claude verifier  (default)
    claude-gpt    Claude auditor → GPT verifier  (flipped)

The two agents are coordinated through the same MCP audit infrastructure,
demonstrating that provider-swapping requires zero tool-code changes.

Outputs
-------
outputs/llm_pipeline_predictions.csv
outputs/llm_pipeline_metrics.json

Usage
-----
    OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... \\
    BASIL_DATA_DIR=... PIPELINE_MODE=gpt-claude python scripts/run_llm_pipeline.py

Env vars
--------
    PIPELINE_MODE       gpt-only | claude-only | gpt-claude | claude-gpt (default: gpt-claude)
    BASIL_DATA_DIR      path to BASIL *.json articles
    SAMPLE_SIZE         sentences per outlet, default 67 (~201 total); 0 = full test split
    AUDITOR_THRESHOLD   decision threshold (falls back to threshold_choice.json)
    VERIFY_THRESHOLD    min confidence for verifier to confirm bias (default 0.5)
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

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)
import numpy as np

from mcp_server.audit import init_db, log_call
from mcp_server.basil_dataset import load_basil_sentences, split_sentence_frame
from mcp_server.basil_paths import resolve_basil_data_dir

import anthropic as anthropic_sdk
import openai as openai_sdk

OUTPUTS = ROOT / "outputs"

GPT_MODEL    = "gpt-4o-mini"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

SAMPLE_PER_OUTLET = int(os.environ.get("SAMPLE_SIZE", "67"))
VERIFY_THRESHOLD  = float(os.environ.get("VERIFY_THRESHOLD", "0.5"))

# Pipeline mode — controls which model is auditor and which is verifier
VALID_MODES = {"gpt-only", "claude-only", "gpt-claude", "claude-gpt"}
PIPELINE_MODE = os.environ.get("PIPELINE_MODE", "gpt-claude").strip().lower()
if PIPELINE_MODE not in VALID_MODES:
    print(f"ERROR: PIPELINE_MODE must be one of {VALID_MODES}", file=sys.stderr)
    sys.exit(1)

# ------------------------------------------------------------------ #
# Prompts                                                             #
# ------------------------------------------------------------------ #

AUDITOR_PROMPT = """You are a media bias classifier. Score the following news sentence for media bias.
Return JSON only — no other text:
{
  "bias_score": <float 0.0 to 1.0>,
  "bias_type": <"lexical", "informational", or "none">,
  "reasoning": <one sentence, max 30 words>
}"""

VERIFIER_PROMPT = """You are a media bias verification agent. An AI auditor has flagged the following \
news sentence as potentially biased. Your job is to verify whether the assessment is correct.

Consider:
- Does the sentence use loaded or emotionally charged language?
- Does it present information in a one-sided or misleading way?
- Could a reasonable, neutral reader consider this biased reporting?

Return JSON only — no other text:
{
  "verified": <true or false>,
  "confidence": <float 0.0 to 1.0>,
  "reasoning": <one sentence, max 30 words>
}"""


# ------------------------------------------------------------------ #
# Threshold                                                           #
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
# JSON parsing helpers                                                #
# ------------------------------------------------------------------ #

import re as _re

def _extract_json(raw: str) -> dict:
    raw = _re.sub(r"```(?:json)?\s*", "", raw).strip()
    m = _re.search(r"\{.*\}", raw, _re.DOTALL)
    if m:
        raw = m.group(0)
    return json.loads(raw)


# ------------------------------------------------------------------ #
# Model callers — one per provider                                    #
# ------------------------------------------------------------------ #

def _call_gpt(sentence: str, prompt: str, role: str) -> dict:
    """Call GPT-4o-mini with the given system prompt."""
    client = openai_sdk.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=GPT_MODEL,
        max_tokens=300,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user",   "content": f"Sentence: {sentence}"},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    data = _extract_json(raw)
    result = {k: (float(v) if isinstance(v, (int, float)) else v)
              for k, v in data.items()}
    log_call(f"{role}_gpt", {"sentence": sentence[:500]}, result)
    return result


def _call_claude(sentence: str, prompt: str, role: str,
                 extra_context: str = "") -> dict:
    """Call Claude-haiku with the given system prompt."""
    client = anthropic_sdk.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    user_content = f"Sentence: {sentence}"
    if extra_context:
        user_content += f"\n\n{extra_context}"
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        system=prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = resp.content[0].text.strip() if resp.content else "{}"
    data = _extract_json(raw)
    result = {k: (float(v) if isinstance(v, (int, float)) else v)
              for k, v in data.items()}
    log_call(f"{role}_claude", {"sentence": sentence[:500]}, result)
    return result


# ------------------------------------------------------------------ #
# Pipeline-mode dispatch                                              #
# ------------------------------------------------------------------ #

def _resolve_pipeline():
    """Return (auditor_fn, verifier_fn_or_None, auditor_label, verifier_label)."""
    if PIPELINE_MODE == "gpt-only":
        audit_fn  = lambda s: _call_gpt(s, AUDITOR_PROMPT, "audit")
        verify_fn = None
        return audit_fn, verify_fn, GPT_MODEL, "none"

    if PIPELINE_MODE == "claude-only":
        audit_fn  = lambda s: _call_claude(s, AUDITOR_PROMPT, "audit")
        verify_fn = None
        return audit_fn, verify_fn, CLAUDE_MODEL, "none"

    if PIPELINE_MODE == "gpt-claude":
        audit_fn  = lambda s:       _call_gpt(s, AUDITOR_PROMPT, "audit")
        verify_fn = lambda s, note: _call_claude(s, VERIFIER_PROMPT, "verify",
                                                  f"Auditor note: {note}")
        return audit_fn, verify_fn, GPT_MODEL, CLAUDE_MODEL

    if PIPELINE_MODE == "claude-gpt":
        audit_fn  = lambda s:       _call_claude(s, AUDITOR_PROMPT, "audit")
        verify_fn = lambda s, note: _call_gpt(s, VERIFIER_PROMPT + \
                                               f"\n\nAuditor note: {note}", "verify")
        return audit_fn, verify_fn, CLAUDE_MODEL, GPT_MODEL

    raise ValueError(f"Unknown PIPELINE_MODE: {PIPELINE_MODE}")


# ------------------------------------------------------------------ #
# Metrics                                                             #
# ------------------------------------------------------------------ #

def _metrics(y_true, y_pred, label: str) -> dict:
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    return {
        "label":         label,
        "sentence_count": int(len(y_true)),
        "accuracy":      round(float(accuracy_score(y_true, y_pred)), 6),
        "precision":     round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall":        round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1_macro":      round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "positive_rate": round(float(y_pred.mean()), 6),
    }


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    # Validate required API keys based on mode
    needed = []
    if PIPELINE_MODE in ("gpt-only", "gpt-claude", "claude-gpt"):
        needed.append("OPENAI_API_KEY")
    if PIPELINE_MODE in ("claude-only", "gpt-claude", "claude-gpt"):
        needed.append("ANTHROPIC_API_KEY")
    missing = [k for k in needed if not os.environ.get(k)]
    if missing:
        print(f"ERROR: missing env vars for mode '{PIPELINE_MODE}': {', '.join(missing)}",
              file=sys.stderr)
        sys.exit(1)

    init_db()
    OUTPUTS.mkdir(exist_ok=True)

    try:
        data_dir = resolve_basil_data_dir()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    threshold = _load_threshold()
    audit_fn, verify_fn, auditor_label, verifier_label = _resolve_pipeline()

    import pandas as pd
    print(f"Loading BASIL from {data_dir} ...")
    frame = load_basil_sentences(data_dir)
    _, test = split_sentence_frame(frame, test_size=0.2, random_state=42)
    test = test.reset_index(drop=True)

    # Stratified sample (or full split if SAMPLE_SIZE=0)
    if SAMPLE_PER_OUTLET > 0:
        parts = []
        for outlet, grp in test.groupby("source"):
            n = min(SAMPLE_PER_OUTLET, len(grp))
            parts.append(grp.sample(n=n, random_state=42))
        sample = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        sample = test

    print(f"Mode      : {PIPELINE_MODE}")
    print(f"Auditor   : {auditor_label}")
    print(f"Verifier  : {verifier_label}")
    print(f"Sample    : {len(sample)} sentences — {dict(sample['source'].value_counts())}")
    print(f"Threshold : {threshold}\n")

    rows: list[dict] = []
    auditor_only_preds: list[int] = []
    final_preds:        list[int] = []
    gold_labels:        list[int] = []
    verifier_calls = 0
    verifier_downgrades = 0

    for i, (_, row) in enumerate(sample.iterrows()):
        sentence = str(row["sentence_text"])
        gold     = int(row["label"])

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sample)}")

        # --- Agent 1: Auditor ---
        try:
            aud = audit_fn(sentence)
        except Exception as e:
            print(f"  [warn] Auditor error at {i}: {e}")
            aud = {"bias_score": 0.5, "bias_type": "none", "reasoning": str(e)}
        time.sleep(0.1)

        aud_score = float(aud.get("bias_score", 0.5))
        aud_pred  = int(aud_score >= threshold)

        # --- Agent 2: Verifier (only if auditor says biased and verifier exists) ---
        ver_verdict    = "skipped"
        ver_verified   = None
        ver_confidence = None
        ver_reasoning  = ""
        final_pred     = aud_pred

        if aud_pred == 1 and verify_fn is not None:
            verifier_calls += 1
            try:
                ver = verify_fn(sentence, aud.get("reasoning", ""))
                ver_verified   = bool(ver.get("verified", False))
                ver_confidence = float(ver.get("confidence", 0.0))
                ver_reasoning  = str(ver.get("reasoning", ""))

                if not ver_verified or ver_confidence < VERIFY_THRESHOLD:
                    final_pred  = 0
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

        rows.append({
            "event_id":         row["event_id"],
            "outlet":           row["source"],
            "sentence":         sentence,
            "gold":             gold,
            "auditor_score":    round(aud_score, 6),
            "auditor_pred":     aud_pred,
            "auditor_type":     aud["bias_type"],
            "auditor_reasoning":aud["reasoning"],
            "verifier_verdict": ver_verdict,
            "verifier_verified":str(ver_verified),
            "verifier_confidence": str(ver_confidence),
            "verifier_reasoning":  ver_reasoning,
            "final_pred":       final_pred,
            "correct":          int(final_pred == gold),
        })

    # ------------------------------------------------------------------ #
    # Metrics                                                             #
    # ------------------------------------------------------------------ #
    print("\nComputing metrics ...")

    pipeline_label = PIPELINE_MODE if verify_fn else f"{PIPELINE_MODE} (no verifier)"
    aud_metrics   = _metrics(gold_labels, auditor_only_preds, f"auditor_only ({auditor_label})")
    final_metrics = _metrics(gold_labels, final_preds,        f"pipeline ({pipeline_label})")

    revision_rate = verifier_downgrades / len(rows)

    outlet_breakdown: dict[str, dict] = {}
    import pandas as pd
    result_df = pd.DataFrame(rows)
    for outlet, grp in result_df.groupby("outlet"):
        outlet_breakdown[str(outlet)] = {
            "sentence_count":    len(grp),
            "auditor_f1":        round(float(f1_score(grp["gold"], grp["auditor_pred"],
                                                       average="macro", zero_division=0)), 4),
            "pipeline_f1":       round(float(f1_score(grp["gold"], grp["final_pred"],
                                                       average="macro", zero_division=0)), 4),
            "verifier_calls":    int((grp["verifier_verdict"] != "skipped").sum()),
            "downgrades":        int((grp["verifier_verdict"] == "downgraded").sum()),
        }

    full_metrics = {
        "pipeline_mode":    PIPELINE_MODE,
        "auditor_model":    auditor_label,
        "verifier_model":   verifier_label,
        "threshold":        threshold,
        "verify_threshold": VERIFY_THRESHOLD,
        "sample_size":      len(rows),
        "auditor_only":     aud_metrics,
        "pipeline":         final_metrics,
        "verifier_calls":   verifier_calls,
        "verifier_downgrades": verifier_downgrades,
        "revision_rate":    round(revision_rate, 6),
        "outlet_breakdown": outlet_breakdown,
    }

    # ------------------------------------------------------------------ #
    # Save outputs                                                        #
    # ------------------------------------------------------------------ #
    csv_path = OUTPUTS / "llm_pipeline_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}  ({len(rows)} rows)")

    metrics_path = OUTPUTS / "llm_pipeline_metrics.json"
    metrics_path.write_text(json.dumps(full_metrics, indent=2))
    print(f"Saved {metrics_path}")

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("DUAL-AGENT LLM PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Mode     : {PIPELINE_MODE}")
    print(f"Auditor  : {auditor_label}")
    print(f"Verifier : {verifier_label}")
    print(f"Sample   : {len(rows)} sentences (stratified by outlet)")
    print(f"Threshold: {threshold}")
    print()
    print(f"{'Metric':<20} {'Auditor-only':>14} {'Pipeline':>14}")
    print("-" * 50)
    for key in ["accuracy", "precision", "recall", "f1_macro"]:
        a = aud_metrics[key]
        p = final_metrics[key]
        diff = p - a
        sign = "+" if diff >= 0 else ""
        print(f"{key:<20} {a:>14.4f} {p:>14.4f}  ({sign}{diff:.4f})")
    print()
    print(f"Verifier calls     : {verifier_calls} / {len(rows)} "
          f"({verifier_calls/len(rows):.1%} of sentences audited positive)")
    print(f"Verifier downgrades: {verifier_downgrades} "
          f"({revision_rate:.1%} of all sentences revised)")
    print()
    print("Per-outlet breakdown (F1 macro):")
    for outlet, m in sorted(outlet_breakdown.items()):
        print(f"  {outlet:<8}  auditor={m['auditor_f1']:.4f}  "
              f"pipeline={m['pipeline_f1']:.4f}  "
              f"downgrades={m['downgrades']}/{m['verifier_calls']}")

    # Example where verifier changed the call
    downgrades = [r for r in rows if r["verifier_verdict"] == "downgraded"]
    if downgrades:
        ex = downgrades[0]
        print(f"\nExample downgrade (outlet={ex['outlet']}, gold={'biased' if ex['gold'] else 'not biased'}):")
        print(f"  Sentence   : {ex['sentence'][:110]}{'...' if len(ex['sentence']) > 110 else ''}")
        print(f"  GPT scored : {ex['auditor_score']:.3f} → biased  [{ex['auditor_reasoning']}]")
        print(f"  Claude said: NOT verified (conf={ex['verifier_confidence']})  [{ex['verifier_reasoning']}]")
        print(f"  Final pred : not biased  |  Gold: {'biased' if ex['gold'] else 'not biased'}")

    print(f"\nOutputs saved to outputs/")


if __name__ == "__main__":
    main()
