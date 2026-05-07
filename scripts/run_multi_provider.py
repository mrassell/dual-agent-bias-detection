#!/usr/bin/env python3
"""Multi-provider auditor experiment.

Runs GPT-4o-mini and Claude-3-5-haiku on the same 200-sentence stratified
BASIL sample and measures how often they disagree on bias scoring.

MCP justification: swapping from one provider to the other required zero
tool-code changes — both log through the same audit infrastructure.

Outputs
-------
outputs/multi_provider_predictions.csv
outputs/multi_provider_metrics.json

Usage
-----
    OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... \\
    BASIL_DATA_DIR=... python scripts/run_multi_provider.py

Env vars
--------
    BASIL_DATA_DIR      path to BASIL *.json articles
    SAMPLE_SIZE         sentences per outlet, default 67 (~200 total)
    AUDITOR_THRESHOLD   decision threshold (falls back to threshold_choice.json)
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

from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

from mcp_server.audit import init_db, log_call
from mcp_server.basil_dataset import load_basil_sentences, split_sentence_frame
from mcp_server.basil_paths import resolve_basil_data_dir

import anthropic as anthropic_sdk
import openai as openai_sdk

OUTPUTS = ROOT / "outputs"
SAMPLE_PER_OUTLET = int(os.environ.get("SAMPLE_SIZE", "67"))

GPT_MODEL    = "gpt-4o-mini"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are a media bias classifier. Score the following news sentence for media bias.
Return JSON only — no other text:
{
  "bias_score": <float 0.0 to 1.0>,
  "bias_type": <"lexical", "informational", or "none">,
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
# LLM scorers                                                         #
# ------------------------------------------------------------------ #

def _parse(raw: str) -> dict:
    """Parse JSON from model response, tolerating markdown fences and preamble."""
    import re
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    # Extract the first {...} block in case there's preamble text
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    data = json.loads(raw)
    return {
        "bias_score": float(data.get("bias_score", 0.5)),
        "bias_type":  str(data.get("bias_type", "none")),
        "reasoning":  str(data.get("reasoning", "")),
    }


def score_gpt(sentence: str) -> dict:
    client = openai_sdk.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=GPT_MODEL,
        max_tokens=300,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Sentence: {sentence}"},
        ],
    )
    result = _parse(resp.choices[0].message.content.strip())
    log_call("score_gpt", {"sentence": sentence[:500]}, result)
    return result


def score_claude(sentence: str) -> dict:
    client = anthropic_sdk.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Sentence: {sentence}"}],
    )
    text = resp.content[0].text.strip() if resp.content else ""
    result = _parse(text)
    log_call("score_claude", {"sentence": sentence[:500]}, result)
    return result


# ------------------------------------------------------------------ #
# Metrics                                                             #
# ------------------------------------------------------------------ #

def _compute_metrics(rows: list[dict], threshold: float) -> dict:
    gpt_scores    = [r["gpt_score"]   for r in rows]
    claude_scores = [r["claude_score"] for r in rows]
    gpt_preds     = [r["gpt_pred"]    for r in rows]
    claude_preds  = [r["claude_pred"] for r in rows]
    outlets       = [r["outlet"]      for r in rows]

    agreement_rate = sum(a == b for a, b in zip(gpt_preds, claude_preds)) / len(rows)
    kappa  = float(cohen_kappa_score(gpt_preds, claude_preds))
    r, p   = pearsonr(gpt_scores, claude_scores)

    outlet_breakdown: dict[str, dict] = {}
    for outlet in sorted(set(outlets)):
        idx = [i for i, o in enumerate(outlets) if o == outlet]
        gp  = [gpt_preds[i]    for i in idx]
        cp  = [claude_preds[i] for i in idx]
        disagree = sum(a != b for a, b in zip(gp, cp))
        outlet_breakdown[outlet] = {
            "sentence_count":        len(idx),
            "agreement_rate":        round(1 - disagree / len(idx), 4),
            "disagreement_count":    disagree,
            "gpt_positive_rate":     round(sum(gp) / len(gp), 4),
            "claude_positive_rate":  round(sum(cp) / len(cp), 4),
        }

    return {
        "gpt_model":           GPT_MODEL,
        "claude_model":        CLAUDE_MODEL,
        "threshold":           threshold,
        "sample_size":         len(rows),
        "agreement_rate":      round(agreement_rate, 6),
        "disagreement_rate":   round(1 - agreement_rate, 6),
        "disagreement_count":  int(sum(a != b for a, b in zip(gpt_preds, claude_preds))),
        "cohen_kappa":         round(kappa, 6),
        "pearson_r":           round(float(r), 6),
        "pearson_p_value":     round(float(p), 6),
        "gpt_positive_rate":   round(sum(gpt_preds)    / len(gpt_preds),    6),
        "claude_positive_rate":round(sum(claude_preds) / len(claude_preds), 6),
        "outlet_breakdown":    outlet_breakdown,
    }


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    missing = [k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if not os.environ.get(k)]
    if missing:
        print(f"ERROR: missing env vars: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    init_db()
    OUTPUTS.mkdir(exist_ok=True)

    try:
        data_dir = resolve_basil_data_dir()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    threshold = _load_threshold()

    import pandas as pd
    print(f"Loading BASIL from {data_dir} ...")
    frame = load_basil_sentences(data_dir)
    _, test = split_sentence_frame(frame, test_size=0.2, random_state=42)
    test = test.reset_index(drop=True)

    parts = []
    for outlet, grp in test.groupby("source"):
        n = min(SAMPLE_PER_OUTLET, len(grp))
        parts.append(grp.sample(n=n, random_state=42))
    sample = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Sample    : {len(sample)} sentences — {dict(sample['source'].value_counts())}")
    print(f"Threshold : {threshold}")
    print(f"Models    : {GPT_MODEL}  vs  {CLAUDE_MODEL}\n")

    rows: list[dict] = []

    for i, (_, row) in enumerate(sample.iterrows()):
        sentence = str(row["sentence_text"])

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sample)}")

        try:
            gpt_result = score_gpt(sentence)
        except Exception as e:
            print(f"  [warn] GPT error at {i}: {e}")
            gpt_result = {"bias_score": 0.5, "bias_type": "none", "reasoning": str(e)}
        time.sleep(0.1)

        try:
            claude_result = score_claude(sentence)
        except Exception as e:
            print(f"  [warn] Claude error at {i}: {e}")
            claude_result = {"bias_score": 0.5, "bias_type": "none", "reasoning": str(e)}
        time.sleep(0.1)

        gpt_score    = gpt_result["bias_score"]
        claude_score = claude_result["bias_score"]
        gpt_pred     = int(gpt_score    >= threshold)
        claude_pred  = int(claude_score >= threshold)

        rows.append({
            "event_id":        row["event_id"],
            "outlet":          row["source"],
            "sentence":        sentence,
            "gold":            int(row["label"]),
            "gpt_score":       round(gpt_score,    6),
            "claude_score":    round(claude_score, 6),
            "gpt_pred":        gpt_pred,
            "claude_pred":     claude_pred,
            "agreement":       int(gpt_pred == claude_pred),
            "gpt_type":        gpt_result["bias_type"],
            "claude_type":     claude_result["bias_type"],
            "gpt_reasoning":   gpt_result["reasoning"],
            "claude_reasoning":claude_result["reasoning"],
        })

    print("\nComputing metrics ...")
    metrics = _compute_metrics(rows, threshold)

    csv_path = OUTPUTS / "multi_provider_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}  ({len(rows)} rows)")

    metrics_path = OUTPUTS / "multi_provider_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved {metrics_path}")

    print("\n" + "=" * 60)
    print("MULTI-PROVIDER SUMMARY")
    print("=" * 60)
    print(f"GPT model    : {GPT_MODEL}")
    print(f"Claude model : {CLAUDE_MODEL}")
    print(f"Sample       : {metrics['sample_size']} sentences (stratified by outlet)")
    print(f"Threshold    : {threshold}")
    print()
    print(f"Agreement rate    : {metrics['agreement_rate']:.4f}")
    print(f"Disagreement rate : {metrics['disagreement_rate']:.4f}  ({metrics['disagreement_count']} sentences)")
    print(f"Cohen's kappa     : {metrics['cohen_kappa']:.4f}")
    print(f"Pearson r         : {metrics['pearson_r']:.4f}  (p={metrics['pearson_p_value']:.4f})")
    print()
    print(f"GPT positive rate    : {metrics['gpt_positive_rate']:.4f}")
    print(f"Claude positive rate : {metrics['claude_positive_rate']:.4f}")
    print()
    print("Per-outlet breakdown:")
    for outlet, m in sorted(metrics["outlet_breakdown"].items()):
        print(f"  {outlet:<8}  agreement={m['agreement_rate']:.4f}  "
              f"disagree={m['disagreement_count']}/{m['sentence_count']}")

    disagree_rows = [r for r in rows if r["agreement"] == 0]
    if disagree_rows:
        ex = disagree_rows[0]
        print(f"\nExample disagreement (outlet={ex['outlet']}, gold={'biased' if ex['gold'] else 'not biased'}):")
        print(f"  Sentence : {ex['sentence'][:120]}{'...' if len(ex['sentence']) > 120 else ''}")
        print(f"  GPT      : {ex['gpt_score']:.3f} → {'biased' if ex['gpt_pred'] else 'not biased'}  [{ex['gpt_reasoning']}]")
        print(f"  Claude   : {ex['claude_score']:.3f} → {'biased' if ex['claude_pred'] else 'not biased'}  [{ex['claude_reasoning']}]")

    print(f"\nSwapping providers required zero tool-code changes.")
    print(f"Both models are fully logged in the MCP audit DB.")


if __name__ == "__main__":
    main()
