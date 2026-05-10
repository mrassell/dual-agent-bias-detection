from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from proposal_math import (
    article_level_scores,
    counts_to_metrics,
    gold_sentence_scores,
    mean_absolute_error,
    revision_rate,
    sentence_bias_scores,
    confusion_counts,
)

DOWNGRADE_THRESHOLD = 0.35  # auditor says biased but NLI < this → flip
UPGRADE_THRESHOLD = 0.65  # auditor says not_biased but NLI > this → flip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the secondary NLI verifier (same entailment stack as MCP nli_check) "
            "over primary sentence predictions."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        default=os.environ.get("NLI_MODEL_NAME", "").strip(),
        help="NLI checkpoint id (sets NLI_MODEL_NAME). Required if env NLI_MODEL_NAME is unset.",
    )
    return parser.parse_args()


def _entailment_scores(sentence: str) -> tuple[float, float, float]:
    """Return (lexical_entail, informational_entail, support_score) using MCP-aligned hypotheses."""
    from mcp_server.nli import BIAS_VERIFY_HYPOTHESES, nli_check

    entails: list[float] = []
    for hyp in BIAS_VERIFY_HYPOTHESES:
        r = nli_check(sentence, hyp)
        entails.append(float(r.get("entailment_probability", 0.0)))
    if len(entails) < 2:
        entails.extend([0.0] * (2 - len(entails)))
    lex_support, inf_support = entails[0], entails[1]
    support_score = max(lex_support, inf_support)
    return lex_support, inf_support, support_score


def run_nli_verifier(record: dict[str, Any]) -> dict[str, Any]:
    sentence = record["sentence_text"]
    lex_support, inf_support, support_score = _entailment_scores(sentence)

    auditor_label = record["primary_prediction"]["label"]
    auditor_score = float(record["primary_prediction"]["confidence"])

    if auditor_label == "biased" and support_score < DOWNGRADE_THRESHOLD:
        revised_bias_score = auditor_score * support_score
        final_label = "not_biased" if revised_bias_score < 0.5 else "biased"
        rationale = f"NLI entailment support low ({support_score:.3f}); auditor score downgraded."
        supported = False
    elif auditor_label == "not_biased" and support_score > UPGRADE_THRESHOLD:
        revised_bias_score = support_score
        final_label = "biased"
        rationale = f"NLI entailment support high ({support_score:.3f}); auditor label upgraded."
        supported = True
    else:
        revised_bias_score = auditor_score
        final_label = auditor_label
        supported = auditor_label == "biased"
        rationale = f"NLI entailment ({support_score:.3f}) consistent with auditor."

    return {
        "supported": supported,
        "support_score": round(support_score, 6),
        "lex_support": round(lex_support, 6),
        "inf_support": round(inf_support, 6),
        "revised_bias_score": round(revised_bias_score, 6),
        "verifier_rationale": rationale,
        "raw_response": f"entail_lex={lex_support:.3f}, entail_inf={inf_support:.3f} (NLI)",
        "verifier_label": final_label,
        "verifier_confidence": round(support_score, 6),
    }


def combine_predictions(primary: dict[str, Any], verifier: dict[str, Any]) -> dict[str, Any]:
    agreement = primary["label"] == verifier["verifier_label"]
    final_label = verifier["verifier_label"]
    final_confidence = round(
        (float(primary["confidence"]) + float(verifier["verifier_confidence"])) / 2.0, 6
    )
    return {
        "agreement": agreement,
        "final_label": final_label,
        "final_confidence": final_confidence,
        "primary_confidence": round(float(primary["confidence"]), 6),
        "verifier_confidence": round(float(verifier["verifier_confidence"]), 6),
    }


def main() -> int:
    args = parse_args()
    if not (args.model_name or "").strip():
        print(
            "ERROR: pass --model-name or set NLI_MODEL_NAME (no default in code).",
            file=sys.stderr,
        )
        return 1
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ["NLI_MODEL_NAME"] = args.model_name.strip()
    from mcp_server.nli import reset_nli_for_tests

    reset_nli_for_tests()

    records: list[dict[str, Any]] = []
    with args.input_jsonl.open() as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    augmented: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        if (i + 1) % 200 == 0:
            print(f"Processing {i + 1} / {len(records)}")

        verifier = run_nli_verifier(record)
        verification = combine_predictions(record["primary_prediction"], verifier)

        primary_scores = sentence_bias_scores(
            record["primary_prediction"]["confidence"],
            record["primary_prediction"]["label"],
            reasoning="Primary model output.",
        )
        verifier_scores = sentence_bias_scores(
            verifier["revised_bias_score"],
            verifier["verifier_label"],
            reasoning=verifier["verifier_rationale"],
        )
        final_scores = sentence_bias_scores(
            verification["final_confidence"],
            verification["final_label"],
            reasoning="Combined primary and verifier confidence.",
        )
        augmented.append(
            {
                **record,
                "secondary_verifier": verifier,
                "verification": verification,
                "primary_scores": primary_scores,
                "verifier_scores": verifier_scores,
                "final_scores": final_scores,
                "gold_scores": gold_sentence_scores(record["gold_reference"]["bias_types"]),
            }
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as handle:
        for record in augmented:
            handle.write(json.dumps(record) + "\n")

    agreement_rate = sum(1 for r in augmented if r["verification"]["agreement"]) / len(augmented)
    summary = {
        "model_name": args.model_name,
        "verifier_backend": "mcp_server.nli nli_check (BART-MNLI entailment)",
        "input_records": len(augmented),
        "agreement_rate": round(agreement_rate, 6),
        "revision_rate": round(revision_rate(augmented), 6),
        "primary_metrics": counts_to_metrics(
            confusion_counts(augmented, lambda r: r["primary_prediction"]["label"])
        ),
        "verifier_metrics": counts_to_metrics(
            confusion_counts(augmented, lambda r: r["secondary_verifier"]["verifier_label"])
        ),
        "final_metrics": counts_to_metrics(
            confusion_counts(augmented, lambda r: r["verification"]["final_label"])
        ),
        "sentence_level_mae": {
            "primary": mean_absolute_error(augmented, "primary_scores"),
            "verifier": mean_absolute_error(augmented, "verifier_scores"),
            "final": mean_absolute_error(augmented, "final_scores"),
        },
        "article_level_scores": article_level_scores(augmented),
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
