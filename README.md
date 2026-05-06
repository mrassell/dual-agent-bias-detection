# dual-agent-bias-detection

## MCP server (BASIL-backed)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**BASIL data path (default after download):**  
`/Users/<you>/basil_workspace/emnlp19-BASIL/data` — 300 article JSON files.

One-time download (same source as the notebook):

```bash
python3 scripts/download_basil_data.py
export BASIL_DATA_DIR="$HOME/basil_workspace/emnlp19-BASIL/data"
```

Optional: set `BASIL_DATA_DIR` if the corpus lives somewhere else.

**Tools**

| Tool | Role |
|------|------|
| `evaluate_basil` | Real **test split** (grouped by `event_id`), gold = any span; metrics: accuracy, macro-F1, MAE, etc. `max_sentences=0` = full test set. |
| `basil_outlet_drift` | Same test split, buckets = **real outlets**; Jensen–Shannon between low vs high mean lexical score. |
| `detect_bias` | Structured JSON scores for any sentence (use on BASIL rows in demos). |
| `nli_check` | Premise vs hypothesis (e.g. check a claim against a BASIL sentence). |
| `temporal_drift_analysis` | Generic bucket JSON (optional). |
| `audit_recent` | SQLite audit tail. |

**Env:** `BASIL_DATA_DIR`, `AUDITOR_MODEL_ID`, `NLI_MODEL_NAME`, `MCP_AUDIT_DB`, `MCP_DISABLE_AUDIT=1`.

**Projector run (requires BASIL on disk):**

```bash
# optional: limit sentences for a faster pass (default 500 in script)
export BASIL_EVAL_CAP=500
python scripts/run_deliverables.py
```

Full test split: `export BASIL_EVAL_CAP=0`

`from mcp_server.basil_dataset import load_basil_sentences`

## Notebooks and scripts

- `Primary_Auditor_LLM.ipynb` — sentence-level auditor (RoBERTa-BABE) emitting structured bias judgments.
- `Secondary_Verifier_LLM.ipynb` — NLI verifier (BART-large-MNLI) checking whether the auditor's judgment is supported; aggregates verified document scores.
- `basil_eda.ipynb` — exploratory analysis of the BASIL corpus.
- `BASIL_Bias_Baseline.ipynb` — older experimental notebook.
- `proposal_math.py` — scoring functions matching the proposal equations (sentence scoring, article aggregation, revision rate, MAE).
- `secondary_verifier.py` — standalone secondary verifier script.

```bash
python secondary_verifier.py \
  --input-jsonl /path/to/auditor_sentence_judgments.jsonl \
  --output-jsonl /path/to/secondary_verifier_predictions.jsonl \
  --summary-json /path/to/secondary_verifier_summary.json
```
