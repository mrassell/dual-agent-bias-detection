# dual-agent-bias-detection

Sentence-level **media bias** detection on **BASIL**: RoBERTa-style auditor, BART-MNLI verifier, drift metrics, and optional cloud-LLM experiments—exposed through an **MCP tool server** and matching **CLI scripts**.

## What MCP does here (and why bother)

**MCP** (Model Context Protocol) is how apps like **Cursor** or **Claude Desktop** attach to a **stdio “tool server”**: they list tools, call them with JSON arguments, and get JSON back—without importing your repo or installing PyTorch in the client.

In this project MCP is useful because it:

1. **Publishes one stable tool API** — bias scores, BASIL eval, NLI, threshold sweep, full dual-agent eval, audit export—same names and payloads for any MCP host.
2. **Keeps heavy ML in one process** — the host stays thin; models run inside `python -m mcp_server`.
3. **Logs every tool call to SQLite** — UUID, timestamp, arguments, full result → reproducibility and `export_audit_artifacts` / `scripts/export_audit_log.py` for write-ups.

Scripts under `scripts/` call **`mcp_server.bias_surface`** (capability layer). **`mcp_server.mcp`** is **transport-only** (stdio FastMCP). Cloud LLM vendor/model choice lives in **`mcp_server.providers`** + **`BIAS_LLM_*`** env vars—not in MCP tool definitions.

**More detail:** [docs/REPO_LAYOUT.md](docs/REPO_LAYOUT.md) (full directory map + MCP section) · [docs/EXPERIMENT_NOTES.md](docs/EXPERIMENT_NOTES.md) (experiments & TA narrative) · [docs/README.md](docs/README.md) (doc index)

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**BASIL data** (default after download): e.g. `$HOME/basil_workspace/emnlp19-BASIL/data` (300 article JSON files).

```bash
python scripts/download_basil_data.py
export BASIL_DATA_DIR="$HOME/basil_workspace/emnlp19-BASIL/data"
```

**Local sentence auditor + NLI (required for RoBERTa/BART tools and eval scripts)** — no hub id is hard-coded; you must set:

```bash
export AUDITOR_MODEL_ID="mediabiasgroup/roberta-babe-ft"   # or your fine-tuned folder
export NLI_MODEL_NAME="facebook/bart-large-mnli"            # or another NLI checkpoint
```

**Run the MCP server (stdio)**

```bash
python -m mcp_server
```

**Env (common):** `BASIL_DATA_DIR`, **`AUDITOR_MODEL_ID` (required)**, **`NLI_MODEL_NAME` (required)** for local-model tools, `MCP_AUDIT_DB`, `MCP_DISABLE_AUDIT=1`  
**Cloud LLM slots:** `BIAS_LLM_<SLOT>_VENDOR`, `BIAS_LLM_<SLOT>_MODEL`, API keys — see [`.env.example`](.env.example).

---

## MCP tools (stdio server)

| Tool | Role |
|------|------|
| `detect_bias` | RoBERTa lexical (+ mirrored info) JSON for any sentence. |
| `nli_check` | BART-MNLI premise↔hypothesis; empty `hypothesis` → explicit bias template pack. |
| `evaluate_basil` | BASIL **test** split (event holdout); gold = any span; metrics + MAE. `max_sentences=0` = all. |
| `basil_outlet_drift` | Outlet buckets + Jensen–Shannon on same split. |
| `temporal_drift_analysis` | Drift from a JSON array of time buckets. |
| `audit_recent` | Last *N* rows from the SQLite audit log. |
| `sweep_auditor_thresholds` | Nine-threshold sweep → `outputs/threshold_sweep.csv`, `threshold_choice.json`, PR + F1 plots. |
| `export_audit_artifacts` | Audit DB → `outputs/audit_log_export.csv` + `audit_log_sample.md`. |
| `run_basil_dual_agent_eval` | RoBERTa + BART-NLI verifier on test split; `max_sentences` optional (full split is slow). |
| `llm_bias_score` | Cloud LLM bias JSON for one sentence; `slot` selects `BIAS_LLM_<SLOT>_*` env group. |
| `llm_verify` | Cloud LLM verifier JSON; `slot` selects env group (same mechanism as `llm_bias_score`). |

---

## Scripts (`scripts/` — CLI mirrors)

| Script | Notes |
|--------|--------|
| `download_basil_data.py` | Download BASIL. |
| `run_deliverables.py` | Quick BASIL demo outputs (`BASIL_EVAL_CAP`). |
| `threshold_sweep.py` | Same logic as MCP `sweep_auditor_thresholds`. |
| `export_audit_log.py` | Same as MCP `export_audit_artifacts`. |
| `run_full_eval.py` | Same as MCP `run_basil_dual_agent_eval` (full test by default). |
| `finetune_basil_auditor.py` | Fine-tune auditor on BASIL train → `outputs/roberta-babe-basil-ft/`. |
| `run_multi_provider.py` | Two configurable LLM slots; Cohen's κ, Pearson r. |
| `run_llm_pipeline.py` | Dual cloud LLM pipeline via `bias_surface` + audit (`BIAS_LLM_AUDITOR_SLOT`, optional `BIAS_LLM_VERIFIER_SLOT`). |
| `visualize_results.py` | Plots from `outputs/`. |
| `secondary_verifier.py` | Batch NLI verifier over JSONL (`mcp_server.nli` / same as MCP `nli_check`). |
| `run_demo.py`, `finetune_basil.py` | Auxiliary; see file docstrings. |

---

## Notebooks & library code

| Item | Role |
|------|------|
| `Primary_Auditor_LLM.ipynb` | RoBERTa-BABE-style auditor experiments. |
| `Secondary_Verifier_LLM.ipynb` | NLI verifier experiments. |
| `basil_eda.ipynb` | BASIL EDA. |
| `BASIL_Bias_Baseline.ipynb` | Older baseline experiments. |
| `proposal_math.py` | Scoring helpers (used by `scripts/secondary_verifier.py`). |

```bash
python scripts/secondary_verifier.py \
  --input-jsonl /path/to/auditor_sentence_judgments.jsonl \
  --output-jsonl /path/to/secondary_verifier_predictions.jsonl \
  --summary-json /path/to/secondary_verifier_summary.json
```

---

## Frontend

Standalone React app in `frontend/`:

```bash
cd frontend && npm install && npm run dev
```

Demo UI by default; set `VITE_API_BASE_URL` when a real API exists. **Railway (UI-only):** root directory `frontend`, build `npm run build`, start `npm run start`. See `frontend/README.md`.

---

## Course / milestone

See `MILESTONE_3_INSTRUCTIONS.md`.
