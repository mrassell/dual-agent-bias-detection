# Experiment notes — verifier, domain adaptation, thresholds, MCP

This document ties the engineering choices to the results artifacts under `outputs/` and the MCP tool server.

**MCP parity:** Threshold sweep, audit CSV export, and full dual-agent BASIL eval are available both as **`scripts/*.py`** and as MCP tools (`sweep_auditor_thresholds`, `export_audit_artifacts`, `run_basil_dual_agent_eval`). **Strict layout:** all paths go through **`mcp_server.bias_surface`**; `pipeline_jobs.py` is an internal helper used only from that surface; **`mcp_server.mcp`** (stdio FastMCP in `mcp/stdio_server.py`) is transport only.

## 1. Verifier (NLI): BART-large-MNLI + explicit bias hypotheses

**Problem:** The previous default NLI (`typeform/distilbert-base-uncased-mnli`) often **collapsed toward neutral** on short news sentences, so entailment/contradiction signals were unusable for grounding.

**Fix:** Use a stronger NLI checkpoint via **`NLI_MODEL_NAME`** (e.g. **`facebook/bart-large-mnli`**), loaded in `mcp_server/nli.py` — **no default is hard-coded**; the variable must be set to run `nli_check`.

**Explicit hypotheses:** When `nli_check` is called with an **empty `hypothesis`**, the server runs a **fixed template pack** on the premise (sentence):

- Lexical / slanted wording bias  
- Informational / framing bias  
- A **neutral reference** template  

The aggregate rule favors bias readings when **max(entail on bias templates) > entail(neutral reference) + 0.02** (see `nli_check_explicit_bias_pack`). Pairwise premise↔hypothesis mode is unchanged when you pass a non-empty hypothesis.

**Scripts:** `scripts/run_full_eval.py` uses the same template strings via `BIAS_VERIFY_HYPOTHESES` from `mcp_server.nli`. The standalone **`scripts/secondary_verifier.py`** CLI calls the same **`nli_check`** stack (no separate zero-shot pipeline).

---

## 2. Domain mismatch: BABE → BASIL fine-tune

**Issue:** `mediabiasgroup/roberta-babe-ft` is trained on **BABE**, not **BASIL**, so lexical scores are out-of-domain for BASIL spans.

**Mitigation:** `scripts/finetune_basil_auditor.py` fine-tunes that checkpoint on the **BASIL training split** (same `event_id` grouped split as the rest of the pipeline) and writes **`outputs/roberta-babe-basil-ft/`**. Point `AUDITOR_MODEL_ID` at that directory (or rely on `run_full_eval.py`, which defaults to it when the folder exists).

**Metrics:** After fine-tuning, re-run `scripts/threshold_sweep.py` and `scripts/run_full_eval.py`. The committed sweep artifact shows **macro-F1 ≈ 0.705** at the F1-optimal threshold **τ = 0.40**, vs **≈ 0.696** at the legacy **τ = 0.50** on the *same* score cache (`outputs/threshold_choice.json`). Milestone write-ups often cite a **~0.63** macro-F1 **before** BASIL adaptation / threshold work on mismatched-domain checkpoints—your exact baseline will differ; use `threshold_sweep.csv` after your own `finetune_basil_auditor.py` run for paper numbers.

---

## 3. Threshold tuning (not hard-coded 0.5)

`scripts/threshold_sweep.py` evaluates **nine thresholds** `[0.20 … 0.60]` on cached RoBERTa lexical probabilities over the **BASIL test split** (event holdout). It writes:

| Artifact | Role |
|----------|------|
| `outputs/threshold_sweep.csv` | Per-threshold accuracy / precision / recall / macro-F1 / MAE |
| `outputs/threshold_choice.json` | **`f1_optimal`** (e.g. **τ ≈ 0.40**) vs **`fixed_threshold_0_50`** for A/B |
| `outputs/precision_recall_curve.png` | PR curve with swept points annotated |
| `outputs/threshold_vs_f1_macro.png` | Macro-F1 vs threshold |

Downstream scripts read `f1_optimal.threshold` when `AUDITOR_THRESHOLD` is unset (`run_multi_provider.py`, `run_llm_pipeline.py`, `run_full_eval.py`, etc.).

---

## 4. Multi-provider experiment (~201 sentences)

`scripts/run_multi_provider.py` scores the **same stratified BASIL sample** (~201 sentences at default `SAMPLE_SIZE`) with **two configurable LLM backends** (`BIAS_LLM_<SLOT>_VENDOR` / `_MODEL`; default slot ids `A` and `B`, overridable via `MULTI_PROVIDER_SLOT_A` / `MULTI_PROVIDER_SLOT_B`). Each score is **`bias_surface.run_llm_bias_score`** (same callable as MCP tool **`llm_bias_score`**), with **one SQLite row per inference** when audit is enabled. Outputs: `outputs/multi_provider_predictions.csv`, `outputs/multi_provider_metrics.json`.

**Dual cloud LLM pipeline:** `scripts/run_llm_pipeline.py` uses **`BIAS_LLM_AUDITOR_SLOT`** and optional **`BIAS_LLM_VERIFIER_SLOT`** (same env mechanism). No `PIPELINE_MODE` / vendor-specific mode strings.

**RoBERTa + NLI dual-agent eval:** `run_basil_dual_agent_eval` / `pipeline_jobs.run_full_basil_auditor_nli_eval` logs **`roberta_auditor_infer`** once per sentence and **`nli_check`** once per hypothesis in the verifier loop (orchestration metadata in `__audit` on the arguments JSON). Set `MCP_DISABLE_AUDIT=1` to skip **all** `log_call` persistence (including these rows).

---

## 5. MCP vs “just API / function calls”

**Concrete proof — multi-provider:** Swapping vendors or models is **configuration-only** (`BIAS_LLM_*`, API keys). Both slots still log through the **same** `log_call` schema and **same SQLite audit DB** (`mcp_server/auditing`). With ad-hoc REST or vendor-native function calling, you typically **re-implement** auth, request shapes, and logging per provider.

**What MCP adds beyond “another RPC”:**

1. **Stable capability surface** — **`mcp_server/bias_surface.py`** is the **only** supported entry for tools and CLIs: it composes `basil_eval`, `nli`, `auditor`, `bias`, `drift`, `auditing`, **`providers`** (cloud LLM slots), and long-running **`pipeline_jobs`**. **`mcp_server/mcp/stdio_server.py`** is **transport only** (FastMCP stdio): it maps tool names → `bias_surface.run_*` and applies MCP-level auditing—it does **not** import evaluator modules directly.
2. **Mandatory audit trail** — every MCP tool invocation (and script calls that use `log_call`) stores UUID, timestamp, arguments, and full JSON result. **`scripts/export_audit_log.py`** materializes `outputs/audit_log_export.csv` and `outputs/audit_log_sample.md` for the paper.
3. **Reproducibility** — the audit DB is the single replay source; not dependent on ad-hoc client logs.

**Short version:** MCP is the **abstraction layer** that keeps the **dual-agent / BASIL** stack **provider-configurable** (for LLM scripts) and **auditable by design**—the same tool names and JSON contract for any MCP-capable client, with SQLite as the ground-truth log—not an accident of printf debugging.

---

## 6. Quick command cheat sheet

```bash
# Fine-tune auditor on BASIL train split
BASIL_DATA_DIR=... python scripts/finetune_basil_auditor.py

# Sweep thresholds + plots (uses AUDITOR_MODEL_ID)
BASIL_DATA_DIR=... python scripts/threshold_sweep.py

# Full test eval + NLI verifier correction
BASIL_DATA_DIR=... python scripts/run_full_eval.py

# Multi-provider disagreement study
BASIL_DATA_DIR=... python scripts/run_multi_provider.py

# Audit export for the write-up
python scripts/export_audit_log.py
```
