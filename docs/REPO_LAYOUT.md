# Repository layout

How the repo is organized and where to look for each concern.

## Top-level map

| Path | Purpose |
|------|---------|
| **`mcp_server/`** | Layered Python package (see below). **`bias_surface.py`** = **capability API** (`run_*`). **`mcp/`** = **MCP stdio transport only**. **`providers/`** = env-driven, vendor-neutral LLMs. **`auditing/`** = SQLite replay log. Domain: `basil_*`, `auditor`, `nli`, `bias`, `drift`, **`pipeline_jobs.py`**. |
| **`scripts/`** | CLI entrypoints; call **`bias_surface.run_*`** (same as MCP tools). |
| **`docs/`** | Long-form write-ups: this file, **`EXPERIMENT_NOTES.md`**. |
| **`outputs/`** | Generated metrics, CSVs, plots, eval JSON. |
| **`frontend/`** | Standalone React UI (demo / future API). |
| **`demo/`** | Small demo server snippet (optional). |
| **`*.ipynb`** | Research notebooks. |
| **`MILESTONE_3_INSTRUCTIONS.md`** | Course / milestone checklist. |

## `mcp_server/` layers

| Path | Role |
|------|------|
| **`mcp/stdio_server.py`** | **Transport only:** FastMCP stdio → `bias_surface.run_*` + MCP-level `log_call` wrap. |
| **`bias_surface.py`** | **Capability surface:** all tool/CLI entrypoints (`run_*`). |
| **`providers/`** | **`chat_providers.py`**: OpenAI / Anthropic / OpenAI-compatible from **`BIAS_LLM_*`**. **`prompts/`**: default scoring/verifier text. |
| **`auditing/`** | **`sqlite_store.py`**: append-only DB. **`invocation.py`**: `__audit` envelopes on rows. |
| **`pipeline_jobs.py`** | Long jobs; **imported only from** `bias_surface`. |
| **`server.py`** | Thin re-export of `mcp.stdio_server` (backward compatibility). |
| **`audit.py`**, **`invocation.py`**, **`chat_providers.py`**, **`prompt_loader.py`** | Same — re-exports; prefer **`auditing`** / **`providers`**. |
| `basil_dataset.py`, `basil_paths.py`, `basil_eval.py` | BASIL loading, splits, metrics. |
| `auditor.py`, `bias.py` | RoBERTa-style scoring + structured JSON. |
| `nli.py` | BART-MNLI + explicit bias templates. |
| `drift.py` | Jensen–Shannon drift helpers. |
| **`__main__.py`** | `python -m mcp_server` → **`mcp.main()`** (stdio MCP). |

## `scripts/` (detail)

| Script | MCP twin (if any) | Purpose |
|--------|-------------------|---------|
| `download_basil_data.py` | — | Fetch BASIL corpus. |
| `run_deliverables.py` | — | Quick slides over BASIL via **`bias_surface.run_*`**. |
| `threshold_sweep.py` | `sweep_auditor_thresholds` | Nine-threshold sweep + plots. |
| `export_audit_log.py` | `export_audit_artifacts` | Audit DB → CSV + Markdown. |
| `run_full_eval.py` | `run_basil_dual_agent_eval` | Full test split: auditor + NLI verifier. |
| `finetune_basil_auditor.py` | — | Supervised fine-tune on BASIL train. |
| `run_multi_provider.py` | — | Two LLM env slots, disagreement metrics. |
| `run_llm_pipeline.py` | — | Dual cloud LLM pipeline (`BIAS_LLM_AUDITOR_SLOT`, optional verifier slot). |
| `visualize_results.py` | — | Plots from `outputs/`. |
| `secondary_verifier.py` | `nli_check` stack | JSONL batch verifier. |
| `run_demo.py`, `finetune_basil.py` | — | Legacy / auxiliary. |

---

## What MCP does here

The **Model Context Protocol** is how an agent host talks to your server over **stdio**: list tools, invoke them, get JSON back.

- **Abstraction:** tool names and schemas are stable; **`providers`** keeps LLM vendor/model configuration in **environment**, not in MCP tool code.
- **Same paths as scripts:** both use **`bias_surface.run_*`**; MCP adds uniform **`log_call`** in **`mcp/stdio_server.py`** for each tool invocation.
- **Audit:** SQLite lives in **`auditing`** and is usable from scripts without MCP.

MCP **standardizes discovery and invocation**; **model-agnostic** behavior is **`mcp_server.providers`** + **`BIAS_LLM_*`**.
