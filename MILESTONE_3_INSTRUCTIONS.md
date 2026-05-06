# Milestone 3 Instructions for Claude Code

This file is the working brief for finishing the `dual-agent-bias-detection` project. The deadline is a few days away. Read this whole file before writing code, then execute in the order given. If a task is ambiguous, prefer the smaller, working version over the more ambitious one.

---

## 1. Project Context

This is a course project for a dual-agent LLM system that detects cognitive bias in news articles. The architecture is:

- **Primary auditor:** RoBERTa-BABE (`mediabiasgroup/roberta-babe-ft`) producing sentence-level lexical bias scores in `[0, 1]`.
- **Secondary verifier:** Currently DistilBERT-MNLI (`typeform/distilbert-base-uncased-mnli`) doing NLI grounding checks. **This is broken — fixing it is task #1.**
- **MCP tool server** exposing six tools: `detect_bias`, `nli_check`, `evaluate_basil`, `basil_outlet_drift`, `temporal_drift_analysis`, `audit_recent`. All calls logged to a SQLite audit DB.
- **Dataset:** BASIL (300 news articles across Fox News, NYT, Huffington Post, sentence-level bias labels).

The repo layout (from README):
```
mcp_server/                  # MCP tool server + tools
scripts/                     # download_basil_data.py, run_deliverables.py
Primary_Auditor_LLM.ipynb
Secondary_Verifier_LLM.ipynb
basil_eda.ipynb
requirements.txt
```

Environment variables that matter: `BASIL_DATA_DIR`, `AUDITOR_MODEL_ID`, `NLI_MODEL_NAME`, `MCP_AUDIT_DB`, `BASIL_EVAL_CAP`.

## 2. Current State (Milestone 2 Baseline)

Auditor on a 500-sentence BASIL test cap at threshold 0.5:

| Metric | Value |
| --- | --- |
| Accuracy | 0.800 |
| Macro F1 | 0.649 |
| Precision | 0.600 |
| Recall | 0.321 |
| MAE (score vs. binary gold) | 0.245 |
| Predicted positive rate | 0.120 |
| Gold positive rate | 0.224 |

**Two known problems:**

1. **Verifier is non-functional.** All NLI predictions collapse to ~0.999 neutral, ~0.0009 entailment, ~0.0004 contradiction. The `supported` verdict is technically correct only because 0.0009 > 0.0004, which is meaningless.
2. **Auditor under-predicts.** Predicted positive rate (0.120) is roughly half the gold rate (0.224). Recall of 0.321 means it misses ~two-thirds of biased sentences.

## 3. Scope Discipline — What to Cut

The original proposal lists features that are NOT going to be built for Milestone 3. Do not start any of these unless every higher-priority task is complete:

- Stability test (paraphrase k times, measure variance)
- Temporal drift on real publication timestamps (the existing generic bucket tool is enough)
- Human annotation sheet
- Bias heatmap, drift timeline UI, full bias-distribution dashboard
- Per-bias-type taxonomy (confirmation, framing, anchoring, etc.) — keep it at lexical + informational
- Stability Panel in frontend
- Hallucination scoring beyond the verifier's verdict
- Trust scores

If you find yourself building any of these, stop and check the priority list.

## 4. Priority Tiers

### Tier 1 — Must-Do (blockers / fixes)

These are non-negotiable. Without these, the project doesn't have a defensible writeup.

1. **Fix the NLI verifier** (Section 5)
2. **Auditor threshold calibration** (Section 6)
3. **Full BASIL test set evaluation** (Section 7)
4. **Evaluation logs / artifacts** dumped to disk for the writeup (Section 8)

### Tier 2 — Defends the MCP architecture

The professor's feedback was that MCP looks redundant vs. function calling. At least one of these has to be done to make MCP earn its keep:

5. **Multi-provider auditor experiment** (Section 9) — highest priority in this tier
6. **External MCP server composition** (Section 10) — only if time permits

### Tier 3 — Deliverables

7. **Minimal React frontend** wired to MCP server (Section 11)
8. **Writeup, README, presentation** (Section 12)

## 5. Task: Fix the NLI Verifier

**Why it's broken:** The current setup likely uses a hypothesis like "this sentence contains bias" against the sentence as premise. MNLI models are trained on entailment between propositions, not on meta-questions about whether text "contains" something. The model has no anchor and defaults to neutral.

**What to do:**

1. Locate the `nli_check` implementation in `mcp_server/` (likely `server.py` or a `tools.py` / `nli.py`). Read it before changing anything.
2. Swap the model. Replace `typeform/distilbert-base-uncased-mnli` with one of:
   - `roberta-large-mnli` (good default, ~1.4GB)
   - `microsoft/deberta-v3-large-mnli` (stronger, ~1.7GB)
   - `cross-encoder/nli-deberta-v3-base` (smaller, faster, still solid)
   Pick based on what runs on the available hardware. Read `NLI_MODEL_NAME` from env so it's swappable.
3. Redesign the hypothesis. Instead of "this sentence contains bias," have the verifier check **specific factual claims** the auditor produced. Pattern:
   - **Premise:** the original sentence
   - **Hypothesis:** a specific extracted claim or an outlet-attributed statement (e.g., "President Obama damaged Medicare")
   - **Verdict:** `grounded` if `P(entailment) > P(contradiction)` AND `P(entailment) > 0.5`. Otherwise `ungrounded`.
4. Add a second mode: `nli_check_quote_attribution` — checks whether a sentence is a direct quote vs. authorial claim (relevant to Figure 7 in Milestone 1, which showed outlets differ in quote-vs-direct framing).
5. Write a small comparison harness in `scripts/compare_verifiers.py`:
   - Loads N (start with 100) auditor outputs from a recent BASIL run
   - Runs both old and new verifier
   - Logs distribution of verdicts and label probabilities for both
   - Outputs a CSV: `sentence, auditor_score, old_verdict, old_p_ent, old_p_con, old_p_neu, new_verdict, new_p_ent, new_p_con, new_p_neu`

**Acceptance criteria:**
- New verifier produces non-degenerate probability distributions (entailment + contradiction together > 0.1 on most inputs).
- CSV from comparison harness shows clearly different verdicts between old and new on at least 30% of sentences.
- All calls go through the MCP `nli_check` tool (don't bypass the server).

## 6. Task: Auditor Threshold Calibration

**Why:** Reporting only threshold 0.5 with recall 0.321 looks bad. A threshold sweep turns "model under-predicts" into "we calibrated to an operating point."

**What to do:**

1. In `scripts/`, create `threshold_sweep.py`.
2. Run the auditor on the BASIL test split (use the same grouped split as `evaluate_basil`). Cache the raw scores so you don't have to rerun the model for each threshold.
3. Compute precision, recall, F1, MAE for thresholds `[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]`.
4. Save:
   - `outputs/threshold_sweep.csv` — one row per threshold
   - `outputs/precision_recall_curve.png` — PR curve from raw scores
   - `outputs/threshold_choice.json` — the F1-optimal threshold and its metrics
5. Use the F1-optimal threshold for the final eval in Section 7.

**Acceptance criteria:**
- The chosen threshold yields macro F1 ≥ 0.65 (current baseline at 0.5 is 0.649, so anything ≥ that is a wash but anywhere ≥ 0.68 would be a meaningful improvement).
- The PR curve is included in the writeup.

## 7. Task: Full BASIL Test Set Evaluation

**What to do:**

1. `export BASIL_EVAL_CAP=0`
2. Run `python scripts/run_deliverables.py` with the F1-optimal threshold from Section 6.
3. Run it twice: auditor-only and auditor + verifier-corrected. The verifier-corrected version downgrades scores when verifier flags `ungrounded`.
4. Save:
   - `outputs/eval_full_auditor_only.json` — metrics dict
   - `outputs/eval_full_verifier_corrected.json` — metrics dict
   - `outputs/eval_full_predictions.csv` — `event_id, outlet, sentence, gold, auditor_score, auditor_pred, verifier_verdict, final_pred`

**Acceptance criteria:**
- Both metrics JSONs exist and contain accuracy, precision, recall, macro F1, MAE.
- Revision rate (fraction of auditor decisions changed by verifier) is computed and logged.
- A per-outlet breakdown is included (Fox / NYT / HuffPost), since cross-outlet comparison is a stated research goal.

## 8. Task: Evaluation Logs and Artifacts

The proposal explicitly calls for "testing logs" as a Milestone 3 deliverable. The MCP audit DB is the natural source.

**What to do:**

1. After all evaluation runs are complete, dump the audit DB to a portable format:
   - `outputs/mcp_audit_log.csv` — every tool call: `timestamp, tool_name, input_args_json, output_json`
   - `outputs/mcp_audit_log_sample.md` — human-readable sample of 20 representative calls for the writeup
2. Write a tiny script `scripts/export_audit_log.py` that reads from the SQLite DB and dumps these.
3. Confirm the audit log captures: every `detect_bias` call, every `nli_check` call, every `evaluate_basil` run, multi-provider calls (Section 9), and any external MCP calls (Section 10).

**Acceptance criteria:**
- Audit log CSV has at minimum 500 rows (one for every test sentence processed at least once).
- The sample MD file is included in the repo and referenced in the README.

## 9. Task: Multi-Provider Auditor Experiment (MCP defense, highest leverage)

**Why this matters:** This is the single experiment that most clearly justifies MCP over function calling. It directly answers the stated research question "Do different models disagree on bias attribution?" and demonstrates that swapping the auditor LLM is trivial because the tools live behind a protocol.

**What to do:**

1. Pick a second auditor. Options, in order of preference:
   - **OpenAI `gpt-4o-mini`** — cheap, fast, has structured-output mode that matches `detect_bias`'s JSON schema
   - **Anthropic `claude-3-5-haiku`** — also cheap, fast
   - **A second HuggingFace model** — fallback if no API budget. Try `unitary/toxic-bert` or `cardiffnlp/twitter-roberta-base-hate` as a different-domain baseline (note in writeup that they're proxies, not direct bias detectors).
2. Wire the second auditor as an MCP **client** that calls the same `detect_bias` tool. The point is: the tool is unchanged, only the client LLM differs. Implement this in `scripts/run_multi_provider.py`.
3. Run on a 200-sentence BASIL subset (random sample, stratified by outlet).
4. Compute:
   - **Per-sentence agreement:** binary agreement rate at chosen threshold
   - **Cohen's kappa** between the two auditors
   - **Score correlation:** Pearson r on the continuous scores
   - **Per-outlet disagreement breakdown**
5. Save:
   - `outputs/multi_provider_predictions.csv` — `sentence, gold, auditor_a_score, auditor_b_score, agreement`
   - `outputs/multi_provider_metrics.json` — kappa, correlation, agreement rate
   - At least one "disagreement case study" example in the writeup

**Acceptance criteria:**
- Both auditors' calls show up in the MCP audit log with distinct client identifiers.
- The two auditors disagree on at least 15% of sentences (if they don't, something is wrong with the second auditor's prompting).
- The writeup section explicitly says: "This experiment was possible without rewriting any tool code because the tools live behind the MCP server."

## 10. Task: External MCP Server Composition (Tier 2, optional)

**Skip this if Sections 5–9 aren't done yet.**

**What to do:**

1. Pick one public MCP server. Easiest options:
   - A web search MCP (Brave Search, Exa, Tavily — most have official MCP servers)
   - A Wikipedia MCP server
   - The official `fetch` MCP server (just fetches URLs)
2. Configure the verifier to be a client of both your server and the external one.
3. Add a verifier mode `nli_check_with_grounding`: when checking a factual claim, first call the external search/wiki tool to retrieve evidence, then run NLI between the retrieved evidence (premise) and the claim (hypothesis).
4. Demo this on 5–10 cases for the writeup. Don't run it on full BASIL — it's expensive and the value is the demonstration, not the metric.

**Acceptance criteria:**
- Audit log shows interleaved calls to both the local and external MCP servers in a single verifier run.
- README has a "How to add another MCP server" section.

## 11. Task: Minimal React Frontend

**Scope discipline:** the goal is a working demo, not a polished product. Do not build the heatmap, drift timeline, or stability panel.

**What to do:**

1. Single-page React app (Vite + plain CSS or Tailwind, no design system).
2. Three sections:
   - **Input:** textarea for a sentence or short passage. "Run Auditor" button.
   - **Results:** auditor score (with the chosen threshold's verdict), verifier verdict and probabilities, and the auditor's reasoning trace.
   - **Audit log:** last 20 entries from `audit_recent`, auto-refreshed after each call. Shows tool name, timestamp, truncated input, truncated output.
3. Wire to the MCP server through a thin FastAPI or Flask shim if direct MCP-from-browser isn't viable, OR have the frontend call MCP directly if the server supports it.
4. Include a "Connect to Claude Desktop" instructions panel — shows the MCP config snippet a user would paste into their Claude Desktop config to use the server themselves. This makes the "MCP server as research artifact" framing tangible.

**Acceptance criteria:**
- Runs locally with one command (e.g., `npm run dev`).
- Can submit a sentence, get auditor + verifier output, and see the corresponding audit log entries appear.
- Screenshots in the writeup.

## 12. Task: Writeup, README, Presentation

**Writeup must include:**

- Updated results table at chosen threshold for **full** BASIL test set, with auditor-only and verifier-corrected columns
- Per-outlet breakdown (Fox / NYT / HuffPost)
- Verifier improvement narrative: "DistilBERT-MNLI collapsed to neutral. We swapped to [model] and redesigned the hypothesis as [pattern]. Distribution went from [old stats] to [new stats]."
- Multi-provider results: kappa, correlation, agreement, one case study
- Honest cuts section: explicitly list what was scoped out (stability test, real-timestamp drift, human annotation, taxonomy expansion) and why
- MCP justification section: respond to the function-calling-vs-MCP critique using the multi-provider experiment as the primary evidence

**README must include:**

- Quickstart (already exists, polish it)
- Run instructions for: full eval, threshold sweep, multi-provider run, frontend, audit log export
- "Connect this MCP server to Claude Desktop" snippet — this is the public-artifact framing
- Tools table with one-line descriptions (already exists, expand)
- Citation block for the writeup

**Presentation:**

- Ryan: motivation + architecture
- Terry: methodology + results
- Maheen: MCP server + frontend demo + conclusions
- Live demo of the frontend submitting a sentence and the audit log lighting up — this is the most compelling 30 seconds of the whole presentation, plan around it.

## 13. Day-by-Day Schedule (4 days)

**Day 1:**
- Section 5: Fix verifier (model swap + hypothesis redesign + comparison harness)
- Section 6: Threshold sweep
- Commit checkpoint

**Day 2:**
- Section 7: Full BASIL eval at chosen threshold (auditor-only + verifier-corrected)
- Section 9: Multi-provider experiment
- Section 8: Audit log export
- Commit checkpoint

**Day 3:**
- Section 11: Frontend MVP
- Section 12: Writeup draft
- Section 10: External MCP composition (only if ahead of schedule)
- Commit checkpoint

**Day 4:**
- Polish, README update, presentation slides
- Final eval rerun if anything material changed
- Final commit + tag

## 14. Operating Rules for Claude Code

- **Always read existing code before editing.** The MCP server has working tools. Don't rewrite what works.
- **Every change goes through the MCP server.** Don't add code paths that bypass it — that defeats the audit log and the architectural point.
- **Save artifacts to `outputs/`** at the repo root. Create the directory if it doesn't exist. Add it to `.gitignore` for large files but commit small JSONs and CSVs that are referenced in the writeup.
- **Use environment variables** for model IDs, paths, caps. Don't hardcode.
- **Commit often** with descriptive messages. One commit per task in Sections 5–11 minimum.
- **If a task is blocked** (missing API key, model too large for hardware, BASIL data not on disk), stop and surface the blocker rather than working around it silently.
- **Do not start Tier 3 tasks before Tier 1 is complete.** Do not start the frontend before the verifier is fixed.
