#!/usr/bin/env python3
"""Generate bias modeling visualizations from Milestone 3 experiment outputs.

Outputs
-------
outputs/viz_bias_rates_by_outlet.png      per-outlet bias rates across models
outputs/viz_score_distributions.png       GPT vs Claude score distributions
outputs/viz_pipeline_funnel.png           dual-agent pipeline flow
outputs/viz_model_comparison.png          F1 / metrics comparison bar chart
outputs/viz_agreement_heatmap.png         GPT vs Claude prediction agreement

Usage
-----
    python scripts/visualize_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

OUTLET_COLORS = {"fox": "#E8534A", "hpo": "#5B8ED6", "nyt": "#4CAF73"}
OUTLET_LABELS = {"fox": "Fox News", "hpo": "HuffPost", "nyt": "New York Times"}

# ------------------------------------------------------------------ #
# Load data                                                           #
# ------------------------------------------------------------------ #

mp  = pd.read_csv(OUTPUTS / "multi_provider_predictions.csv")
llm = pd.read_csv(OUTPUTS / "llm_pipeline_predictions.csv")

with open(OUTPUTS / "llm_pipeline_metrics.json") as f:
    llm_metrics = json.load(f)
with open(OUTPUTS / "multi_provider_metrics.json") as f:
    mp_metrics = json.load(f)


# ================================================================== #
# 1. Bias rates by outlet                                             #
# ================================================================== #

fig, ax = plt.subplots(figsize=(9, 5))

outlets   = ["fox", "hpo", "nyt"]
x         = np.arange(len(outlets))
width     = 0.22

gold_rates  = [mp.groupby("outlet").apply(lambda g: g["gold"].mean())[o] for o in outlets]
gpt_rates   = [mp.groupby("outlet").apply(lambda g: g["gpt_pred"].mean())[o] for o in outlets]
claude_rates= [mp.groupby("outlet").apply(lambda g: g["claude_pred"].mean())[o] for o in outlets]
pipe_rates  = [llm.groupby("outlet").apply(lambda g: g["final_pred"].mean())[o] for o in outlets]

b1 = ax.bar(x - 1.5*width, gold_rates,   width, label="Gold label",         color="#888888")
b2 = ax.bar(x - 0.5*width, gpt_rates,    width, label="GPT-4o-mini (auditor)", color="#4A90D9")
b3 = ax.bar(x + 0.5*width, claude_rates, width, label="Claude-haiku (solo)",  color="#E8534A")
b4 = ax.bar(x + 1.5*width, pipe_rates,   width, label="GPT→Claude pipeline",  color="#4CAF73")

ax.set_xticks(x)
ax.set_xticklabels([OUTLET_LABELS[o] for o in outlets], fontsize=11)
ax.set_ylabel("Positive (biased) rate", fontsize=11)
ax.set_title("Bias Detection Rate by Outlet and Model", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0, 0.55)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

for bars in [b1, b2, b3, b4]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                f"{h:.0%}", ha="center", va="bottom", fontsize=7.5)

fig.tight_layout()
fig.savefig(OUTPUTS / "viz_bias_rates_by_outlet.png", dpi=150)
plt.close(fig)
print("Saved viz_bias_rates_by_outlet.png")


# ================================================================== #
# 2. Bias score distributions (GPT vs Claude)                        #
# ================================================================== #

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
bins = np.linspace(0, 1, 21)

for ax, outlet in zip(axes, outlets):
    grp = mp[mp["outlet"] == outlet]
    ax.hist(grp["gpt_score"],    bins=bins, alpha=0.65, label="GPT-4o-mini",
            color="#4A90D9", edgecolor="white", linewidth=0.4)
    ax.hist(grp["claude_score"], bins=bins, alpha=0.65, label="Claude-haiku",
            color="#E8534A", edgecolor="white", linewidth=0.4)
    ax.axvline(0.4, color="black", linestyle="--", linewidth=1.2, label="Threshold (0.40)")
    ax.set_title(OUTLET_LABELS[outlet], fontsize=11, fontweight="bold")
    ax.set_xlabel("Bias score", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    if ax == axes[0]:
        ax.set_ylabel("Sentence count", fontsize=10)
        ax.legend(fontsize=8)

fig.suptitle("Bias Score Distributions: GPT-4o-mini vs Claude-haiku", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUTPUTS / "viz_score_distributions.png", dpi=150)
plt.close(fig)
print("Saved viz_score_distributions.png")


# ================================================================== #
# 3. Dual-agent pipeline funnel                                       #
# ================================================================== #

total      = llm_metrics["sample_size"]
aud_pos    = llm_metrics["verifier_calls"]
aud_neg    = total - aud_pos
confirmed  = aud_pos - llm_metrics["verifier_downgrades"]
downgraded = llm_metrics["verifier_downgrades"]
final_pos  = llm_metrics["pipeline"]["sentence_count"] * llm_metrics["pipeline"]["positive_rate"]

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")

def funnel_box(ax, x, y, w, h, color, label, count, pct=None):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor="white",
                                    linewidth=1.5, alpha=0.88)
    ax.add_patch(rect)
    pct_str = f"\n({pct:.0%})" if pct is not None else ""
    ax.text(x + w/2, y + h/2, f"{label}\nn={count}{pct_str}",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="white")

# Row 1 — all sentences
funnel_box(ax, 1, 5.5, 8, 1.0, "#555555", "All sentences", total)

# Row 2 — auditor split
funnel_box(ax, 0.2, 3.8, 3.8, 1.0, "#4A90D9",
           "GPT: not biased\n→ skip verifier", aud_neg, aud_neg/total)
funnel_box(ax, 5.0, 3.8, 4.0, 1.0, "#E8A23A",
           "GPT: biased\n→ send to Claude", aud_pos, aud_pos/total)

# Row 3 — verifier split
funnel_box(ax, 5.0, 2.0, 1.8, 1.1, "#E8534A",
           "Downgraded", downgraded, downgraded/total)
funnel_box(ax, 7.1, 2.0, 1.8, 1.1, "#4CAF73",
           "Confirmed", confirmed, confirmed/total)

# Row 4 — final
funnel_box(ax, 1, 0.3, 8, 1.0, "#333333",
           f"Final prediction: biased", int(final_pos), int(final_pos)/total)

# Arrows
for (x1, y1, x2, y2) in [
    (5, 5.5, 2.2, 4.8), (5, 5.5, 7.0, 4.8),
    (6.0, 3.8, 5.9, 3.1), (7.8, 3.8, 7.9, 3.1),
    (2.2, 3.8, 3.5, 1.3), (5.9, 2.0, 4.5, 1.3), (7.9, 2.0, 6.5, 1.3),
]:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=1.5))

ax.set_title("Dual-Agent Pipeline: Sentence Flow", fontsize=13, fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig(OUTPUTS / "viz_pipeline_funnel.png", dpi=150)
plt.close(fig)
print("Saved viz_pipeline_funnel.png")


# ================================================================== #
# 4. Model comparison bar chart (F1, precision, recall)              #
# ================================================================== #

fig, ax = plt.subplots(figsize=(9, 5))

models  = ["GPT-4o-mini\n(auditor only)", "GPT→Claude\n(pipeline)", "RoBERTa-BABE\n(baseline)"]
x       = np.arange(len(models))
width   = 0.25

# Load RoBERTa baseline if available
try:
    with open(OUTPUTS / "eval_full_auditor_only.json") as f:
        roberta = json.load(f)
    rob_f1  = roberta["f1_macro"]
    rob_pre = roberta["precision"]
    rob_rec = roberta["recall"]
except FileNotFoundError:
    rob_f1 = rob_pre = rob_rec = 0.0

f1_vals  = [llm_metrics["auditor_only"]["f1_macro"],
            llm_metrics["pipeline"]["f1_macro"],
            rob_f1]
pre_vals = [llm_metrics["auditor_only"]["precision"],
            llm_metrics["pipeline"]["precision"],
            rob_pre]
rec_vals = [llm_metrics["auditor_only"]["recall"],
            llm_metrics["pipeline"]["recall"],
            rob_rec]

b1 = ax.bar(x - width, f1_vals,  width, label="F1 Macro",  color="#4A90D9")
b2 = ax.bar(x,         pre_vals, width, label="Precision", color="#4CAF73")
b3 = ax.bar(x + width, rec_vals, width, label="Recall",    color="#E8534A")

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0, 0.9)
ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

fig.tight_layout()
fig.savefig(OUTPUTS / "viz_model_comparison.png", dpi=150)
plt.close(fig)
print("Saved viz_model_comparison.png")


# ================================================================== #
# 5. Agreement heatmap: GPT pred vs Claude pred                      #
# ================================================================== #

from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, outlet in zip(axes, outlets):
    grp = mp[mp["outlet"] == outlet]
    cm  = confusion_matrix(grp["gpt_pred"], grp["claude_pred"], labels=[0, 1])
    im  = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not biased", "Biased"], fontsize=9)
    ax.set_yticklabels(["Not biased", "Biased"], fontsize=9)
    ax.set_xlabel("Claude prediction", fontsize=10)
    ax.set_ylabel("GPT prediction", fontsize=10)
    ax.set_title(OUTLET_LABELS[outlet], fontsize=11, fontweight="bold")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > cm.max()*0.5 else "black")

fig.suptitle("GPT vs Claude Prediction Agreement by Outlet", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUTPUTS / "viz_agreement_heatmap.png", dpi=150)
plt.close(fig)
print("Saved viz_agreement_heatmap.png")

print(f"\nAll visualizations saved to {OUTPUTS}/")
