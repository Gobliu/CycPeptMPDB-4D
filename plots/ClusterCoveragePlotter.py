#!/usr/bin/env python3
"""Cluster-coverage figure for the MD ensembles.

Two panels:
  (a) mean cumulative coverage vs number of clusters (+/- 1 std band), and
  (b) mean coverage retained when keeping only the top-N clusters (+/- 1 std).

Reads cumulative_cluster_coverage.xlsx and summary_coverage.xlsx and writes
ClusterCoveragePlotter.png, all alongside this script.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.transforms import ScaledTranslation

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
CUMULATIVE_XLSX = SCRIPT_DIR / "cumulative_cluster_coverage.xlsx"
SUMMARY_XLSX = SCRIPT_DIR / "summary_coverage.xlsx"

# --- Plot config ---
BAR_COLOR = "#4C72B0"   # blue, matching the other plots in this dir
N_CLUSTERS_SHOWN = 100  # curve is essentially saturated by here

# --- Load data ---
# (a) pooled (hexane+water), 300-frame ensembles
cum = pd.read_excel(CUMULATIVE_XLSX, sheet_name="300frames_all")
# (b) pooled summary: avg +/- sd coverage at fixed top-N cutoffs
summary = pd.read_excel(SUMMARY_XLSX, sheet_name="summary")
topn_labels = [s.replace("top_", "Top ").replace("Top all", "All")
               for s in summary["hexane+water"]]

# --- Plot ---
sns.set_theme(style="whitegrid", context="talk")
fig, axes = plt.subplots(1, 2, figsize=(16.5, 5.5))

# (a) Cumulative coverage vs number of clusters
ax = axes[0]
d = cum[cum["cluster_index"] <= N_CLUSTERS_SHOWN]
x = d["cluster_index"]
mean = d["mean_coverage_pct"]
std = d["std_coverage_pct"]
ax.plot(x, mean, color=BAR_COLOR, lw=2)
ax.fill_between(x, mean - std, mean + std, color=BAR_COLOR, alpha=0.18,
                label="$\\pm$1 std")
ax.set_xlim(1, N_CLUSTERS_SHOWN)
ax.set_ylim(0, 100)
ax.set_xlabel("Number of clusters", fontsize=16)
ax.set_ylabel("Cumulative coverage (%)", fontsize=16)
ax.set_title("Coverage vs cluster count (300 frames)", fontsize=16)
ax.legend(loc="lower right", frameon=False, fontsize=14)

# (b) Mean coverage retained by the top-N clusters (pooled summary)
ax = axes[1]
xpos = np.arange(len(summary))
ax.errorbar(xpos, summary["avg"], yerr=summary["sd"],
            fmt="o-", color=BAR_COLOR, lw=2, markersize=8,
            capsize=5, elinewidth=1.4)
ax.set_xticks(xpos)
ax.set_xticklabels(topn_labels)
ax.set_xlim(-0.4, len(summary) - 0.6)
ax.set_ylim(80, 102)
ax.set_xlabel("Top-N clusters of 300 frames", fontsize=16)
ax.set_ylabel("Coverage by 100 frames (%)", fontsize=16)
ax.set_title("Cluster coverage (100 vs. 300 frames)", fontsize=16)

for ax in axes:
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=15, width=2, direction="in", pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")

# Bold panel letters at each panel's top-left corner (matches Figure2-6)
label_offset = ScaledTranslation(-34 / 72, 6 / 72, fig.dpi_scale_trans)
for ax, letter in zip(axes, "ab"):
    ax.text(0.0, 1.0, letter, transform=ax.transAxes + label_offset,
            fontsize=22, fontweight="bold", va="bottom", ha="right")

plt.tight_layout()
fig.savefig(SCRIPT_DIR / "ClusterCoveragePlotter.pdf", bbox_inches="tight")           # vector, for LaTeX
fig.savefig(SCRIPT_DIR / "ClusterCoveragePlotter.png", dpi=300, bbox_inches="tight")  # preview only
plt.show()
