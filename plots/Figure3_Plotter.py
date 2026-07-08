#!/usr/bin/env python3
"""Combined 2x3 grid: MD vs NMR Ramachandran for all 6 validation peptides.

Reads dihedrals.csv (per-model phi/psi backbone dihedrals) and writes
Figure3.png, both alongside this script.
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.transforms import ScaledTranslation

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "dihedrals.csv"

# --- Plot config ---
# Canonical peptide order / casing shared with the other plots in this dir.
PEPTIDES = ["7L9D", "7L96", "7L98", "7UBG", "7UZL", "8CWA"]
MD_COLOR  = "#4C72B0"   # blue
NMR_COLOR = "#C44E52"   # red

# --- Load data (CSV peptide ids are lower-case) ---
data = {pid: {"MD": ([], []), "NMR": ([], [])} for pid in PEPTIDES}
with open(CSV_PATH) as fh:
    for row in csv.DictReader(fh):
        if not row["phi"] or not row["psi"]:
            continue
        pid = row["peptide"].upper()
        data[pid][row["source"]][0].append(float(row["phi"]))
        data[pid][row["source"]][1].append(float(row["psi"]))

# --- Plot ---
sns.set_theme(style="whitegrid", context="talk")
fig, axes = plt.subplots(2, 3, figsize=(16.5, 11), sharex=True, sharey=True)

letter_offset = ScaledTranslation(-34 / 72, 6 / 72, fig.dpi_scale_trans)
for ax, pid, letter in zip(axes.flat, PEPTIDES, "abcdef"):
    md_x, md_y   = data[pid]["MD"]
    nmr_x, nmr_y = data[pid]["NMR"]
    ax.scatter(md_x, md_y, s=14, c=MD_COLOR, alpha=0.35, edgecolors="none",
               label=f"MD (n={len(md_x)})")
    ax.scatter(nmr_x, nmr_y, s=60, facecolors="none",
               edgecolors=NMR_COLOR, linewidths=1.6,
               label=f"NMR (n={len(nmr_x)})")

    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 90))
    ax.set_yticks(np.arange(-180, 181, 90))
    ax.axhline(0, color="lightgrey", lw=0.5)
    ax.axvline(0, color="lightgrey", lw=0.5)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.03, 0.08), fontsize=14,
                    frameon=True, framealpha=0.85, edgecolor="grey")
    leg.get_frame().set_linewidth(0.9)

    # Peptide id as the panel title, centered at the top (title size, 16)
    ax.text(0.5, 0.96, pid, transform=ax.transAxes,
            ha="center", va="top", fontsize=16, fontweight="bold")
    # Bold panel letter at the top-left corner
    ax.text(0.0, 1.0, letter, transform=ax.transAxes + letter_offset,
            fontsize=22, fontweight="bold", va="bottom", ha="right")

    ax.tick_params(axis="both", labelsize=15, width=2, direction="in", pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")

# Outer labels only
for ax in axes[-1, :]:
    ax.set_xlabel(r"$\phi$ (°)", fontsize=16)
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\psi$ (°)", fontsize=16)

plt.tight_layout()
fig.savefig(SCRIPT_DIR / "Figure3.pdf", bbox_inches="tight")           # vector, for LaTeX
fig.savefig(SCRIPT_DIR / "Figure3.png", dpi=300, bbox_inches="tight")  # preview only
plt.show()
