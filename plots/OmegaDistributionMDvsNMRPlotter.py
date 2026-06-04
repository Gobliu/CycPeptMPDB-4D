#!/usr/bin/env python3
"""Omega (cis/trans) distribution for the MD and NMR validation ensembles.

Reads dihedrals.csv (per-model backbone dihedrals) and writes
OmegaDistributionMDvsNMRPlotter.png, both alongside this script.
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "dihedrals.csv"

# --- Plot config ---
MD_COLOR  = "#4C72B0"   # blue
NMR_COLOR = "#C44E52"   # red


def wrap_omega(deg):
    """Map omega into [-90, 270) so the trans peak sits centered at 180°."""
    return deg + 360.0 if deg < -90.0 else deg


# --- Load data ---
omegas_by_source = {"MD": [], "NMR": []}
with open(CSV_PATH) as fh:
    for row in csv.DictReader(fh):
        if not row["omega"]:
            continue
        omegas_by_source[row["source"]].append(wrap_omega(float(row["omega"])))

# --- Plot ---
sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(9, 5))

bins = np.arange(-90, 271, 5)
# Normalise each ensemble to a fraction so the very different sample sizes
# (MD >> NMR) are comparable.
md, nmr = omegas_by_source["MD"], omegas_by_source["NMR"]
ax.hist(md, bins=bins, weights=np.ones(len(md)) / len(md),
        color=MD_COLOR, alpha=0.55, edgecolor="none",
        label=f"MD (n={len(md)})")
ax.hist(nmr, bins=bins, weights=np.ones(len(nmr)) / len(nmr),
        histtype="step", color=NMR_COLOR, linewidth=2.0,
        label=f"NMR (n={len(nmr)})")

# cis/trans reference lines
ax.axvline(180, color="grey", lw=1.0, ls="--")
ax.axvline(0, color="grey", lw=1.0, ls="--")
ax.text(180, ax.get_ylim()[1], " trans", ha="left", va="top", fontsize=12, color="grey")
ax.text(0, ax.get_ylim()[1], " cis", ha="left", va="top", fontsize=12, color="grey")

ax.set_xlim(-90, 270)
ax.set_xticks(np.arange(-90, 271, 90))
ax.set_xlabel(r"$\omega$ (°)", fontsize=16)
ax.set_ylabel("Fraction", fontsize=16)
ax.set_title(r"$\omega$ distribution: MD vs NMR", fontsize=16)
ax.grid(alpha=0.3)
ax.legend(loc="upper left", frameon=False, fontsize=14)
ax.tick_params(width=2, length=6, color="black", labelsize=15, direction="in")
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_edgecolor("black")

plt.tight_layout()
fig.savefig(SCRIPT_DIR / "OmegaDistributionMDvsNMRPlotter.png", dpi=300,
            bbox_inches="tight")
plt.show()
