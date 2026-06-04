#!/usr/bin/env python3
"""Omega (cis/trans) distribution for the MD and NMR validation ensembles.

Reads dihedrals.csv (per-model backbone dihedrals) and writes
OmegaDistributionPlotter.png, both alongside this script.
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
BAR_COLOR = "#4C72B0"   # blue, matching the other plots in this dir


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
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ax, src in zip(axes, ["MD", "NMR"]):
    omegas = omegas_by_source[src]
    ax.hist(omegas, bins=np.arange(-90, 271, 5),
            color=BAR_COLOR, edgecolor="black", linewidth=0.4)

    # cis/trans reference lines
    ax.axvline(180, color="#C44E52", lw=1.2, ls="--", label=r"trans ($\pm$180°)")
    ax.axvline(0, color="#DD8452", lw=1.2, ls="--", label="cis (0°)")

    ax.set_xlim(-90, 270)
    ax.set_xticks(np.arange(-90, 271, 90))
    ax.set_xlabel(r"$\omega$ (°)", fontsize=16)
    ax.set_title(f"{src} (n={len(omegas)})", fontsize=16)
    ax.grid(alpha=0.3)
    ax.tick_params(width=2, length=6, color="black", labelsize=15, direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")

axes[0].set_ylabel("Count", fontsize=16)
axes[0].legend(frameon=False, fontsize=14)

plt.tight_layout()
fig.savefig(SCRIPT_DIR / "OmegaDistributionPlotter.png", dpi=300,
            bbox_inches="tight")
plt.show()
