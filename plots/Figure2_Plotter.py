#!/usr/bin/env python3
"""Figure 2: publication-ready 3-panel composite.

Layout:
    (a) NOE constraint deviations  -- top, spanning full width
    (b) Omega distribution MD vs NMR -- bottom left
    (c) NMR-derived conformer coverage vs RMSD threshold -- bottom right

Merges NOEPlotter.py, OmegaDistributionMDvsNMRPlotter.py and CoveragePlotter.py
into a single figure. Writes Figure2.png alongside this script.
"""
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation
from pathlib import Path

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CSV_DIR = REPO_ROOT / "csvs"
NOE_CSV = CSV_DIR / "all_NOE_MD_comparison.csv"

# --- Shared style ---
BLOCK_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
PDB_LIST = ["7L9D", "7L96", "7L98", "7UBG", "7UZL", "8CWA"]
MD_COLOR = "#4C72B0"   # blue
NMR_COLOR = "#C44E52"  # red


# ============================================================
# Panel (a) -- NOE constraint deviations
# ============================================================
def plot_noe(ax):
    """One bar per NOE restraint: MD r^-6 distance minus the NMR upper bound.

    Bars <= 0 satisfy the restraint; bars > 0 are violations. Grey whiskers span
    the 5th-95th percentile of the per-frame MD distance (relative to r_traj), so
    a whisker crossing zero flags a transiently violated restraint. Restraints are
    grouped and colored by peptide, matching panel (c).
    """
    df = pd.read_csv(NOE_CSV)
    df = df[df["status"] == "ok"].reset_index(drop=True)

    x, dev, lo_err, hi_err, colors, names = noe_group_layout(df)
    ax.bar(x, dev, color=colors, width=0.72, edgecolor="white", linewidth=0.3, zorder=3)
    ax.errorbar(x, dev, yerr=[lo_err, hi_err], fmt="none",
                ecolor="0.35", elinewidth=0.7, capsize=1.2, alpha=0.7, zorder=4)
    ax.axhline(0, color="black", linewidth=1.0, alpha=0.8, zorder=5)

    ymin = (df["md_p5_A"] - df["upper_A"]).min()
    ymax = (df["md_p95_A"] - df["upper_A"]).max()
    ax.set_ylim(ymin - 0.25, ymax + 1.0)   # headroom for the color key
    ax.set_xlim(x[0] - 1.5, x[-1] + 1.5)
    ax.set_ylabel(r"$\mathrm{r_{traj}} - \mathrm{r_{NOE}}\ (\mathrm{Å})$", fontsize=16)
    ax.set_title("NOE distance deviations", fontsize=16)

    # PDB identified by color (matches panel c); no per-restraint x labels
    ax.set_xticks([])
    handles = [Patch(facecolor=BLOCK_COLORS[PDB_LIST.index(p)], label=p) for p in names]
    ax.legend(handles=handles, ncol=len(handles), frameon=False, fontsize=14,
              loc="upper center", bbox_to_anchor=(0.5, 1.0),
              columnspacing=1.2, handlelength=1.2, handletextpad=0.5)
    ax.tick_params(axis="y", labelsize=15, width=2, direction="in", pad=2)
    ax.grid(False, axis="x")
    ax.grid(alpha=0.3, axis="y")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")


def noe_group_layout(df, gap=2):
    """Assign x positions (gaps between peptides) and per-bar geometry, colored by peptide."""
    x, dev, lo_err, hi_err, colors, names = [], [], [], [], [], []
    cursor = 0
    for i, pep in enumerate(PDB_LIST):
        sub = df[df["peptide"] == pep]
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            x.append(cursor)
            dev.append(row["dev_A"])
            # whisker = 5th/95th percentile of the MD distance, relative to r_traj
            lo_err.append(row["md_r6_A"] - row["md_p5_A"])
            hi_err.append(row["md_p95_A"] - row["md_r6_A"])
            colors.append(BLOCK_COLORS[i])
            cursor += 1
        names.append(pep)
        cursor += gap
    return (np.array(x), np.array(dev), np.array(lo_err),
            np.array(hi_err), colors, names)


# ============================================================
# Panel (b) -- Omega distribution MD vs NMR
# ============================================================
def plot_omega(ax):
    omegas_by_source = {"MD": [], "NMR": []}
    with open(SCRIPT_DIR / "dihedrals.csv") as fh:
        for row in csv.DictReader(fh):
            if not row["omega"]:
                continue
            omegas_by_source[row["source"]].append(wrap_omega(float(row["omega"])))

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
    # cis/trans labels next to their peaks (matching Figure 6b)
    ax.text(25, 0.40, "Cis", fontsize=16, fontweight="bold", ha="left", va="center")
    ax.text(200, 0.40, "Trans", fontsize=16, fontweight="bold", ha="left", va="center")

    ax.set_xlim(-90, 270)
    ax.set_xticks(np.arange(-90, 271, 90))
    ax.set_xlabel(r"$\omega$ (°)", fontsize=16)
    ax.set_ylabel("Fraction", fontsize=16)
    ax.set_title(r"$\omega$ distribution", fontsize=16)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", frameon=False, fontsize=14)
    ax.tick_params(width=2, length=6, color="black", labelsize=15, direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")


def wrap_omega(deg):
    """Map omega into [-90, 270) so the trans peak sits centered at 180°."""
    return deg + 360.0 if deg < -90.0 else deg


# ============================================================
# Panel (c) -- NMR-derived conformer coverage vs RMSD threshold
# ============================================================
def plot_coverage(ax):
    df = pd.read_csv(CSV_DIR / "coverage.csv")
    x = df.iloc[:, 0]

    for i, pdb in enumerate(PDB_LIST):
        ax.plot(x, df[pdb], color=BLOCK_COLORS[i], linestyle="--",
                linewidth=1.5, label=pdb)

    ax.set_ylabel("Coverage", fontsize=16)
    ax.set_xlabel("RMSD threshold (Å)", fontsize=16)
    ax.set_title("NMR-derived conformer coverage", fontsize=16)
    ax.tick_params(axis="y", labelsize=15, width=2, direction="in", pad=2)
    ax.tick_params(axis="x", labelsize=15, width=2, direction="in", pad=2)
    ax.grid(alpha=0.3)
    ax.legend(ncol=1, frameon=False, loc="upper left", fontsize=14)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")


# ============================================================
# Compose the figure
# ============================================================
def main():
    sns.set_theme(style="whitegrid", context="talk")

    fig = plt.figure(figsize=(16.5, 11), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax_noe = fig.add_subplot(gs[0, :])     # top, full width
    ax_omega = fig.add_subplot(gs[1, 0])   # bottom left
    ax_cov = fig.add_subplot(gs[1, 1])     # bottom right

    plot_noe(ax_noe)
    plot_omega(ax_omega)
    plot_coverage(ax_cov)

    # Panel labels -- fixed offset in points from each panel's top-left corner,
    # so the label sits the same absolute distance out regardless of panel width.
    label_offset = ScaledTranslation(-34 / 72, 6 / 72, fig.dpi_scale_trans)
    for ax, label in [(ax_noe, "a"), (ax_omega, "b"), (ax_cov, "c")]:
        ax.text(0.0, 1.0, label, transform=ax.transAxes + label_offset,
                fontsize=22, fontweight="bold", va="bottom", ha="right")

    fig.savefig(SCRIPT_DIR / "Figure2.pdf", bbox_inches="tight")           # vector, for LaTeX
    fig.savefig(SCRIPT_DIR / "Figure2.png", dpi=300, bbox_inches="tight")  # preview only
    plt.show()


if __name__ == "__main__":
    main()
