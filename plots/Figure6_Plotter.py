#!/usr/bin/env python3
"""Figure 6: peptide-bond omega geometry and its distribution.

Layout:
    (a) cis / trans cartoon  -- left, loaded from Figure6a.png
    (b) omega distribution   -- right, CycPeptMPDB-4D vs 3D vs CREMP

Writes Figure6.png / Figure6.pdf alongside this script.
"""
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.transforms import ScaledTranslation

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PANEL_A_IMG = SCRIPT_DIR / "Figure6a.png"

# --- Data sources: label → pt file ---
PT_DICT = {
    'Vacuum':  REPO_ROOT / 'pts' / 'omega_histogram_cycpeptmpdb_vacuum.pt',
    'H2O':     REPO_ROOT / 'pts' / 'omega_histogram_cycpeptmpdb_h2o.pt',
    'CHCl3':   REPO_ROOT / 'pts' / 'omega_histogram_cycpeptmpdb_chcl3.pt',
    'CREMP':   REPO_ROOT / 'pts' / 'omega_histogram_cremp.pt',
    'Hexane':  REPO_ROOT / 'pts' / 'omega_histogram_4d_hexane.pt',
    'Water':   REPO_ROOT / 'pts' / 'omega_histogram_4d_water.pt',
}

# --- Combined curves: combined_label → list of source labels ---
COMBINE_GROUPS = {
    'CycPeptMPDB-3D': ['Vacuum', 'H2O', 'CHCl3'],
    'CycPeptMPDB-4D': ['Hexane', 'Water'],
}
SERIES_ORDER = ['CycPeptMPDB-4D', 'CycPeptMPDB-3D', 'CREMP']


# ============================================================
# Compose the figure
# ============================================================
def main():
    sns.set_theme(style='whitegrid', context='talk')

    fig = plt.figure(figsize=(16.5, 5.5), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.7])
    ax_img = fig.add_subplot(gs[0, 0])   # cis/trans cartoon
    ax_omega = fig.add_subplot(gs[0, 1])  # omega distribution

    plot_panel_a(ax_img)
    plot_omega(ax_omega)

    # Bold panel letters at each panel's top-left corner
    label_offset = ScaledTranslation(-34 / 72, 6 / 72, fig.dpi_scale_trans)
    for ax, letter in [(ax_img, 'a'), (ax_omega, 'b')]:
        ax.text(0.0, 1.0, letter, transform=ax.transAxes + label_offset,
                fontsize=22, fontweight='bold', va='bottom', ha='right')

    fig.savefig(SCRIPT_DIR / "Figure6.pdf", bbox_inches='tight')           # vector, for LaTeX
    fig.savefig(SCRIPT_DIR / "Figure6.png", dpi=300, bbox_inches='tight')  # preview only
    plt.show()


def plot_panel_a(ax):
    """Left panel: the cis/trans omega cartoon image."""
    ax.imshow(mpimg.imread(PANEL_A_IMG))
    ax.set_title("Peptide bond geometry", fontsize=16)
    ax.axis('off')


def plot_omega(ax):
    """Right panel: omega distribution for 4D vs 3D vs CREMP."""
    df = build_omega_dataframe()

    sns.lineplot(
        data=df[df['series'].isin(SERIES_ORDER)],
        x='angle_deg', y='prob',
        hue='series', hue_order=SERIES_ORDER,
        linewidth=2, ax=ax,
    )

    ax.set_title(r'Distribution of peptide dihedral angle $\omega$', fontsize=16)
    ax.set_xlabel(r'$\omega$ (°)', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.set_xlim(-90, 270)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, title=None, fontsize=14)

    # cis / trans labels next to their peaks
    ax.text(40, 0.046, 'Cis', fontsize=16, fontweight='bold', ha='left', va='center')
    ax.text(200, 0.046, 'Trans', fontsize=16, fontweight='bold', ha='left', va='center')

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')
    ax.tick_params(width=2, length=6, color='black', labelsize=15, direction='in')


def build_omega_dataframe():
    """Load per-source omega histograms, normalise, and build combined curves."""
    centers_wrapped = None
    hist_by_label = {}
    records = []

    for label, path in PT_DICT.items():
        d = torch.load(path)
        hist = d['hist_total'].float()      # (360,)
        edges = d['bin_edges'].float()      # (361,)

        if centers_wrapped is None:
            centers = (edges[:-1] + edges[1:]) / 2.0
            centers_wrapped = torch.where(centers <= 270, centers, centers - 360)

        hist_norm = hist / hist.sum()
        hist_by_label[label] = hist_norm
        for x, y in zip(centers_wrapped.cpu().numpy(), hist_norm.cpu().numpy()):
            records.append({'angle_deg': x, 'prob': y, 'series': label})

    for combined_label, source_labels in COMBINE_GROUPS.items():
        combined = sum(hist_by_label[l] for l in source_labels)
        combined = combined / combined.sum()
        for x, y in zip(centers_wrapped.cpu().numpy(), combined.cpu().numpy()):
            records.append({'angle_deg': x, 'prob': y, 'series': combined_label})

    return pd.DataFrame.from_records(records).sort_values('angle_deg')


if __name__ == "__main__":
    main()
