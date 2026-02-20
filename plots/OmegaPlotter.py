import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# --- Data sources: label → pt file ---
pt_dict = {
    'Vacuum':  REPO_ROOT / 'pts' / 'omega_histogram_cycpeptmpdb_vacuum.pt',
    'H2O':     REPO_ROOT / 'pts' / 'omega_histogram_cycpeptmpdb_h2o.pt',
    'CHCl3':   REPO_ROOT / 'pts' / 'omega_histogram_cycpeptmpdb_chcl3.pt',
    'CREMP':   REPO_ROOT / 'pts' / 'omega_histogram_cremp.pt',
    'Hexane':  REPO_ROOT / 'pts' / 'omega_histogram_4d_hexane.pt',
    'Water':   REPO_ROOT / 'pts' / 'omega_histogram_4d_water.pt',
}

# --- Combined curves: combined_label → list of source labels ---
combine_groups = {
    'CycPeptMPDB-3D': ['Vacuum', 'H2O', 'CHCl3'],
    'CycPeptMPDB-4D': ['Hexane', 'Water'],
}

records = []
bin_edges_master = None
centers_wrapped_master = None
hist_by_label = {}

for label, path in pt_dict.items():
    d = torch.load(path)
    hist = d['hist_total'].float()          # (360,)
    edges = d['bin_edges'].float()          # (361,)

    # Master binning from the first file
    if bin_edges_master is None:
        bin_edges_master = edges.clone()
        centers = (bin_edges_master[:-1] + bin_edges_master[1:]) / 2.0
        centers_wrapped_master = torch.where(centers <= 270, centers, centers - 360)

    hist_norm = hist / hist.sum()
    hist_by_label[label] = hist_norm

    for x, y in zip(centers_wrapped_master.cpu().numpy(), hist_norm.cpu().numpy()):
        records.append({'angle_deg': x, 'prob': y, 'series': label})

# --- Build combined curves ---
for combined_label, source_labels in combine_groups.items():
    combined_hist = sum(hist_by_label[l] for l in source_labels)
    combined_hist = combined_hist / combined_hist.sum()
    for x, y in zip(centers_wrapped_master.cpu().numpy(), combined_hist.cpu().numpy()):
        records.append({'angle_deg': x, 'prob': y, 'series': combined_label})

# Tidy DataFrame and sort by angle for clean lines
df = pd.DataFrame.from_records(records).sort_values('angle_deg')

# --- Plot ---
sns.set_theme(style='whitegrid', context='talk')
plt.figure(figsize=(9, 5))

order = ['CycPeptMPDB-4D', 'CycPeptMPDB-3D', 'CREMP']
ax = sns.lineplot(
    data=df[df['series'].isin(order)],
    x='angle_deg', y='prob',
    hue='series',
    hue_order=order,
    linewidth=2
)

ax.set_title(r'Distribution of peptide dihedral angle $\omega$', fontsize=16)
ax.set_xlabel(r'$\omega$ (°)', fontsize=16)
ax.set_ylabel('Density', fontsize=16)  # you normalized to sum=1
ax.set_xlim(-90, 270)
ax.set_ylim(0, None)
ax.grid(alpha=0.3)
ax.legend(frameon=False, title=None)

# Strong black frame and ticks
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_edgecolor('black')
ax.tick_params(width=2, length=6, color='black', labelsize=15, direction='in')

plt.tight_layout()
plt.show()
