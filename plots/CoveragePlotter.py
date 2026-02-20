import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# --- Load data ---
df = pd.read_csv(REPO_ROOT / 'csvs' / 'coverage.csv')

block_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
pdb_list = ['7L9D', '7L96', '7L98', '7UBG', '7UZL', '8CWA']

x = df.iloc[:, 0]

for i, pdb in enumerate(pdb_list):
    color = block_colors[i]
    plt.plot(x, df[pdb], color=color, linestyle='--', linewidth=1.5, label=pdb)

# Horizontal zero line
# plt.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.8)

# Set y-range and axis labels
plt.ylabel('Coverage', fontsize=16)
plt.xlabel('RMSD threshold (Å)', fontsize=16)  # Larger y-axis label font
# Remove x-axis ticks and labels entirely
# plt.xticks(np.linspace(0, 0.1, 11))  # removes tick marks and labels
# plt.tick_params(axis='x', bottom=False, labelbottom=False)

# Make y-tick labels larger
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.tick_params(axis='y', width=2, direction='in', pad=2)
plt.tick_params(axis='x', width=2, direction='in', pad=2)
# Grid, legend, and frame styling
plt.grid(alpha=0.3)
plt.legend(ncol=1, frameon=False, loc='upper left', fontsize=14)

# Thicken figure frame
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.show()
