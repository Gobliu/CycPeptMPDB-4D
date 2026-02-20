import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# --- Load data ---
df = pd.read_csv(REPO_ROOT / 'csvs' / 'NOE.csv')

# Identify NaN rows in delta_dist
nan_mask = df['delta_dist'].isna()

# Fill NaN with 0 for plotting (so the data length stays consistent)
df['delta_dist'] = df['delta_dist'].fillna(0)
y = df['delta_dist'].values
x = np.arange(len(df))

# --- Find all NaN row indices, skip the first if it's at position 0 ---
nan_idx = np.where(nan_mask.values)[0].tolist()
if nan_idx and nan_idx[0] == 0:
    nan_idx = nan_idx[1:]  # drop the first NaN index

# --- Split into blocks (continuous non-NaN regions) ---
blocks = []
start = 0
for ni in nan_idx:
    if ni - start > 0:
        blocks.append((start, ni - 1))
    start = ni + 1
if start < len(df):
    blocks.append((start, len(df) - 1))

# --- Plot ---
plt.figure(figsize=(12, 5))

block_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
pdb_list = ['7L9D', '7L96', '7L98', '7UBG', '7UZL', '8CWA']

for i, (s, e) in enumerate(blocks):
    idx = np.arange(s, e + 1)
    plt.bar(idx, y[idx], color=block_colors[i], edgecolor='none', width=0.9, label=pdb_list[i])

# Horizontal zero line
plt.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.8)

# Set y-range and axis labels
plt.ylim(-3.2, 0.9)
plt.ylabel(r'$\mathrm{r_{traj}} - \mathrm{r_{NOE}}\ (\mathrm{Å})$', fontsize=16)
plt.xlabel('Unambiguous long distance NOE constraints', fontsize=16)  # Larger y-axis label font
# Remove x-axis ticks and labels entirely
plt.xticks([])  # removes tick marks and labels
plt.tick_params(axis='x', bottom=False, labelbottom=False)

# Make y-tick labels larger
plt.yticks(fontsize=16)
plt.tick_params(axis='y', width=2, direction='in', pad=2)
# Grid, legend, and frame styling
plt.grid(alpha=0.3, axis='y')
plt.legend(ncol=3, frameon=False, loc='upper right', fontsize=14)

# Thicken figure frame
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.show()
