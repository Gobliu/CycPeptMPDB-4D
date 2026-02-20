import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CSV_PATH = REPO_ROOT / "csvs" / "CycPeptMPDB-4D.csv"

# --- Load data ---
raw = pd.read_csv(CSV_PATH)

# --- Reshape wide → long ---
records = []
for env, all_col, bb_col in [
    ('Hexane', 'Hexane_avgRMSD_All', 'Hexane_avgRMSD_BackBone'),
    ('Water',  'Water_avgRMSD_All',  'Water_avgRMSD_BackBone'),
]:
    tmp = raw[['Monomer_Length', all_col, bb_col]].copy()
    tmp.columns = ['Monomer_Length', 'avgRMSD_all', 'avgRMSD_bb']
    tmp['Env'] = env
    records.append(tmp)

df = pd.concat(records, ignore_index=True)

# --- Convert nm → Å ---
for col in ['avgRMSD_all', 'avgRMSD_bb']:
    df[col] = df[col] * 10

df['Monomer_Length'] = df['Monomer_Length'].astype('Int64')
df['LengthLabel'] = df['Monomer_Length'].astype(str) + '-mer'

# --- Seaborn style ---
sns.set_theme(style="whitegrid", context="talk")

sorted_ints = sorted(df['Monomer_Length'].dropna().unique())
lengths = [str(i) + '-mer' for i in sorted_ints]

palette = sns.color_palette("muted", len(lengths))
length_color = dict(zip(lengths, palette))
env_dashes = {'Hexane': (1, 0), 'Water': (4, 2)}  # solid vs dashed

# --- Create figure ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
metrics = [('avgRMSD_all', 'All atoms', (0, 3.5)),
           ('avgRMSD_bb', 'Main-chain heavy atoms', (0, 2.5))]

for ax, (metric, title, xlim) in zip(axes, metrics):
    sub = df.dropna(subset=[metric, 'Monomer_Length'])
    for L in lengths:
        subL = sub[sub['LengthLabel'] == L]
        for env in ['Hexane', 'Water']:
            subLE = subL[subL['Env'] == env]
            if subLE.empty:
                continue
            sns.kdeplot(
                subLE[metric],
                ax=ax,
                color=length_color[L],
                lw=2,
                dashes=env_dashes[env],
                label=f'{L} (in {env.lower()})'
            )

    ax.set_xlim(xlim)
    ax.set_xlabel('RMSD (Å)', fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=15, width=2, direction='in', pad=2)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')

# Y-label only on left
axes[0].set_ylabel('Density', fontsize=16)
axes[1].set_ylabel('')

# Legend only on backbone plot
axes[1].legend(frameon=False, ncol=1, loc='upper right', fontsize=14)

plt.tight_layout()
plt.show()
