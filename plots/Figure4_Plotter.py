import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.transforms import ScaledTranslation
from pathlib import Path

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CSV_PATH = REPO_ROOT / "csvs" / "CycPeptMPDB-4D.csv"

# --- Load data ---
raw = pd.read_csv(CSV_PATH)

# --- Reshape wide → long ---
records = []
for env, all_col, bb_col, gr_col in [
    ('Hexane', 'Hexane_avgRMSD_All', 'Hexane_avgRMSD_BackBone', 'Hexane_avgGR'),
    ('Water',  'Water_avgRMSD_All',  'Water_avgRMSD_BackBone',  'Water_avgGR'),
]:
    tmp = raw[['Monomer_Length', all_col, bb_col, gr_col]].copy()
    tmp.columns = ['Monomer_Length', 'avgRMSD_all', 'avgRMSD_bb', 'avgGR']
    tmp['Env'] = env
    records.append(tmp)

df = pd.concat(records, ignore_index=True)

# avgRMSD and avgGR are already stored in Å in the CSV — no unit conversion.

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
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.5))
metrics = [('avgRMSD_all', 'All heavy atoms',      (0, 3.5),   'RMSD (Å)'),
           ('avgRMSD_bb',  'Backbone heavy atoms', (0, 2.5),   'RMSD (Å)'),
           ('avgGR',       'Radius of gyration',   (4.0, 6.7), r'$R_g$ (Å)')]

for ax, (metric, title, xlim, xlabel) in zip(axes, metrics):
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
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=15, width=2, direction='in', pad=2)

    # Hide the first (origin) tick label on both axes to avoid overlap
    ax.get_xticklabels()[0].set_visible(False)
    ax.get_yticklabels()[0].set_visible(False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')

# Y-label only on left
axes[0].set_ylabel('Density', fontsize=16)
axes[1].set_ylabel('')
axes[2].set_ylabel('')

# Legend only on the middle (backbone heavy atoms) panel
axes[1].legend(frameon=False, ncol=1, loc='upper right', fontsize=14)

# Bold panel letters at each panel's top-left corner
label_offset = ScaledTranslation(-25 / 72, 0 / 72, fig.dpi_scale_trans)
for ax, letter in zip(axes, 'abc'):
    ax.text(0.0, 1.0, letter, transform=ax.transAxes + label_offset,
            fontsize=22, fontweight='bold', va='bottom', ha='right')

plt.tight_layout()
fig.savefig(SCRIPT_DIR / "Figure4.pdf", bbox_inches='tight')           # vector, for LaTeX
fig.savefig(SCRIPT_DIR / "Figure4.png", dpi=300, bbox_inches='tight')  # preview only
plt.show()
