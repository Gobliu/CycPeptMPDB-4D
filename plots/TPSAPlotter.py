import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CSV_PATH = REPO_ROOT / "csvs" / "CycPeptMPDB-4D_clean.csv"

# --- Load data ---
df = pd.read_csv(CSV_PATH)

# --- Plot config ---
colors = {"Hexane": "#4C72B0", "Water": "#C44E52"}
metrics = [
    ('3D_SASA', '3D-SASA'),
    ('3D_NPSA', '3D-NPSA'),
    ('3D_PSA',  '3D-PSA'),
]

# --- Subplots ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for i, (ax, (suffix, title)) in enumerate(zip(axes, metrics)):
    # KDEs
    sns.kdeplot(df[f'Hexane_{suffix}'].dropna(), color=colors["Hexane"], linewidth=2, label="in hexane", ax=ax)
    sns.kdeplot(df[f'Water_{suffix}'].dropna(),  color=colors["Water"],  linewidth=2, label="in water",  ax=ax)

    # Labels / titles
    ax.set_xlabel(r'Area (nm$^2$)', fontsize=14)
    ax.set_title(f'Distribution of {title}', fontsize=16)
    if i == 0:
        ax.set_ylabel('Density', fontsize=14)
    else:
        ax.set_ylabel('')

    # Grid
    ax.grid(alpha=0.3)

    # Tick styling (per axis)
    ax.tick_params(axis='both', labelsize=16, width=2, direction='in', pad=2)

    # Thicker frame
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

axes[0].legend(frameon=False, fontsize=14)

plt.tight_layout()
plt.show()
