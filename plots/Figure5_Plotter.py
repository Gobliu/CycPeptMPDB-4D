import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.transforms import ScaledTranslation
from pathlib import Path

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CSV_PATH = REPO_ROOT / "csvs" / "CycPeptMPDB-4D.csv"

# --- Load data ---
df = pd.read_csv(CSV_PATH)

# --- Plot config ---
# SASA / NPSA / PSA columns are already stored in Å^2 in the CSV — no conversion.
colors = {"Hexane": "#4C72B0", "Water": "#C44E52"}
metrics = [
    ('3D_SASA', '3D-SASA'),
    ('3D_NPSA', '3D-NPSA'),
    ('3D_PSA',  '3D-PSA'),
]

# --- Subplots ---
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.5), sharey=True)

for i, (ax, (suffix, title)) in enumerate(zip(axes, metrics)):
    # KDEs
    sns.kdeplot(df[f'Hexane_{suffix}'].dropna(), color=colors["Hexane"], linewidth=2, label="in hexane", ax=ax)
    sns.kdeplot(df[f'Water_{suffix}'].dropna(),  color=colors["Water"],  linewidth=2, label="in water",  ax=ax)

    # Labels / titles
    ax.set_xlabel(r'Area (Å$^2$)', fontsize=16)
    ax.set_title(f'Distribution of {title}', fontsize=16)
    if i == 0:
        ax.set_ylabel('Density', fontsize=16)
    else:
        ax.set_ylabel('')

    # Grid
    ax.grid(alpha=0.3)

    # Tick styling (per axis)
    ax.tick_params(axis='both', labelsize=15, width=2, direction='in', pad=2)

    # Thicker frame
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')

axes[0].legend(frameon=False, fontsize=14)

# Bold panel letters at each panel's top-left corner
label_offset = ScaledTranslation(-25 / 72, 0 / 72, fig.dpi_scale_trans)
for ax, letter in zip(axes, 'abc'):
    ax.text(0.0, 1.0, letter, transform=ax.transAxes + label_offset,
            fontsize=22, fontweight='bold', va='bottom', ha='right')

plt.tight_layout()
fig.savefig(SCRIPT_DIR / "Figure5.pdf", bbox_inches='tight')           # vector, for LaTeX
fig.savefig(SCRIPT_DIR / "Figure5.png", dpi=300, bbox_inches='tight')  # preview only
plt.show()
