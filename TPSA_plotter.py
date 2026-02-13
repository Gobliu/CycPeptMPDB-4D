import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load data ---
hex_df = pd.read_csv('./csvs/all_hexane_sa.dat', delim_whitespace=True)
wat_df = pd.read_csv('./csvs/all_water_sa_clean.dat', delim_whitespace=True)

# --- Plot config ---
colors = {"Hexane": "#4C72B0", "Water": "#C44E52"}
cols_to_plot = [2, 3, 4]                # 0-based: plots columns 3, 4, 5 in the file
titles = ['3D-SASA', '3D-NPSA', '3D-PSA']

# --- Subplots ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for i, (ax, col_idx) in enumerate(zip(axes, cols_to_plot)):
    # KDEs
    sns.kdeplot(hex_df.iloc[:, col_idx], color=colors["Hexane"], linewidth=2, label="in hexane", ax=ax)
    sns.kdeplot(wat_df.iloc[:, col_idx], color=colors["Water"], linewidth=2, label="in water",  ax=ax)

    # Labels / titles
    ax.set_xlabel(r'Area (nm$^2$)', fontsize=14)
    ax.set_title(f'Distribution of {titles[i]}', fontsize=16)
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