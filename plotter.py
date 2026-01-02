# track changes
# LW 20250505

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

log_dir = Path('./')
log_name = 'log'
gap = 1

# Optional: set x or y limits. Set to None for auto scaling
xlim = (None, None)         # e.g., (0, 500)
ylim = (None, None)         # e.g., (0, 0.2)

# Define the plot configuration: {title: (x_col, y_col, y_factor)}
plot_config = {
    "train loss": (None, 5, 1),
    "mu": (None, 9, 1),
    "lr": (None, 11, 1e+3)
}

# Read the space- or tab-delimited log file
df = pd.read_csv(log_dir / log_name, sep=r'\s+', header=None, engine='python')

# Downsample the data
df_downsampled = df.iloc[::gap]
# print(len(df_downsampled))

# Plotting
plt.figure(figsize=(10, 6))
for label, (x_col, y_col, y_factor) in plot_config.items():
    x = df_downsampled.index if x_col is None else df_downsampled.iloc[:, x_col]
    y = df_downsampled.iloc[:, y_col] * y_factor
    plt.plot(x, y, label=f"{label} × {y_factor:g}")

plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title(log_name)
plt.legend()
plt.grid(True)
if xlim: plt.xlim(*xlim)
if ylim: plt.ylim(*ylim)
plt.tight_layout()
plt.show()
