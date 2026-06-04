#!/usr/bin/env python3
"""Combined 2x3 grid: MD vs NMR Ramachandran for all 6 peptides."""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

OUT = "/cluster/scratch/JG2LW/validation/dihedral_analysis"
PEPTIDES = ["7l96", "7l98", "7l9d", "7ubg", "7uzl", "8cwa"]
MD_COLOR  = "#1f77b4"
NMR_COLOR = "#d62728"

data = {pid: {"MD": ([], []), "NMR": ([], [])} for pid in PEPTIDES}
with open(os.path.join(OUT, "dihedrals.csv")) as fh:
    for row in csv.DictReader(fh):
        if not row["phi"] or not row["psi"]:
            continue
        data[row["peptide"]][row["source"]][0].append(float(row["phi"]))
        data[row["peptide"]][row["source"]][1].append(float(row["psi"]))

fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
for ax, pid in zip(axes.flat, PEPTIDES):
    md_x, md_y   = data[pid]["MD"]
    nmr_x, nmr_y = data[pid]["NMR"]
    ax.scatter(md_x, md_y, s=12, c=MD_COLOR, alpha=0.35, edgecolors="none",
               label=f"MD (n={len(md_x)})")
    ax.scatter(nmr_x, nmr_y, s=55, facecolors="none",
               edgecolors=NMR_COLOR, linewidths=1.5,
               label=f"NMR (n={len(nmr_x)})")
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 90))
    ax.set_yticks(np.arange(-180, 181, 90))
    ax.axhline(0, color="lightgrey", lw=0.5)
    ax.axvline(0, color="lightgrey", lw=0.5)
    ax.set_aspect("equal")
    ax.set_title(pid, fontsize=13)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

for ax in axes[-1, :]:
    ax.set_xlabel(r"$\phi$ (deg)")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\psi$ (deg)")

fig.suptitle("Ramachandran: MD vs NMR  (all 6 cyclic peptides)",
             fontsize=15, y=0.995)
fig.tight_layout()
out = os.path.join(OUT, "combined_MDvsNMR.png")
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out}")
