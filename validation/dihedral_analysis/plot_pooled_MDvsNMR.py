#!/usr/bin/env python3
"""Pooled Ramachandran: merge dihedrals from all 6 peptides into one plot."""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

OUT = "/cluster/scratch/JG2LW/validation/dihedral_analysis"
MD_COLOR  = "#1f77b4"
NMR_COLOR = "#d62728"

md_phi, md_psi   = [], []
nmr_phi, nmr_psi = [], []

with open(os.path.join(OUT, "dihedrals.csv")) as fh:
    for row in csv.DictReader(fh):
        if not row["phi"] or not row["psi"]:
            continue
        phi, psi = float(row["phi"]), float(row["psi"])
        if row["source"] == "MD":
            md_phi.append(phi); md_psi.append(psi)
        else:
            nmr_phi.append(phi); nmr_psi.append(psi)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(md_phi, md_psi, s=10, c=MD_COLOR, alpha=0.25,
           edgecolors="none", label=f"MD  (n={len(md_phi)})")
ax.scatter(nmr_phi, nmr_psi, s=45, facecolors="none",
           edgecolors=NMR_COLOR, linewidths=1.2,
           label=f"NMR (n={len(nmr_phi)})")
ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
ax.set_xticks(np.arange(-180, 181, 90))
ax.set_yticks(np.arange(-180, 181, 90))
ax.axhline(0, color="lightgrey", lw=0.5)
ax.axvline(0, color="lightgrey", lw=0.5)
ax.set_xlabel(r"$\phi$ (deg)", fontsize=12)
ax.set_ylabel(r"$\psi$ (deg)", fontsize=12)
ax.set_aspect("equal")
ax.set_title("Pooled Ramachandran: 6 cyclic peptides, MD vs NMR",
             fontsize=13)
ax.legend(loc="upper right", fontsize=10, frameon=True)
fig.tight_layout()
out = os.path.join(OUT, "pooled_MDvsNMR.png")
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out}  (MD={len(md_phi)}, NMR={len(nmr_phi)})")
