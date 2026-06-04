#!/usr/bin/env python3
"""
Re-plot per-peptide Ramachandran maps using only MD vs NMR coloring
(no residue-type subdivision). Reads dihedrals.csv produced by
compute_dihedrals.py.
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

OUT = "/cluster/scratch/JG2LW/validation/dihedral_analysis"
PEPTIDES = ["7l96", "7l98", "7l9d", "7ubg", "7uzl", "8cwa"]

# Load CSV
data = {pid: {"MD": ([], []), "NMR": ([], [])} for pid in PEPTIDES}
with open(os.path.join(OUT, "dihedrals.csv")) as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        pid = row["peptide"]
        src = row["source"]
        if not row["phi"] or not row["psi"]:
            continue
        data[pid][src][0].append(float(row["phi"]))
        data[pid][src][1].append(float(row["psi"]))

MD_COLOR  = "#1f77b4"   # blue
NMR_COLOR = "#d62728"   # red

for pid in PEPTIDES:
    fig, ax = plt.subplots(figsize=(6, 6))
    md_x, md_y   = data[pid]["MD"]
    nmr_x, nmr_y = data[pid]["NMR"]
    ax.scatter(md_x, md_y, s=14, c=MD_COLOR, alpha=0.35,
               edgecolors="none", label=f"MD (n={len(md_x)})")
    ax.scatter(nmr_x, nmr_y, s=60, facecolors="none",
               edgecolors=NMR_COLOR, linewidths=1.6,
               label=f"NMR (n={len(nmr_x)})")
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 90))
    ax.set_yticks(np.arange(-180, 181, 90))
    ax.axhline(0, color="lightgrey", lw=0.5)
    ax.axvline(0, color="lightgrey", lw=0.5)
    ax.set_xlabel(r"$\phi$ (deg)")
    ax.set_ylabel(r"$\psi$ (deg)")
    ax.set_aspect("equal")
    ax.set_title(f"{pid}: Ramachandran  (MD vs NMR)")
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    fig.tight_layout()
    out = os.path.join(OUT, f"{pid}_ramachandran_MDvsNMR.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

print("Done.")
