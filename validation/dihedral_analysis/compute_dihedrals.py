#!/usr/bin/env python3
"""
Compute backbone dihedrals (phi, psi, omega) for cyclic peptides and produce
Ramachandran plots comparing MD and NMR ensembles.

Output:
  - dihedrals.csv         (raw per-model, per-residue values)
  - <pdb>_overlay.png     (MD vs NMR overlay, one per peptide)
  - global_overlay.png    (all peptides combined, MD vs NMR)
  - omega_distribution.png (omega cis/trans histogram)
"""

import csv
import os
from collections import OrderedDict
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Residue classification
# ---------------------------------------------------------------------------
L_STANDARD = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
              "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "SER", "THR",
              "TRP", "TYR", "VAL"}
L_PROLINE = {"PRO"}
D_AMINO = {"DAL", "DAR", "DAN", "DAS", "DCY", "DGN", "DGL", "DHI", "DIL",
           "DLE", "DLY", "DPN", "DPR", "DSG", "DSN", "DTH", "DTR", "DTY",
           "DVA", "MED", "DM0"}
N_METHYL = {"MAA", "MEA", "MLE", "NLE", "MVA", "MEI", "NMA", "MEG", "MEN",
            "MEM", "MEF", "MEQ", "MEW", "MEY", "MEH", "ABA"}


def classify(resname):
    rn = resname.upper()
    if rn in L_PROLINE:
        return "L-Pro"
    if rn in L_STANDARD:
        return "L-AA"
    if rn in D_AMINO:
        if rn == "DPR":
            return "D-Pro"
        return "D-AA"
    if rn in N_METHYL:
        return "N-Me-AA"
    return "Other"


CLASS_STYLE = {
    "L-AA":    dict(color="#1f77b4", marker="o"),
    "L-Pro":   dict(color="#2ca02c", marker="s"),
    "D-AA":    dict(color="#d62728", marker="^"),
    "D-Pro":   dict(color="#9467bd", marker="v"),
    "N-Me-AA": dict(color="#ff7f0e", marker="D"),
    "Other":   dict(color="#7f7f7f", marker="x"),
}

# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------
def parse_pdb_models(path):
    """Return list of models. Each model: OrderedDict[resnum] -> dict.
    resnum dict: {'resname': str, 'atoms': {atom_name: (x,y,z)}}
    """
    models = []
    current = None
    has_model_tag = False
    with open(path) as fh:
        for line in fh:
            if line.startswith("MODEL"):
                has_model_tag = True
                if current is not None:
                    models.append(current)
                current = OrderedDict()
            elif line.startswith("ENDMDL"):
                if current is not None:
                    models.append(current)
                current = None
            elif line.startswith(("ATOM", "HETATM")):
                if current is None:
                    current = OrderedDict()
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                try:
                    resnum = int(line[22:26])
                except ValueError:
                    continue
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except ValueError:
                    continue
                if resnum not in current:
                    current[resnum] = {"resname": resname, "atoms": {}}
                current[resnum]["atoms"][atom_name] = (x, y, z)
    if current is not None and (not has_model_tag or len(current) > 0):
        # Trailing model with no ENDMDL or single-model file
        if not models or current is not models[-1]:
            models.append(current)
    # Filter out empty models
    models = [m for m in models if m]
    return models


# ---------------------------------------------------------------------------
# Dihedral computation
# ---------------------------------------------------------------------------
def dihedral(p1, p2, p3, p4):
    p1 = np.asarray(p1); p2 = np.asarray(p2)
    p3 = np.asarray(p3); p4 = np.asarray(p4)
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return np.degrees(np.arctan2(y, x))


def compute_dihedrals_for_model(model):
    """Assume cyclic peptide: residue 1 connects to last residue."""
    resnums = list(model.keys())
    n = len(resnums)
    results = []
    for i, rn in enumerate(resnums):
        rec = model[rn]
        atoms = rec["atoms"]
        if not all(k in atoms for k in ("N", "CA", "C")):
            results.append((rn, rec["resname"], None, None, None))
            continue
        prev_rn = resnums[(i - 1) % n]
        next_rn = resnums[(i + 1) % n]
        prev_atoms = model[prev_rn]["atoms"]
        next_atoms = model[next_rn]["atoms"]

        phi = psi = omega = None
        if "C" in prev_atoms:
            phi = dihedral(prev_atoms["C"], atoms["N"], atoms["CA"], atoms["C"])
        if "N" in next_atoms:
            psi = dihedral(atoms["N"], atoms["CA"], atoms["C"], next_atoms["N"])
        if "CA" in prev_atoms and "C" in prev_atoms:
            omega = dihedral(prev_atoms["CA"], prev_atoms["C"], atoms["N"], atoms["CA"])
        results.append((rn, rec["resname"], phi, psi, omega))
    return results


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
ROOT = "/cluster/scratch/JG2LW/validation"
OUT = os.path.join(ROOT, "dihedral_analysis")
os.makedirs(OUT, exist_ok=True)

PEPTIDES = ["7l96", "7l98", "7l9d", "7ubg", "7uzl", "8cwa"]

# Collect all data
records = []  # (peptide, source, model_idx, resnum, resname, restype, phi, psi, omega)

for pid in PEPTIDES:
    for source, sub, suffix in [("NMR", "NMR", ".pdb"),
                                ("MD",  "MD",  "_100frames_align.pdb")]:
        path = os.path.join(ROOT, sub, f"{pid}{suffix}")
        if not os.path.exists(path):
            print(f"[WARN] missing {path}")
            continue
        models = parse_pdb_models(path)
        print(f"{pid} {source}: {len(models)} models")
        for mi, model in enumerate(models):
            for rn, resname, phi, psi, omega in compute_dihedrals_for_model(model):
                records.append((pid, source, mi, rn, resname,
                                classify(resname), phi, psi, omega))

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------
csv_path = os.path.join(OUT, "dihedrals.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["peptide", "source", "model", "resnum", "resname",
                "restype", "phi", "psi", "omega"])
    for r in records:
        row = list(r)
        for i in (6, 7, 8):
            row[i] = "" if row[i] is None else f"{row[i]:.3f}"
        w.writerow(row)
print(f"Wrote {csv_path} ({len(records)} rows)")

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def draw_axes(ax):
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 90))
    ax.set_yticks(np.arange(-180, 181, 90))
    ax.axhline(0, color="lightgrey", lw=0.5)
    ax.axvline(0, color="lightgrey", lw=0.5)
    ax.set_xlabel(r"$\phi$ (deg)")
    ax.set_ylabel(r"$\psi$ (deg)")
    ax.set_aspect("equal")


def legend_handles():
    handles = []
    for label, style in CLASS_STYLE.items():
        handles.append(Line2D([0], [0], marker=style["marker"],
                              color="w", markerfacecolor=style["color"],
                              markeredgecolor=style["color"],
                              markersize=7, label=label, linestyle=""))
    handles.append(Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="none", markeredgecolor="black",
                          markersize=9, label="NMR (edge)", linestyle=""))
    handles.append(Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="grey", markersize=4,
                          alpha=0.4, label="MD (filled)", linestyle=""))
    return handles


def scatter_pts(ax, subset, source):
    """subset: list of (restype, phi, psi). source: 'MD' or 'NMR'."""
    by_type = {}
    for restype, phi, psi in subset:
        if phi is None or psi is None:
            continue
        by_type.setdefault(restype, ([], []))
        by_type[restype][0].append(phi)
        by_type[restype][1].append(psi)
    for restype, (xs, ys) in by_type.items():
        st = CLASS_STYLE.get(restype, CLASS_STYLE["Other"])
        if source == "MD":
            ax.scatter(xs, ys, s=12, c=st["color"], marker=st["marker"],
                       alpha=0.35, edgecolors="none")
        else:  # NMR
            ax.scatter(xs, ys, s=55, facecolors="none",
                       edgecolors=st["color"], marker=st["marker"],
                       linewidths=1.4)


# ---------------------------------------------------------------------------
# Per-peptide overlay plots
# ---------------------------------------------------------------------------
for pid in PEPTIDES:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    md_pts = [(r[5], r[6], r[7]) for r in records
              if r[0] == pid and r[1] == "MD"]
    nmr_pts = [(r[5], r[6], r[7]) for r in records
               if r[0] == pid and r[1] == "NMR"]
    scatter_pts(ax, md_pts, "MD")
    scatter_pts(ax, nmr_pts, "NMR")
    draw_axes(ax)
    ax.set_title(f"{pid}: Ramachandran (MD filled, NMR open)")
    ax.legend(handles=legend_handles(), loc="upper left",
              bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)
    fig.tight_layout()
    out = os.path.join(OUT, f"{pid}_ramachandran.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

# ---------------------------------------------------------------------------
# Global summary: 2x3 grid
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, pid in zip(axes.flat, PEPTIDES):
    md_pts = [(r[5], r[6], r[7]) for r in records
              if r[0] == pid and r[1] == "MD"]
    nmr_pts = [(r[5], r[6], r[7]) for r in records
               if r[0] == pid and r[1] == "NMR"]
    scatter_pts(ax, md_pts, "MD")
    scatter_pts(ax, nmr_pts, "NMR")
    draw_axes(ax)
    ax.set_title(pid)
fig.suptitle("Ramachandran (MD filled vs NMR open) — all peptides",
             fontsize=14, y=1.00)
fig.legend(handles=legend_handles(), loc="center right",
           bbox_to_anchor=(1.08, 0.5), fontsize=9, frameon=False)
fig.tight_layout()
out = os.path.join(OUT, "global_overlay.png")
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out}")

# ---------------------------------------------------------------------------
# Omega distribution (cis/trans) histogram per source
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, src in zip(axes, ["MD", "NMR"]):
    omegas = [r[8] for r in records if r[1] == src and r[8] is not None]
    ax.hist(omegas, bins=np.arange(-180, 181, 5),
            color="#4C72B0", edgecolor="black", linewidth=0.4)
    ax.set_xlim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_xlabel(r"$\omega$ (deg)")
    ax.set_title(f"{src} (n={len(omegas)})")
    ax.axvline(180, color="red", lw=0.6, ls="--")
    ax.axvline(-180, color="red", lw=0.6, ls="--")
    ax.axvline(0, color="orange", lw=0.6, ls="--")
axes[0].set_ylabel("Count")
fig.suptitle(r"$\omega$ distribution (trans $\approx \pm 180$°, cis $\approx 0$°)")
fig.tight_layout()
out = os.path.join(OUT, "omega_distribution.png")
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out}")

print("Done.")
