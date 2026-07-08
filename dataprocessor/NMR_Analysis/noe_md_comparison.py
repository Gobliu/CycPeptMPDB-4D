#!/usr/bin/env python3
"""
NOE vs MD distance comparison for CycPeptMPDB-4D validation peptides.

For each peptide in ./CHCl3/:
  - parse the deposited NMR restraint file (`{PID}.mr`, either XPLOR-CNS
    or CYANA .upl format) and the MD trajectory (`{PID}_CHCl3_Traj.pdb`)
  - keep only unambiguous single-atom NOE pairs
  - compute the trajectory-averaged distance <r^-6>^{-1/6} plus the
    5th and 95th percentile of the per-frame distance distribution
  - write a combined CSV table

Categories: S/M/W/vW are taken from the XPLOR comment field when present,
otherwise binned from the upper bound (<=3.0 -> S; <=3.5 -> M; <=5.0 -> W; else vW).

Paths are script-relative:
    restraints    <script_dir>/CHCl3/{PID}.mr             (this folder)
    trajectories  <Data>/CycPeptMPDB_4D/June2026/CHCl3/{PID}_CHCl3_Traj.pdb
    output        <repo>/csvs/all_NOE_MD_comparison.csv
"""

from __future__ import annotations
import os, re, csv
from pathlib import Path
from collections import defaultdict
import numpy as np

XPLOR_PEPTIDES = ["7L9D", "7L96", "7L98"]
CYANA_PEPTIDES = ["7UBG", "7UZL", "8CWA"]

SCRIPT_DIR = Path(__file__).resolve().parent               # dataprocessor/NMR_Analysis
REPO_ROOT = SCRIPT_DIR.parent.parent                       # repo root
DATA_DIR = REPO_ROOT.parent / "Data" / "CycPeptMPDB_4D"    # sibling Data dir
RESTRAINT_DIR = SCRIPT_DIR / "CHCl3"                        # {PID}.mr / {pid}_nmr-data.str
TRAJ_DIR = DATA_DIR / "June2026" / "CHCl3"                  # {PID}_CHCl3_Traj.pdb
OUT_CSV = REPO_ROOT / "csvs" / "all_NOE_MD_comparison.csv"

# ---------------------------------------------------------------- trajectory
def parse_traj(path: str) -> list[dict]:
    """Parse a multi-MODEL PDB trajectory. Returns list of {(resnum, atomname): xyz}."""
    frames = []
    cur = None
    with open(path) as f:
        for l in f:
            if l.startswith("MODEL"):
                cur = {}
                continue
            if l.startswith("ENDMDL"):
                if cur is not None:
                    frames.append(cur)
                    cur = None
                continue
            if not l.startswith("ATOM"):
                continue
            aname = l[12:16].strip()
            rnum  = int(l[22:26])
            try:
                x = float(l[30:38]); y = float(l[38:46]); z = float(l[46:54])
            except ValueError:
                continue
            cur[(rnum, aname)] = np.array([x, y, z])
        if cur is not None:
            frames.append(cur)
    return frames


def residue_atom_map(sample_frame):
    atoms = defaultdict(set)
    for (r, a) in sample_frame:
        atoms[r].add(a)
    return atoms


# ---------------------------------------------------------------- atom mapping (.mr -> traj)
def to_traj_atom(rnum, aname, res_atoms):
    """Map a deposited atom name to a single trajectory atom, or None."""
    ra = res_atoms.get(rnum, set())
    if not ra:
        return None
    # canonical backbone amide labels
    if aname == "HN":
        return "H" if "H" in ra else ("HN" if "HN" in ra else None)
    if aname == "H1":  # ITZ non-canonical amide
        return "H" if "H" in ra else ("H1" if "H1" in ra else None)
    # direct match
    if aname in ra:
        return aname
    # Wuethrich -> IUPAC prochiral rename (HB1/HB2 -> HB2/HB3, etc.)
    prochiral = {
        "HB": ("HB1", "HB2", "HB3"),
        "HG": ("HG1", "HG2", "HG3"),
        "HD": ("HD1", "HD2", "HD3"),
    }
    for h1, h2, h3 in prochiral.values():
        if (h1 not in ra) and (h2 in ra) and (h3 in ra):
            if aname == h1: return h2
            if aname == h2: return h3
    return None


# ---------------------------------------------------------------- XPLOR-CNS parser
XPLOR_NOE_RE = re.compile(
    r"assi\s*\(\s*resi\s+(\d+)\s+and\s+name\s+(\S+?)\s*\)\s*"
    r"\(\s*resi\s+(\d+)\s+and\s+name\s+(\S+?)\s*\)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*(?:!\s*(.*))?",
    re.IGNORECASE,
)

def cat_from_intensity(n: int | None) -> str | None:
    if n is None: return None
    if n >= 9: return "S"
    if n >= 6: return "M"
    if n >= 3: return "W"
    return "vW"

def parse_xplor(path):
    """Parse XPLOR-CNS `assi(resi N and name X)(resi N and name X) 0.0 0.0 up ! comment` lines."""
    out = []
    in_hbond = False
    with open(path) as f:
        for l in f:
            if "H-Bond" in l or "H-BOND" in l:
                in_hbond = True
            if in_hbond:
                continue
            if not l.lstrip().lower().startswith("assi"):
                continue
            m = XPLOR_NOE_RE.match(l.strip())
            if not m:
                continue
            r1, a1, r2, a2, lo, tol, up, com = m.groups()
            com = (com or "").strip()
            ltr = re.search(r"\b(S|M|W|vw|VW|Vw|vW)\b", com)
            letter = ltr.group(1).upper() if ltr else None
            if letter == "VW":
                letter = "vW"
            num = re.search(r"(?:\()?(\d{1,2})(?:\))?", com)
            intensity = int(num.group(1)) if num else None
            cat = letter if letter else cat_from_intensity(intensity)
            out.append(dict(r1=int(r1), a1=a1, r2=int(r2), a2=a2,
                            upper=float(up), category=cat,
                            intensity=intensity, comment=com))
    return out


# ---------------------------------------------------------------- CYANA parser
def parse_cyana(path):
    """Parse CYANA `.upl` format restraints: `r1 rn1 a1 r2 rn2 a2 upper`."""
    out = []
    lines = open(path).read().splitlines()
    noe_start = None
    for i, l in enumerate(lines):
        if re.search(r"noe.*\.upl", l, re.I):
            noe_start = i
            break
    if noe_start is None:
        return out
    for l in lines[noe_start:]:
        s = l.strip()
        if not s or s.startswith("#") or s.startswith("###"):
            continue
        parts = re.split(r"\s+", s)
        if len(parts) < 7 or not parts[0].isdigit() or not parts[3].isdigit():
            continue
        r1 = int(parts[0]); rn1 = parts[1]; a1 = parts[2]
        r2 = int(parts[3]); rn2 = parts[4]; a2 = parts[5]
        try:
            upper = float(parts[6])
        except ValueError:
            continue
        if upper <= 3.0: cat = "S"
        elif upper <= 3.5: cat = "M"
        elif upper <= 5.0: cat = "W"
        else: cat = "vW"
        out.append(dict(r1=r1, a1=a1, r2=r2, a2=a2, upper=upper,
                        category=cat, intensity=None,
                        comment=f"{rn1} - {rn2}"))
    return out


# ---------------------------------------------------------------- distance math
def r6_avg(distances):
    """<r^-6>^{-1/6} average -- the physically appropriate NOE average."""
    d = np.asarray(distances, dtype=float)
    if len(d) == 0:
        return None
    return (np.mean(d ** -6.0)) ** (-1.0 / 6.0)


# ---------------------------------------------------------------- main
def analyze_peptide(pid, restraint_dir, traj_dir):
    traj_path = os.path.join(traj_dir, f"{pid}_CHCl3_Traj.pdb")
    mr_path   = os.path.join(restraint_dir, f"{pid}.mr")
    if not os.path.exists(traj_path) or not os.path.exists(mr_path):
        return []
    frames = parse_traj(traj_path)
    sample = frames[0]
    res_atoms = residue_atom_map(sample)
    src = "XPLOR" if pid in XPLOR_PEPTIDES else "CYANA"
    restraints = parse_xplor(mr_path) if src == "XPLOR" else parse_cyana(mr_path)

    rows = []
    for r in restraints:
        ta1 = to_traj_atom(r["r1"], r["a1"], res_atoms)
        ta2 = to_traj_atom(r["r2"], r["a2"], res_atoms)
        mr_pair = f"r{r['r1']}:{r['a1']}-r{r['r2']}:{r['a2']}"
        if ta1 is None or ta2 is None:
            rows.append(dict(peptide=pid, format=src,
                             mr_pair=mr_pair,
                             traj_pair=f"[unresolved ta1={ta1} ta2={ta2}]",
                             category=r["category"], intensity=r["intensity"],
                             upper_A="", md_r6_A="", md_mean_A="",
                             md_p5_A="", md_p95_A="", dev_A="",
                             violates="", status="atom_missing",
                             comment=r["comment"]))
            continue

        ds = []
        for fr in frames:
            p1 = fr.get((r["r1"], ta1))
            p2 = fr.get((r["r2"], ta2))
            if p1 is None or p2 is None:
                continue
            ds.append(np.linalg.norm(p1 - p2))

        if not ds:
            rows.append(dict(peptide=pid, format=src,
                             mr_pair=mr_pair,
                             traj_pair=f"r{r['r1']}:{ta1}-r{r['r2']}:{ta2}",
                             category=r["category"], intensity=r["intensity"],
                             upper_A=r["upper"], md_r6_A="", md_mean_A="",
                             md_p5_A="", md_p95_A="", dev_A="",
                             violates="", status="no_frames",
                             comment=r["comment"]))
            continue

        arr = np.array(ds)
        md = r6_avg(arr)
        p5, p95 = np.percentile(arr, [5, 95])
        dev = md - r["upper"]
        rows.append(dict(peptide=pid, format=src,
                         mr_pair=mr_pair,
                         traj_pair=f"r{r['r1']}:{ta1}-r{r['r2']}:{ta2}",
                         category=r["category"], intensity=r["intensity"],
                         upper_A=r["upper"], md_r6_A=md,
                         md_mean_A=arr.mean(), md_p5_A=p5, md_p95_A=p95,
                         dev_A=dev, violates=(md > r["upper"]),
                         status="ok", comment=r["comment"]))
    return rows


def main():
    all_rows = []
    for pid in XPLOR_PEPTIDES + CYANA_PEPTIDES:
        rows = analyze_peptide(pid, RESTRAINT_DIR, TRAJ_DIR)
        ok = sum(1 for r in rows if r["status"] == "ok")
        viol = sum(1 for r in rows if r["status"] == "ok" and r["violates"])
        print(f"{pid}: {len(rows)} restraints  ok={ok}  viol={viol}")
        all_rows.extend(rows)

    cols = ["peptide", "format", "mr_pair", "traj_pair", "category", "intensity",
            "upper_A", "md_r6_A", "md_mean_A", "md_p5_A", "md_p95_A",
            "dev_A", "violates", "status", "comment"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            rec = dict(r)
            for k in ("upper_A", "md_r6_A", "md_mean_A", "md_p5_A", "md_p95_A", "dev_A"):
                if isinstance(rec.get(k), (float, int)) and rec[k] != "":
                    rec[k] = f"{rec[k]:.3f}"
            w.writerow(rec)
    print(f"\nWrote {OUT_CSV} ({len(all_rows)} rows).")


if __name__ == "__main__":
    main()
