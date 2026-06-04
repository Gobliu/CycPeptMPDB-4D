"""
Filter CycPeptMPDB-4D peptides by ideal backbone ring detection.

Reads PDB structures from one environment (default: Hexane), runs
find_backbone_cycle() on each, and writes a CSV containing only the
peptides whose backbone cycle was successfully detected.

Usage:
    python dataprocessor/BackboneFilter.py
    python dataprocessor/BackboneFilter.py --env water
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd

from OmegaComputer import (
    count_extra_backbone_atoms,
    read_pdb,
    infer_bonds,
    build_graph,
    find_backbone_cycle,
)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR_4D = REPO_ROOT.parent / "Data" / "CycPeptMPDB_4D"
CSV_PATH = REPO_ROOT / "csvs" / "CycPeptMPDB-4D.csv"

ENV_SUFFIX_MAP = {"Water": "H2O", "Hexane": "Hexane"}


def filter_ideal_backbone(env="Hexane"):
    """Return a DataFrame of peptides with an ideal backbone ring.

    Ideal means find_backbone_cycle() returns exactly residue_len omega
    dihedral sets (one per residue), i.e. every peptide bond is detected.
    """
    suffix = ENV_SUFFIX_MAP[env]
    pdb_dir = DATA_DIR_4D / env / "Structures"

    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"[{env}] Total peptides: {len(df)}")
    print(f"[{env}] PDB directory:  {pdb_dir}")

    pass_ids = []
    fail_count = 0
    missing_count = 0

    for idx, row in df.iterrows():
        pdb_path = pdb_dir / f"{row.Source}_{row.CycPeptMPDB_ID}_{suffix}_Str.pdb"

        if not pdb_path.exists():
            missing_count += 1
            continue

        t0 = time.time()
        n_extra = count_extra_backbone_atoms(str(pdb_path))

        if n_extra > 0:
            fail_count += 1
            print(f"[{idx}] FAIL  {row.Source}_{row.CycPeptMPDB_ID}  "
                  f"(n_extra={n_extra}, non-standard backbone)")
            sys.stdout.flush()
            continue

        c_list = read_pdb(str(pdb_path))
        if not c_list or not c_list[0]:
            fail_count += 1
            print(f"[{idx}] {row.Source}_{row.CycPeptMPDB_ID}: empty PDB")
            continue

        bonds, atom_types = infer_bonds(c_list[0])
        graph = build_graph(atom_types, bonds)
        residue_len = int(row.Monomer_Length_in_Main_Chain)
        backbone_set = find_backbone_cycle(
            graph, atom_types,
            residue_len=residue_len,
        )

        elapsed = time.time() - t0
        n_omega = len(backbone_set)
        if n_omega == residue_len:
            pass_ids.append(row.CycPeptMPDB_ID)
            print(f"[{idx}] PASS  {row.Source}_{row.CycPeptMPDB_ID}  "
                  f"({n_omega}/{residue_len} omegas, {elapsed:.1f}s)")
        else:
            fail_count += 1
            print(f"[{idx}] FAIL  {row.Source}_{row.CycPeptMPDB_ID}  "
                  f"({n_omega}/{residue_len} omegas, {elapsed:.1f}s)")
        sys.stdout.flush()

    passed_df = df[df["CycPeptMPDB_ID"].isin(pass_ids)].copy()

    print(f"\n{'=' * 50}")
    print(f"Environment:  {env}")
    print(f"Total:        {len(df)}")
    print(f"Missing PDB:  {missing_count}")
    print(f"Pass:         {len(passed_df)}")
    print(f"Fail:         {fail_count}")
    print(f"{'=' * 50}")

    return passed_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter peptides by ideal backbone ring")
    parser.add_argument("--env", default="Hexane", choices=list(ENV_SUFFIX_MAP.keys()),
                        help="Solvent environment (default: Hexane)")
    args = parser.parse_args()

    passed_df = filter_ideal_backbone(env=args.env)

    out_path = REPO_ROOT / "csvs" / f"CycPeptMPDB-4D_ideal_backbone_{args.env.lower()}.csv"
    passed_df.to_csv(out_path, index=False)
    print(f"Saved {len(passed_df)} peptides to {out_path}")
