"""
D/L chirality checker for cyclic-peptide residues (geometry only).

For each main-chain residue, classify its stereocenter as L (+1), D (-1), or
achiral (0, e.g. glycine) directly from the 3D structure, as the sign of a
scalar triple product of substituent directions at the stereocenter.

Two residue topologies are handled, distinguished automatically by whether the
alpha carbon is bonded to its own carbonyl carbon:

  - alpha-amino acid: CA is bonded to C(=O). Stereocenter = CA with substituents
    N, C(=O), CB (first sidechain atom), HA.
        sign of (N - CB) x (HA - CB) . (C - CA)

  - beta-homo-amino acid (e.g. bHph): CA is NOT bonded to C(=O) -- an extra CH2
    (named CB) sits between CA and the carbonyl -- so CB is BACKBONE, and the
    sidechain is CA's other carbon neighbour (named CD), not CB. The same triple
    product is used with the atoms remapped by role: sidechain CB -> CD,
    carbonyl-direction C -> CB.
        sign of (N - CD) x (HA - CD) . (CB - CA)

The two formulas share one sign convention (L = +1), verified to agree with the
RDKit CIP assignment (S = L) on all 955 bHph stereocenters in the dataset.

Run standalone per peptide.
"""
import numpy as np

BOND_MAX = 1.8   # max heavy-atom covalent bond length (Angstrom)
H_BOND_MAX = 1.3  # max C-H bond length (Angstrom); heavy neighbours sit >=1.4 A away


def dl_checker(pdb_path, res_length):
    """Return an int array of length res_length: +1 L, -1 D, 0 achiral.

    Residues are read by PDB sequence number and must be numbered
    1..res_length (main-chain order). A residue whose Cα does not carry exactly
    one hydrogen (glycine/sarcosine: 2 H; Aib-like: 0 H) reports 0; any other
    residue missing a required backbone/sidechain atom raises (fast-fail).
    """
    residues = _read_residue_atoms(pdb_path, res_length)

    dl_flag = np.zeros(res_length, dtype=int)
    for i, atoms in enumerate(residues):
        dl_flag[i] = _residue_chirality(atoms, i + 1)

    print("dl_flag (1:L -1:D 0:achiral)", dl_flag)
    return dl_flag


def _residue_chirality(atoms, res_num):
    """L (+1) / D (-1) / achiral (0) for one residue's stereocenter."""
    for required in ("N", "CA", "C"):
        if required not in atoms:
            raise ValueError(f"residue {res_num}: missing backbone atom {required}")

    ca, n, c = atoms["CA"], atoms["N"], atoms["C"]

    # A Cα is a stereocenter only if it carries exactly one hydrogen. Count the
    # hydrogens bonded to CA geometrically (naming-independent), so glycine and
    # sarcosine (2 H on CA) and Aib-like residues (0 H, two methyls) are all
    # correctly reported achiral regardless of how the H atoms are named.
    h_on_ca = [xyz for name, xyz in atoms.items()
               if name.startswith("H") and _dist(ca, xyz) <= H_BOND_MAX]
    if len(h_on_ca) != 1:
        return 0  # 0 H (Aib) or >=2 H (Gly/Sar) -> no Cα-H stereocenter
    ha = h_on_ca[0]

    if "CB" not in atoms:
        raise ValueError(f"residue {res_num}: chiral Cα (1 H) but no CB sidechain atom")
    cb = atoms["CB"]

    if _dist(ca, c) <= BOND_MAX:
        # alpha-amino acid: CA bonded to its carbonyl. sidechain = CB, apex = C
        pivot, apex = cb, c
    else:
        # beta-homo-amino acid (e.g. bHph): CA not bonded to C. CB is the backbone
        # CH2 toward the carbonyl; the sidechain is CA's other carbon (CD).
        cd = _sidechain_carbon(atoms, ca, cb)
        if cd is None:
            raise ValueError(f"residue {res_num}: beta-residue sidechain carbon not found")
        pivot, apex = cd, cb  # sidechain = CD, carbonyl-direction = CB

    triple = np.dot(np.cross(n - pivot, ha - pivot), apex - ca)
    return int(np.sign(triple))


def _sidechain_carbon(atoms, ca, cb):
    """CA's bonded carbon that is NOT the backbone CH2 (CB): the sidechain carbon."""
    for name, xyz in atoms.items():
        if not name.startswith("C") or name in ("CA", "C"):
            continue
        if np.allclose(xyz, cb):
            continue
        if _dist(ca, xyz) <= BOND_MAX:
            return xyz
    return None


def _read_residue_atoms(pdb_path, res_length):
    """List (length res_length) of {atom_name: xyz} for residues numbered 1..res_length."""
    residues = [dict() for _ in range(res_length)]
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            res_idx = int(line[22:26]) - 1
            if not 0 <= res_idx < res_length:
                continue
            name = line[12:16].strip()
            residues[res_idx][name] = np.array(
                (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            )
    return residues


def _dist(p, q):
    return float(np.linalg.norm(p - q))


if __name__ == "__main__":
    from pathlib import Path

    PDB_PATH = Path("pep_md3.pdb")  # structure to check -- edit and rerun
    RES_LENGTH = 6                  # number of main-chain residues -- edit and rerun
    dl_checker(str(PDB_PATH), RES_LENGTH)
