# Chirality Check — Full Coverage

How to verify the D/L configuration of **every stereocenter in every peptide**, in both
solvents, against the intended sequence. This is a geometry check on the representative
structures; it needs no external chirality reference — the intended D/L comes from the
monomer codes in the `Sequence` column.

## Inputs

| What | Path |
|------|------|
| Structures | `Data/CycPeptMPDB_4D/June2026/{Water,Hexane}/Structures/{Source}_{id}_{H2O\|Hexane}_Str.pdb` |
| Sequence + IDs | `csvs/CycPeptMPDB-4D.csv` (`Sequence`, `Monomer_Length`, `Source`, `CycPeptMPDB_ID`) |
| Stereocenter test | `dataprocessor/DLChecker.py` |

## The stereocenter test

`DLChecker._residue_chirality(atoms, res_num)` classifies one residue's Cα from 3D geometry
as the sign of a scalar triple product of substituent directions:

- returns **+1 = L**, **−1 = D**, **0 = achiral**
- a Cα is a stereocenter only if it carries **exactly one** hydrogen (counted geometrically,
  ≤1.3 Å). Glycine/sarcosine (2 H), N-substituted glycines/peptoids (2 H) and Aib-type
  (0 H) therefore return 0 — correctly, they have no Cα stereocenter.
- handles two backbone topologies automatically (α-amino acid, and β-homo like bHph),
  distinguished by whether Cα is bonded to its own carbonyl carbon.
- **raises `ValueError`** when a residue has no backbone N (an N-terminal acetyl cap or
  other non-amino-acid unit) or is missing a required atom.

## Full-coverage procedure (the parts that are easy to get wrong)

1. **Read every residue `1..Monomer_Length`, not `Monomer_Length_in_Main_Chain`.**
   Structures number residues `1..Monomer_Length`; the main-chain length is smaller for
   **Lariat** peptides, so capping at the main chain silently skips the branch/tail
   monomer(s). Use the full `Sequence` length.
2. **Map residue `i` (1-based) to `Sequence[i-1]` (0-based).** The structure numbers monomers
   in sequence order, so this holds for Circles and Lariats alike (validate: agreement
   should be ~100%).
3. **Intended D/L from the monomer code:** strip N-methyl markers (`Me_`, `me`) — N-methylation
   does not change the Cα configuration — then a leading `d` means D (−1), otherwise L (+1).
4. **Caps:** the first residue of the 2021_Kelly Lariats is an acetyl cap (`ac-`, atoms
   `C, CH3, O, H×3`, no N). `_residue_chirality` raises `ValueError` here — treat as achiral
   and continue to the next residue; **do not let it abort the whole peptide.**
5. **Achiral residues (`flag == 0`) are expected** for Gly, Sar, N-substituted glycines
   (`*_Gly`, and the `Mono##` peptoid codes) and Aib. They have Cα-H count 2 or 0. A residue
   that reads achiral **but has Cα-H = 1** would be a chiral residue masked by a build defect —
   flag that case.
6. **Check both solvents** (`Water` and `Hexane`). Chirality is a covalent property, so a
   correct dataset agrees in both.

## Script

Save as `dataprocessor/ChiralityCheck.py` and run with the project environment.

```python
"""Full-coverage D/L chirality audit: every residue of every peptide, both solvents."""
import sys, ast
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent            # dataprocessor/
REPO_ROOT  = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
import DLChecker

CSV      = REPO_ROOT / "csvs" / "CycPeptMPDB-4D.csv"
DATA_DIR = REPO_ROOT.parent / "Data" / "CycPeptMPDB_4D" / "June2026"
ENVS     = [("Water", "H2O"), ("Hexane", "Hexane")]


def expected_dl(token: str) -> int:
    """Intended Cα configuration from a monomer code: -1 = D, +1 = L."""
    t = token
    for prefix in ("Me_", "me"):            # N-methylation leaves Cα config unchanged
        if t.startswith(prefix):
            t = t[len(prefix):]
    return -1 if t[:1] == "d" else +1        # leading 'd' = D-residue


def audit() -> None:
    df = pd.read_csv(CSV, low_memory=False)
    seen = chiral = achiral = capped = incomplete = 0
    disagreements, broken = [], []

    for _, row in df.iterrows():
        if not str(row.Sequence).startswith("["):
            continue
        seq  = ast.literal_eval(row.Sequence)     # full monomer list
        nres = len(seq)                           # ALL residues, incl. Lariat branch
        src  = str(row.Source).strip()
        cid  = row.CycPeptMPDB_ID

        for env, suffix in ENVS:
            pdb = DATA_DIR / env / "Structures" / f"{src}_{cid}_{suffix}_Str.pdb"
            if not pdb.exists():
                continue
            residues = DLChecker._read_residue_atoms(str(pdb), nres)

            for i in range(nres):                 # residue i+1  <->  seq[i]
                seen += 1
                try:
                    flag = DLChecker._residue_chirality(residues[i], i + 1)
                except ValueError as e:
                    if "missing backbone atom N" in str(e):
                        capped += 1               # acetyl cap / non-amino-acid unit
                    else:
                        incomplete += 1           # missing CA/C/CB -> broken residue
                        broken.append((cid, env, i + 1, seq[i], str(e)))
                    continue

                if flag == 0:
                    achiral += 1                  # Gly/Sar/peptoid/Aib -- no stereocenter
                    continue

                chiral += 1
                if flag != expected_dl(str(seq[i])):
                    disagreements.append((cid, env, i + 1, seq[i], flag))

    print(f"residues examined       : {seen}")
    print(f"  chiral, checked       : {chiral}")
    print(f"  achiral (no center)   : {achiral}")
    print(f"  caps / non-AA units   : {capped}")
    print(f"  incomplete (missing)  : {incomplete}")
    for cid, env, ri, tok, msg in broken:
        print(f"     INCOMPLETE cpid={cid} {env} res#{ri} '{tok}': {msg}")

    print(f"chirality disagreements : {len(disagreements)}")
    for cid, env, ri, tok, flag in disagreements:
        want = "D" if expected_dl(str(tok)) < 0 else "L"
        got  = "D" if flag < 0 else "L"
        print(f"   cpid={cid} {env} res#{ri} '{tok}': sequence={want}, structure={got}")


if __name__ == "__main__":
    audit()
```

A clean dataset prints `incomplete = 0` and `chirality disagreements = 0`. Every listed
disagreement names the peptide, solvent, residue index, monomer, and the direction
(what the sequence intends vs. what the structure has).

## Confirming a flag is real (before fixing anything)

`DLChecker` is geometry; a flag could in principle be a checker edge case. Confirm any hit
with an independent method — especially for prolines (ring, secondary amine):

1. **Independent CIP determinant** — a different formula on the same coordinates. Calibrate
   the sign on a few known-correct residues of the same type, then compare:
   ```python
   import numpy as np
   def cip_sign(a):                      # a = {atom_name: xyz} for one residue
       ca = a["CA"]
       ha = min(((np.linalg.norm(v - ca), v) for k, v in a.items()
                 if k.startswith("H")))[1]          # the single Cα-H
       return int(np.sign(np.linalg.det(np.array([a["N"] - ha, a["C"] - ha, a["CB"] - ha]))))
   ```
   If `cip_sign` and `DLChecker` agree, and both differ from a calibrated correct residue,
   the flag is real.

2. **SMILES CIP (RDKit)** — the deposited `SMILES` is the source of truth for the intended
   chemistry. Assign stereo and read the residue's CIP code: for proline, **L-Pro = S,
   D-Pro = R**. If the SMILES agrees with the sequence label but the structure disagrees,
   the **structure** is the error (not the label).

3. **All trajectory frames** — chirality is covalent, so it is identical in every frame.
   Re-run `cip_sign` on each `MODEL` of the trajectory
   (`.../June2026/{env}/Trajectories/{tag}_Traj.pdb`); a real error shows the wrong sign in
   **all** frames, which also tells you it originates in the source MD, not in structure
   extraction.

## Coverage checklist

The audit is complete only when:

- every residue of every peptide is examined (`residues examined` == `2 × sum(Monomer_Length)`),
- `incomplete == 0` (no missing atoms),
- every achiral-reading residue is a genuine achiral monomer (Cα-H ∈ {0, 2}; none with
  Cα-H = 1),
- both solvents are covered.
