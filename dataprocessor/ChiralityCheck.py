"""Full-coverage D/L chirality audit: every residue of every peptide, both solvents.

See docs/chirality_check.md for the method, the cross-checks that confirm a flag is real,
and the coverage checklist.
"""
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
