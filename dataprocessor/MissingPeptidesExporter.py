"""
Export the list of in-scope PAMPA peptides that were NOT simulated in the 4D set.

Scope = the 5 source publications (Wang, Kelly, Naylor, Townsend, Furukawa),
i.e. csvs/CycPeptMPDB_Peptide_5publications.csv. A peptide is "missing" when it
has an experimental PAMPA value but its CycPeptMPDB_ID does not appear in the
simulated set (csvs/CycPeptMPDB-4D.csv). Only the identity/experimental columns
are kept, matching the CycPeptMPDB-4D.csv layout.

Regenerate any time with:  python MissingPeptidesExporter.py
"""
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CSV_DIR = REPO_ROOT / "csvs"

# Identity/experimental columns kept in the output (same order as CycPeptMPDB-4D.csv)
META_COLS = [
    "CycPeptMPDB_ID", "Source", "SMILES", "Sequence",
    "Original_Name_in_Source_Literature", "Structurally_Unique_ID",
    "PAMPA", "Monomer_Length", "Monomer_Length_in_Main_Chain", "Molecule_Shape",
]


def export_missing_peptides(scope_csv: Path, simulated_csv: Path, out_csv: Path) -> pd.DataFrame:
    """Write in-scope PAMPA peptides absent from the simulated set; return the rows."""
    assert scope_csv.exists(), f"Scope CSV missing: {scope_csv}"
    assert simulated_csv.exists(), f"Simulated CSV missing: {simulated_csv}"

    scope = pd.read_csv(scope_csv, low_memory=False, encoding="utf-8-sig")
    simulated_ids = set(pd.read_csv(simulated_csv, low_memory=False)["CycPeptMPDB_ID"])

    missing = scope[scope["PAMPA"].notna() & ~scope["CycPeptMPDB_ID"].isin(simulated_ids)]
    missing = missing[META_COLS].sort_values("CycPeptMPDB_ID").reset_index(drop=True)

    missing.to_csv(out_csv, index=False)
    print(f"Wrote {len(missing)} missing peptides -> {out_csv}")
    return missing


if __name__ == "__main__":
    export_missing_peptides(
        scope_csv=CSV_DIR / "CycPeptMPDB_Peptide_5publications.csv",
        simulated_csv=CSV_DIR / "CycPeptMPDB-4D.csv",
        out_csv=CSV_DIR / "missing_peptides.csv",
    )
