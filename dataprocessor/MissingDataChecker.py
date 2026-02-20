import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
import warnings


def find_file_with_pattern(directory: Path, prefix: str, suffix: str) -> Optional[Path]:
    """
    Searches for a file that starts with prefix and ends with suffix.
    """
    if not directory.exists():
        return None
    pattern = f"{prefix}*{suffix}"
    matches = list(directory.glob(pattern))
    return matches[0] if matches else None


def audit_missing_data(
    csv_path: str,
    base_data_dir: Optional[str] = None,
    output_txt: Optional[str] = None
) -> None:
    """
    Unified audit for all missing data in the CSV:
      - Structure files (2 filesystem checks: Water, Hexane)
      - RMSD (4 columns, with filesystem lookup)
      - SA   (6 columns)
      - MMPBSA (1 column)

    Args:
        csv_path: Path to the CSV file
        base_data_dir: Base data directory for filesystem checks (optional)
        output_txt: Optional path to save the output table as a text file
    """
    csv_p = Path(csv_path)
    assert csv_p.exists(), f"CSV not found: {csv_path}"

    df: pd.DataFrame = pd.read_csv(csv_path, low_memory=False)

    # --- Define all target columns by category ---
    # Order matches the CSV column order: Water first, then Hexane

    # Filesystem-only checks (not CSV columns)
    structure_checks = ["Wat-Str", "Hex-Str"]

    target_cols: Dict[str, List[str]] = {
        "RMSD": [
            "Water_avgRMSD_All", "Water_avgRMSD_BackBone",
            "Hexane_avgRMSD_All", "Hexane_avgRMSD_BackBone",
        ],
        "MMPBSA": [
            "Desolvation_Free_Energy",
        ],
        "SA": [
            "Water_3D_SASA", "Water_3D_NPSA", "Water_3D_PSA",
            "Hexane_3D_SASA", "Hexane_3D_NPSA", "Hexane_3D_PSA",
        ],
    }

    # Short display names for table header
    short_names = {
        "Water_avgRMSD_All": "Wat-RMSD-A",
        "Water_avgRMSD_BackBone":  "Wat-RMSD-B",
        "Hexane_avgRMSD_All": "Hex-RMSD-A",
        "Hexane_avgRMSD_BackBone":  "Hex-RMSD-B",
        "Desolvation_Free_Energy": "Desolv",
        "Water_3D_SASA":  "Wat-SASA",
        "Water_3D_NPSA":  "Wat-NPSA",
        "Water_3D_PSA":   "Wat-PSA",
        "Hexane_3D_SASA": "Hex-SASA",
        "Hexane_3D_NPSA": "Hex-NPSA",
        "Hexane_3D_PSA":  "Hex-PSA",
    }

    # Filesystem lookup configs
    rmsd_fs_config = {
        "Water_avgRMSD_All":  ("Water",  "_avgRMSD_all.xvg"),
        "Water_avgRMSD_BackBone":   ("Water",  "_avgRMSD_bb.xvg"),
        "Hexane_avgRMSD_All": ("Hexane", "_avgRMSD_all.xvg"),
        "Hexane_avgRMSD_BackBone":  ("Hexane", "_avgRMSD_bb.xvg"),
    }

    # Structure file config: {display_name: (env_dir, env_suffix)}
    # Pattern: {Source}_{CycPeptMPDB_ID}_{env_suffix}_Str.pdb
    structure_fs_config = {
        "Wat-Str": ("Water", "H2O"),
        "Hex-Str": ("Hexane", "Hexane"),
    }

    # Flatten all target columns
    all_target_cols = []
    for cols in target_cols.values():
        all_target_cols.extend(cols)

    missing_cols = [c for c in all_target_cols if c not in df.columns]
    if missing_cols:
        warnings.warn(f"Columns not in CSV (skipped): {missing_cols}", RuntimeWarning)

    existing_cols = [c for c in all_target_cols if c in df.columns]

    # Pre-compute structure missing flags if base_data_dir is provided
    base_p = Path(base_data_dir) if base_data_dir else None
    str_missing: Dict[str, List[bool]] = {}
    if base_p:
        for str_name, (env_dir, env_suffix) in structure_fs_config.items():
            str_dir = base_p / env_dir / "Structures"
            flags = []
            for _, row in df.iterrows():
                source = str(row['Source']).strip()
                cp_id = str(row['CycPeptMPDB_ID']).strip()
                pdb_path = str_dir / f"{source}_{cp_id}_{env_suffix}_Str.pdb"
                flags.append(not pdb_path.exists())
            str_missing[str_name] = flags

    # A row is "missing" if any CSV column is NaN OR any structure file is missing
    csv_missing_mask = df[existing_cols].isna().any(axis=1)
    str_missing_mask = pd.Series(False, index=df.index)
    for flags in str_missing.values():
        str_missing_mask = str_missing_mask | pd.Series(flags, index=df.index)
    missing_mask = csv_missing_mask | str_missing_mask
    df_missing = df[missing_mask]

    output_lines: List[str] = []

    def out(line: str) -> None:
        print(line)
        output_lines.append(line)

    if df_missing.empty:
        out(f"\n[PASS] No missing values found in {csv_p.name}.")
        if output_txt:
            _save(output_lines, output_txt)
        return

    # --- Build table ---
    out(f"\n[!] Found {len(df_missing)} peptides with missing data.")

    # Header: Structure checks first, then CSV columns
    col_width = 12
    hdr_parts = [f"{'No.':<5}", f"{'Source':<25}", f"{'ID':<8}"]
    for str_name in structure_checks:
        hdr_parts.append(f"{str_name:<{col_width}}")
    for col in existing_cols:
        hdr_parts.append(f"{short_names.get(col, col):<{col_width}}")
    header = " | ".join(hdr_parts)
    separator = "-" * len(header)

    out(separator)
    out(header)
    out(separator)

    for idx, (df_idx, row) in enumerate(df_missing.iterrows(), start=1):
        source = str(row['Source']).strip()
        cp_id = str(row['CycPeptMPDB_ID']).strip()
        file_prefix = f"{source}_{cp_id}"

        parts = [f"{idx:<5}", f"{source:<25}", f"{cp_id:<8}"]

        # Structure file checks
        for str_name in structure_checks:
            if str_name in str_missing:
                status = "MISSING" if str_missing[str_name][df_idx] else "OK"
            else:
                status = "N/A"
            parts.append(f"{status:<{col_width}}")

        # CSV column checks
        for col in existing_cols:
            if col in df.columns and not pd.isna(row[col]):
                status = "OK"
            elif col in rmsd_fs_config and base_p:
                env_name, suffix = rmsd_fs_config[col]
                tgt_dir = base_p / env_name / 'avg_pairwise_rmsd'
                match = find_file_with_pattern(tgt_dir, file_prefix, suffix)
                status = "EXISTS" if match else "MISSING"
            else:
                status = "MISSING"
            parts.append(f"{status:<{col_width}}")

        out(" | ".join(parts))

    # --- Summary ---
    sep2 = "=" * len(header)
    out(sep2)
    out("Audit Summary")
    out(sep2)
    out(f"Total entries checked:          {len(df)}")
    out(f"Total with any missing value:   {len(df_missing)}")
    out("-" * 50)

    out(f"\n  [Structures]")
    for str_name in structure_checks:
        if str_name in str_missing:
            n_missing = sum(str_missing[str_name])
            out(f"    {str_name:<25} missing: {n_missing}")

    for category, cols in target_cols.items():
        out(f"\n  [{category}]")
        for col in cols:
            if col in df.columns:
                n_missing = df[col].isna().sum()
                out(f"    {short_names.get(col, col):<25} missing: {n_missing}")

    out(sep2)

    if output_txt:
        _save(output_lines, output_txt)


def _save(lines: List[str], output_txt: str) -> None:
    output_path = Path(output_txt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n[INFO] Output saved to: {output_path}")


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent

    FINAL_CSV = REPO_ROOT / "csvs" / "CycPeptMPDB-4D.csv"
    DATA_DIR = REPO_ROOT.parent / "Data" / "CycPeptMPDB_4D"
    OUTPUT_TXT = REPO_ROOT / "csvs" / "missing_data_audit.txt"

    audit_missing_data(
        csv_path=str(FINAL_CSV),
        base_data_dir=str(DATA_DIR),
        output_txt=str(OUTPUT_TXT)
    )
