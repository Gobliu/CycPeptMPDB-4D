from pathlib import Path
import numpy as np
import pandas as pd

from dataprocessor.utils import find_match_in_df


def load_mmpbsa_file(dat_path: str) -> pd.DataFrame:
    """
    Load a mmpbsa .dat file and return a DataFrame with columns:
    [alias, Desolvation_Free_Energy]
    """
    df = pd.read_csv(dat_path, sep=r'\s+', header=None)
    # Strip trailing replica suffix (e.g., _1) only for non-Townsend entries
    mask = ~df[0].str.contains('Townsend', na=False)
    df.loc[mask, 0] = df.loc[mask, 0].str.replace(r'_\d+$', '', regex=True)
    # Keep columns: 0 (ID), 1 (Desolvation_Free_Energy)
    df = df.iloc[:, [0, 1]].copy()
    df.columns = ['alias', 'Desolvation_Free_Energy']
    return df


def process_mmpbsa_data(
    csv_path: str,
    mmpbsa_dat,
    output_path: str
) -> None:
    """
    Reads desolvation free energy from .dat file and merges into the CSV
    by matching aliases to CycPeptMPDB_ID.
    """
    csv_p = Path(csv_path)
    all_df = pd.read_csv(csv_p, low_memory=False)

    col_name = 'Desolvation_Free_Energy'
    if col_name not in all_df.columns:
        all_df[col_name] = np.nan

    if isinstance(mmpbsa_dat, str):
        mmpbsa_dat = [mmpbsa_dat]

    parts = []
    for dp in mmpbsa_dat:
        print(f"Loading MMPBSA data from {dp}...")
        parts.append(load_mmpbsa_file(dp))
    sa_df = pd.concat(parts, ignore_index=True)

    for alias in sa_df['alias']:
        match_mask, skip = find_match_in_df(alias, all_df)
        if skip:
            continue
        print(f"Processing alias: {alias}, matches found: {match_mask.sum()}")
        ref_idx = all_df.index[match_mask][0]
        val = sa_df.loc[sa_df['alias'] == alias, 'Desolvation_Free_Energy'].values[0]
        all_df.at[ref_idx, col_name] = val

    # Save updated CSV
    out_p = Path(output_path)
    all_df.to_csv(out_p, index=False)
    print(f"Output saved to: {out_p}")


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent

    INPUT_CSV = REPO_ROOT / "csvs" / "CycPeptMPDB-4D.csv"
    OUTPUT_CSV = REPO_ROOT / "csvs" / "CycPeptMPDB-4D.csv"

    MMPBSA_DAT = [str(REPO_ROOT / "csvs" / "hexa_fe_100frame")]

    process_mmpbsa_data(
        csv_path=str(INPUT_CSV),
        mmpbsa_dat=MMPBSA_DAT,
        output_path=str(OUTPUT_CSV)
    )
