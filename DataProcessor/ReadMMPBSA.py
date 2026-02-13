import os
import re
from pathlib import Path
import numpy as np
import pandas as pd


def load_mmpbsa_file(dat_path: str) -> pd.DataFrame:
    """
    Load a mmpbsa .dat file and return a DataFrame with columns:
    [alias, Desolvation free energy]
    """
    df = pd.read_csv(dat_path, sep=r'\s+', header=None)
    # Strip trailing _1 (or similar suffix) from ID column
    df[0] = df[0].str.replace(r'_\d+$', '', regex=True)
    # Keep columns: 0 (ID), 1 (Desolvation free energy)
    df = df.iloc[:, [0, 1]].copy()
    df.columns = ['alias', 'Desolvation free energy']
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

    col_name = 'Desolvation free energy'
    if col_name not in all_df.columns:
        all_df[col_name] = np.nan

    # Support single path (str) or list of paths
    if isinstance(mmpbsa_dat, str):
        mmpbsa_dat = [mmpbsa_dat]

    parts = []
    for dp in mmpbsa_dat:
        print(f"Loading MMPBSA data from {dp}...")
        parts.append(load_mmpbsa_file(dp))
    sa_df = pd.concat(parts, ignore_index=True)

    for alias in sa_df['alias']:
        if alias.startswith('Kelly') or alias.startswith('Naylor'):
            cyc_id = int(alias.split('_')[1])
            match_mask = all_df['CycPeptMPDB_ID'] == cyc_id
            assert match_mask.any(), f"No match found in reference for CycPeptMPDB_ID: {cyc_id}"
        else:
            origin_name = alias[:-1] if alias.endswith('_') else alias
            match_mask = all_df['Original_Name_in_Source_Literature'] == origin_name
            if not match_mask.any():
                print(f"Warning: No match found in all_df for Original_Name_in_Source_Literature: {origin_name}")
                continue
        print(f"Processing alias: {alias}, matches found: {match_mask.sum()}")
        ref_idx = all_df.index[match_mask][0]
        val = sa_df.loc[sa_df['alias'] == alias, 'Desolvation free energy'].values[0]
        all_df.at[ref_idx, col_name] = val

    # Save updated CSV
    out_p = Path(output_path)
    all_df.to_csv(out_p, index=False)
    print(f"Output saved to: {out_p}")


if __name__ == "__main__":
    REPO_ROOT = Path("/home/liuw/GitHub/CycPeptMPDB-4D")

    INPUT_CSV = REPO_ROOT / "csvs" / "CycPeptMPDB-4D_clean.csv"
    OUTPUT_CSV = REPO_ROOT / "csvs" / "CycPeptMPDB-4D_clean.csv"

    MMPBSA_DAT = [str(REPO_ROOT / "csvs" / "all_water_mmpbsa.dat")]

    process_mmpbsa_data(
        csv_path=str(INPUT_CSV),
        mmpbsa_dat=MMPBSA_DAT,
        output_path=str(OUTPUT_CSV)
    )
