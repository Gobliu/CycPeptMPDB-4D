import os
import re
from pathlib import Path
import numpy as np
import pandas as pd


def load_dat_file(dat_path: str) -> pd.DataFrame:
    """
    Load a .dat file and return a DataFrame with columns:
    [alias, 3D_SASA, 3D_NPSA, 3D_PSA]
    """
    df = pd.read_csv(dat_path, sep=r'\s+', header=None)
    # Strip trailing underscore from ID column
    df[0] = df[0].str.rstrip('_')
    # Keep columns: 0 (ID), -6 (SASA), -5 (NPSA), -4 (PSA)
    df = df.iloc[:, [0, -6, -5, -4]].copy()
    df.columns = ['alias', '3D_SASA', '3D_NPSA', '3D_PSA']
    return df


def process_sa_data(
    csv_path: str,
    hexane_dat: str,
    water_dat: str,
    output_path: str
) -> None:
    """
    Reads surface area features from .dat files and merges them into the CSV
    by matching aliases to CycPeptMPDB_ID.
    """
    csv_p = Path(csv_path)
    all_df = pd.read_csv(csv_p, low_memory=False)

    # 2. Load .dat files (each env can have multiple files, e.g. clean + bad)
    dat_files = {'Hexane': hexane_dat, 'Water': water_dat}

    feature_names = ['3D_SASA', '3D_NPSA', '3D_PSA']

    # Initialize empty columns for each env + feature combination
    for env in dat_files:
        for feat in feature_names:
            col = f'{env}_{feat}'
            if col not in all_df.columns:
                all_df[col] = np.nan

    for env, dat_paths in dat_files.items():
        # Support single path (str) or list of paths
        if isinstance(dat_paths, str):
            dat_paths = [dat_paths]

        parts = []
        for dp in dat_paths:
            print(f"Loading {env} data from {dp}...")
            parts.append(load_dat_file(dp))
        sa_df = pd.concat(parts, ignore_index=True)

        # Add direct mappings for Kelly_XXXX and Naylor_XXXX
        for alias in sa_df['alias']:
            if alias.startswith('Kelly') or alias.startswith('Naylor'):
                cyc_id = int(alias.split('_')[1])            
                match_mask = all_df['CycPeptMPDB_ID'] == cyc_id
                assert match_mask.any(), f"No match found in reference for CycPeptMPDB_ID: {cyc_id}"
            else:
                origin_name = alias[:-1] if alias.endswith('_') else alias
                # print("origin_name:", origin_name)
                match_mask = all_df['Original_Name_in_Source_Literature'] == origin_name
                if not match_mask.any():        # Because some peptides from 2022_Taechalertpaisarn
                    print(f"Warning: No match found in all_df for Original_Name_in_Source_Literature: {origin_name}")
                    continue
            print(f"Processing alias: {alias}, matches found: {match_mask.sum()}")        
            ref_idx = all_df.index[match_mask][0]
            for feat in feature_names:
                col = f'{env}_{feat}'
                val = sa_df.loc[sa_df['alias'] == alias, feat].values[0]
                all_df.at[ref_idx, col] = val
    # Save updated CSV
    out_p = Path(output_path)
    all_df.to_csv(out_p, index=False)
    print(f"Output saved to: {out_p}")


if __name__ == "__main__":
    REPO_ROOT = Path("/home/liuw/GitHub/CycPeptMPDB-4D")

    INPUT_CSV = REPO_ROOT / "csvs" / "CycPeptMPDB-4D_clean.csv"
    OUTPUT_CSV = REPO_ROOT / "csvs" / "CycPeptMPDB-4D_clean.csv"

    HEXANE_DAT = [str(REPO_ROOT / "csvs" / "all_hexane_sa.dat")]
    WATER_DAT = [
        str(REPO_ROOT / "csvs" / "all_water_sa_clean.dat"),
        str(REPO_ROOT / "csvs" / "all_water_sa_bad.dat"),
    ]

    process_sa_data(
        csv_path=str(INPUT_CSV),
        hexane_dat=HEXANE_DAT,
        water_dat=WATER_DAT,
        output_path=str(OUTPUT_CSV)
    )
