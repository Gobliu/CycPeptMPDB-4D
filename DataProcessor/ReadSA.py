from pathlib import Path
import numpy as np
import pandas as pd

from DataProcessor.utils import find_match_in_df


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

    dat_files = {'Hexane': hexane_dat, 'Water': water_dat}
    feature_names = ['3D_SASA', '3D_NPSA', '3D_PSA']

    # Initialize empty columns for each env + feature combination
    for env in dat_files:
        for feat in feature_names:
            col = f'{env}_{feat}'
            if col not in all_df.columns:
                all_df[col] = np.nan

    for env, dat_paths in dat_files.items():
        if isinstance(dat_paths, str):
            dat_paths = [dat_paths]

        parts = []
        for dp in dat_paths:
            print(f"Loading {env} data from {dp}...")
            parts.append(load_dat_file(dp))
        sa_df = pd.concat(parts, ignore_index=True)

        for alias in sa_df['alias']:
            match_mask, skip = find_match_in_df(alias, all_df)
            if skip:
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
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent

    INPUT_CSV = REPO_ROOT / "csvs" / "CycPeptMPDB-4D_clean.csv"
    OUTPUT_CSV = REPO_ROOT / "csvs" / "CycPeptMPDB-4D_clean.csv"

    HEXANE_DAT = [
        str(REPO_ROOT / "csvs" / "hexa-hexane_sa"),
        str(REPO_ROOT / "csvs" / "hepta-hexane_sa"),
    ]
    WATER_DAT = [
        str(REPO_ROOT / "csvs" / "hexa-water_sa"),
        str(REPO_ROOT / "csvs" / "hepta-water_sa"),
    ]

    process_sa_data(
        csv_path=str(INPUT_CSV),
        hexane_dat=HEXANE_DAT,
        water_dat=WATER_DAT,
        output_path=str(OUTPUT_CSV)
    )
