import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional

def read_first_value(path: Path) -> float:
    """
    Read the first numeric value from the first line of a text file.
    Raises exceptions for any errors instead of returning np.nan.
    """
    # Let FileNotFoundError propagate naturally
    with path.open('r') as f:
        line = f.readline().strip()
        if not line:
            raise ValueError(f'Empty first line in file: {path}')
        
        try:
            # Extracts the first column from the first line
            return float(line.split()[0])
        except ValueError as e:
            raise ValueError(f'Cannot parse first value in file: {path}. Line content: "{line}"') from e
        except IndexError as e:
            raise IndexError(f'No values found in first line of file: {path}. Line content: "{line}"') from e


def find_file_with_pattern(directory: Path, prefix: str, suffix: str) -> Optional[Path]:
    """
    Defensively searches for a file that starts with prefix and ends with suffix.
    Handles cases like 'Source_ID-Extra_avgRMSD_all.xvg'.
    """
    # Pattern: prefix + anything + suffix
    # We use glob to find matches in the directory
    pattern = f"{prefix}*{suffix}"
    matches = list(directory.glob(pattern))
    
    if not matches:
        return None
    
    # If multiple matches found, we take the first one
    if len(matches) > 1:
         warnings.warn(f"Multiple matches found for {pattern}: {matches}. Using first.", RuntimeWarning)
    
    return matches[0]

def process_rmsd_data(
    csv_path: str, 
    base_data_dir: str, 
    env: str, 
    output_path: str
) -> None:
    """
    Updates RMSD values in the metadata CSV for a specific environment (e.g., Water or Hexane).
    Uses Title Case for column names (e.g., Water_avgRMSD_all).
    Always overwrites with new value; prints [CONFLICT] if existing value differs.
    """
    # 1. Path setup and verification using pathlib
    csv_p = Path(csv_path)
    assert csv_p.exists(), f"Input CSV not found at: {csv_path}"
    
    out_p = Path(output_path)
    
    # Standardize environment string to Title Case for columns
    env_title = env.capitalize()
    
    tgt_dir = Path(base_data_dir) / env_title / 'avg_pairwise_rmsd'
    assert tgt_dir.exists(), f"Target directory not found at: {tgt_dir}"

    # 2. Load DataFrame
    df: pd.DataFrame = pd.read_csv(csv_path, low_memory=False)
    
    # Define column names with Title Case
    col_all = f'{env_title}_avgRMSD_All'
    col_bb = f'{env_title}_avgRMSD_BackBone'
    
    # Initialize columns only if they don't exist
    if col_all not in df.columns:
        df[col_all] = np.nan
    if col_bb not in df.columns:
        df[col_bb] = np.nan

    # Track stats for the final summary
    stats = {
        col_all: {"pre_existing": df[col_all].notna().sum(), "newly_filled": 0},
        col_bb: {"pre_existing": df[col_bb].notna().sum(), "newly_filled": 0}
    }

    # 3. Iterate and update values, reporting conflicts
    for idx, row in df.iterrows():
        source = str(row['Source']).strip()
        cp_id = str(row['CycPeptMPDB_ID']).strip()
        file_prefix = f"{source}_{cp_id}"

        # Update 'all' atoms column
        path_all = find_file_with_pattern(tgt_dir, file_prefix, "_avgRMSD_all.xvg")
        if path_all:
            val = read_first_value(path_all)
            if not pd.isna(val):
                existing = df.at[idx, col_all]
                if pd.notna(existing) and not np.isclose(existing, val):
                    print(
                        f"  [CONFLICT] {col_all} for {file_prefix} "
                        f"(row {idx}): existing={existing}, new={val}"
                    )
                if pd.isna(existing):
                    stats[col_all]["newly_filled"] += 1
                df.at[idx, col_all] = val

        # Update 'backbone' column
        path_bb = find_file_with_pattern(tgt_dir, file_prefix, "_avgRMSD_bb.xvg")
        if path_bb:
            val = read_first_value(path_bb)
            if not pd.isna(val):
                existing = df.at[idx, col_bb]
                if pd.notna(existing) and not np.isclose(existing, val):
                    print(
                        f"  [CONFLICT] {col_bb} for {file_prefix} "
                        f"(row {idx}): existing={existing}, new={val}"
                    )
                if pd.isna(existing):
                    stats[col_bb]["newly_filled"] += 1
                df.at[idx, col_bb] = val

    # 4. Save results
    df.to_csv(out_p, index=False)

    # 5. Final Summary
    total_rows = len(df)
    still_missing_all = df[col_all].isna().sum()
    still_missing_bb = df[col_bb].isna().sum()

    print(f"\nProcessing Complete for environment: {env_title}")
    print(f"Output saved to: {out_p}")
    print("=" * 70)
    print(f"{'Metric':<25} | {col_all:<20} | {col_bb:<20}")
    print("-" * 70)
    print(f"{'Total Rows':<25} | {total_rows:<20} | {total_rows:<20}")
    print(f"{'Pre-existing Values':<25} | {stats[col_all]['pre_existing']:<20} | {stats[col_bb]['pre_existing']:<20}")
    print(f"{'Newly Filled':<25} | {stats[col_all]['newly_filled']:<20} | {stats[col_bb]['newly_filled']:<20}")
    print(f"{'Still Missing':<25} | {still_missing_all:<20} | {still_missing_bb:<20}")
    print("=" * 70)

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent
    BASE_DATA_DIR = REPO_ROOT.parent / "Data" / "CycPeptMPDB_4D"
    
    # Use the existing output as input to allow iterative updates
    INPUT_CSV = REPO_ROOT / "csvs" / "CycPeptMPDB-4D.csv"
    OUTPUT_CSV = REPO_ROOT / "csvs" / "CycPeptMPDB-4D.csv"
    
    # Process Water environment
    process_rmsd_data(
        csv_path=str(INPUT_CSV), 
        base_data_dir=str(BASE_DATA_DIR), 
        env="Water",
        output_path=str(OUTPUT_CSV)
    )