import os
import shutil
import pandas as pd
from typing import List, Optional, Set

def get_source_specific_filename(
    row: pd.Series, 
    file_type: str
) -> str:
    """
    Generates the source filename based on publication-specific naming conventions.
    """
    source: str = row['Source']
    cp_id: str = str(row['CycPeptMPDB_ID'])
    orig_name: str = str(row.get('Original_Name_in_Source_Literature', ''))

    if file_type == "log":
        if source == '2015_Wang':
            return f"2015_Wang_{cp_id}-{orig_name}_1st_frame.log"
        elif source == '2016_Furukawa':
            return f"2016_Furukawa_{cp_id}__{orig_name}_1st_frame.log"
        elif source == '2018_Naylor':
            return f"2018_Naylor_{cp_id}_1st_frame.log"
        elif source == '2020_Townsend':
            return f"2020_Townsend_{cp_id}-{orig_name}_1st_frame.log"
        elif source == '2021_Kelly':
            return f"2021_Kelly_{cp_id}_1st_frame.log"
    
    elif file_type == "pdb":
        if source == '2015_Wang':
            return f"2015_Wang_{cp_id}-{orig_name}_100frames_20.3-50ns.pdb"
        elif source == '2016_Furukawa':
            return f"2016_Furukawa_{cp_id}__{orig_name}_100frames_20.3-50ns.pdb"
        elif source == '2018_Naylor':
            return f"2018_Naylor_{cp_id}_100frames_20.3-50ns.pdb"
        elif source == '2020_Townsend':
            return f"2020_Townsend_{cp_id}-{orig_name}_100frames_20.3-50ns.pdb"
        elif source == '2021_Kelly':
            return f"2021_Kelly_{cp_id}_100frames_20.3-50ns.pdb"
            
    raise ValueError(f"Unsupported source/type combination: {source} / {file_type}")

def check_exist_rename_move(
    csv_path: str, 
    source_dir: str, 
    target_dir: str, 
    env: str,
    mode: str = "log"
) -> None:
    """
    Validates, renames, and moves files from a raw source directory to a standardized target directory.
    
    Args:
        csv_path: Path to the metadata CSV.
        source_dir: Directory containing raw files.
        target_dir: Directory to move standardized files to.
        env: The environment identifier (e.g., 'Hexane' or 'Water').
        mode: 'log' for log files or 'pdb' for trajectory files.
    """
    assert os.path.exists(csv_path), f"CSV path does not exist: {csv_path}"
    assert os.path.exists(source_dir), f"Source directory does not exist: {source_dir}"
    os.makedirs(target_dir, exist_ok=True)

    df: pd.DataFrame = pd.read_csv(csv_path, low_memory=False)
    valid_sources: Set[str] = {"2015_Wang", "2016_Furukawa", "2018_Naylor", "2020_Townsend", "2021_Kelly"}
    
    missing_files: List[str] = []
    processed_count: int = 0

    # Map environment name to standardized suffix shorthand
    env_suffix_map = {"Water": "H2O", "Hexane": "Hexane"}
    env_label = env_suffix_map.get(env, env)

    for _, row in df.iterrows():
        if row['Source'] not in valid_sources:
            continue

        # Standardized target name based on environment argument
        suffix = f"{env_label}.log" if mode == "log" else f"{env_label}_Traj.pdb"
        target_filename = f"{row['Source']}_{row['CycPeptMPDB_ID']}_{suffix}"
        target_path = os.path.join(target_dir, target_filename)

        # Calculate source path
        try:
            src_filename = get_source_specific_filename(row, file_type=mode)
            src_full_path = os.path.join(source_dir, row['Source'], src_filename)
        except ValueError as e:
            print(f"Skipping row due to metadata error: {e}")
            continue

        # Logic Fix: Even if target exists, we check the source to ensure integrity
        source_exists = os.path.exists(src_full_path)
        target_exists = os.path.exists(target_path)

        if target_exists:
            # File already processed
            # print(f"Target already exists, skipping: {target_path}")
            continue

        if source_exists:
            print(f"Moving: {src_full_path} -> {target_path}")
            shutil.move(src_full_path, target_path)
            processed_count += 1
        else:
            # print(f"Missing source file: {src_full_path}")
            missing_files.append(src_full_path)
            continue

    print(f"\nProcessing Complete for mode: {mode} in environment: {env}")
    print(f"Moved: {processed_count}")
    print(f"Total missing from source: {len(missing_files)}")

if __name__ == '__main__':
    # Configuration
    BASE_PATH = "/home/liuw/GitHub"
    REPO_PATH = os.path.join(BASE_PATH, "CycPeptMPDB-4D")
    DATA_PATH = os.path.join(BASE_PATH, "Data", "CycPeptMPDB_4D")
    
    CSV_PATH = os.path.join(REPO_PATH, "csvs", "CycPeptMPDB_Peptide_5publications.csv")
    
    # 1. Process Water Logs
    # check_exist_rename_move(
    #     csv_path=CSV_PATH,
    #     source_dir=os.path.join(DATA_PATH, "Water", "pdb_log_20.3-50ns"),
    #     target_dir=os.path.join(DATA_PATH, "Water", "Logs"),
    #     env="Water",
    #     mode="log"
    # )

    # 2. Process Hexane Trajectories
    check_exist_rename_move(
        csv_path=CSV_PATH,
        source_dir=os.path.join(DATA_PATH, "Hexane", "Logs"),
        target_dir=os.path.join(DATA_PATH, "Hexane", "Logs"),
        env="Hexane",
        mode="log"
    )