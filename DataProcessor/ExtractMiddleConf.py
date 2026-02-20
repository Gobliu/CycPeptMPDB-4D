import re
import os
import pandas as pd
from typing import List, Optional, Generator

ENV_SUFFIX_MAP = {"Water": "H2O", "Hexane": "Hexane"}
RAW_TIME_NS_COLUMN_MAP = {
    "Water": "Water_raw_time_ns",
    "Hexane": "Hexane_raw_time_ns",
}
SUPPORTED_ENVS = tuple(ENV_SUFFIX_MAP.keys())


def get_cluster_middle_time(log_file_path: str) -> Optional[float]:
    """
    Parses the GROMACS clustering log to find the middle structure 't' value
    of the first cluster. Returns time in ns.
    """
    assert os.path.exists(log_file_path), f"Log file missing: {log_file_path}"
    
    with open(log_file_path, 'r') as f:
        in_table = False
        for line in f:
            if "cl. | #st  rmsd | middle rmsd | cluster members" in line:
                in_table = True
                continue
            
            if in_table:
                # Regex looks for cluster 1, then captures the middle time column
                # Pattern: 1 | count rmsd | TIME rmsd | ...
                match = re.search(r"^\s*1\s+\|.*?\|\s+([\d\.]+)\s+[\d\.]+", line)
                if match:
                    return float(match.group(1))
    return None

def pdb_model_generator(file_path: str) -> Generator[List[str], None, None]:
    """
    Yields blocks of lines corresponding to individual MODELS in a multi-model PDB.
    """
    current_model: List[str] = []
    with open(file_path, 'r') as f:
        for line in f:
            current_model.append(line)
            if line.startswith("ENDMDL"):
                yield current_model
                current_model = []

def extract_pdb_by_time(input_pdb: str, output_pdb: str, target_time_ps: float) -> None:
    """
    Extracts a specific MODEL from a multi-model PDB file by matching 
    the timestamp in the TITLE record.
    """
    assert os.path.exists(input_pdb), f"Trajectory file missing: {input_pdb}"
    
    found = False
    for model_lines in pdb_model_generator(input_pdb):
        title_line = next((l for l in model_lines if l.startswith("TITLE")), None)
        assert title_line is not None, f"Model in {input_pdb} lacks a TITLE record"
        
        time_match = re.search(r"t=\s*([\d\.]+)", title_line)
        assert time_match is not None, f"Could not parse time from TITLE: {title_line.strip()}"
        
        current_time = float(time_match.group(1))
        
        if abs(current_time - target_time_ps) < 1e-3:
            with open(output_pdb, 'w') as f_out:
                f_out.writelines(model_lines)
            found = True
            break
            
    assert found, f"Target time {target_time_ps} ps not found in trajectory {input_pdb}"

def process_structures(csv_path: str, base_data_dir: str, env_name: str) -> None:
    """
    Processes logs and trajectories to extract middle-cluster structures.
    Tracks and reports missing files specifically (log-only vs traj-only),
    and writes env-specific raw_time_ns into the input CSV.
    """
    assert env_name in SUPPORTED_ENVS, (
        f"Unsupported environment: {env_name}. "
        f"Supported environments are: {', '.join(SUPPORTED_ENVS)}"
    )

    env_dir = os.path.join(base_data_dir, env_name)
    log_dir = os.path.join(env_dir, "Logs")
    traj_dir = os.path.join(env_dir, "Trajectories")
    str_dir = os.path.join(env_dir, "Structures")

    assert os.path.exists(csv_path), f"CSV path missing: {csv_path}"
    assert os.path.exists(log_dir), f"Log directory missing: {log_dir}"
    assert os.path.exists(traj_dir), f"Trajectory directory missing: {traj_dir}"
    
    os.makedirs(str_dir, exist_ok=True)

    df: pd.DataFrame = pd.read_csv(csv_path, low_memory=False)
    required_cols = {"Source", "CycPeptMPDB_ID"}
    missing_required = required_cols - set(df.columns)
    assert not missing_required, f"CSV missing required columns: {missing_required}"

    for col in ("Water_Structure_ID", "Hexane_Structure_ID", "raw_time_ns"):
        if col in df.columns:
            df = df.drop(columns=[col])

    raw_time_col = RAW_TIME_NS_COLUMN_MAP[env_name]
    if raw_time_col not in df.columns:
        df[raw_time_col] = pd.NA
    
    # Categorization for reporting
    missing_both: List[str] = []
    log_only: List[str] = []
    traj_only: List[str] = []
    error_files: List[str] = []
    processed_count: int = 0

    for idx, row in df.iterrows():
        source = str(row["Source"]).strip()
        cp_id = str(row["CycPeptMPDB_ID"]).strip()
        env_suffix = ENV_SUFFIX_MAP[env_name]
        id_tag = f"{source}_{cp_id}_{env_suffix}"
        
        log_path = os.path.join(log_dir, f"{id_tag}.log")
        traj_path = os.path.join(traj_dir, f"{id_tag}_Traj.pdb")
        structure_path = os.path.join(str_dir, f"{id_tag}_Str.pdb")

        raw_time_ns: Optional[float] = None
        if os.path.exists(log_path):
            try:
                raw_time_ns = get_cluster_middle_time(log_path)
                if raw_time_ns is not None:
                    df.at[idx, raw_time_col] = raw_time_ns
            except (AssertionError, Exception) as e:
                print(f"Error parsing raw_time_ns from {log_path}: {e}")
                error_files.append(id_tag)

        if os.path.exists(structure_path):
            processed_count += 1
            continue

        log_exists = os.path.exists(log_path)
        traj_exists = os.path.exists(traj_path)

        if not log_exists and not traj_exists:
            print(f"[{env_name}] Missing both log and trajectory for ID: {id_tag}, paths: {log_path}, {traj_path}")
            missing_both.append(id_tag)
            continue
        elif log_exists and not traj_exists:
            log_only.append(id_tag)
            continue
        elif not log_exists and traj_exists:
            traj_only.append(id_tag)
            continue

        print(f"[{env_name}] Processing ID: {id_tag}")
        
        try:
            assert raw_time_ns is not None, f"Cluster 1 middle time not found in {log_path}"
            
            target_time_ps = round(raw_time_ns * 1000.0, 4)
            extract_pdb_by_time(traj_path, structure_path, target_time_ps)
            processed_count += 1
            
        except (AssertionError, Exception) as e:
            print(f"Error processing {id_tag}: {e}")
            error_files.append(id_tag)

    # Final Report
    print(f"\n" + "="*50)
    print(f"REPORT FOR ENVIRONMENT: {env_name}")
    print(f"="*50)
    print(f"Successfully processed (inc. existing): {processed_count}")
    print(f"Total entries with missing files: {len(missing_both) + len(log_only) + len(traj_only)}")
    print(f"Processing errors: {len(error_files)}")
    
    if missing_both:
        print(f"\n[!] MISSING BOTH LOG AND TRAJ ({len(missing_both)}):")
        # print("\n".join(missing_both))
        
    if log_only:
        print(f"\n[?] LOG ONLY (TRAJ MISSING) ({len(log_only)}):")
        # print("\n".join(log_only))
        
    if traj_only:
        print(f"\n[?] TRAJ ONLY (LOG MISSING) ({len(traj_only)}):")
        # print("\n".join(traj_only))
        
    if error_files:
        print(f"\n[X] PROCESSING ERRORS ({len(error_files)}):")
        # print("\n".join(error_files))
    print(f"="*50 + "\n")
    df.to_csv(csv_path, index=False)
    print(f"[{env_name}] Updated column in CSV: {raw_time_col}")
    print(f"[{env_name}] CSV path: {csv_path}")

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(SCRIPT_DIR)
    CSV_PATH = os.path.join(REPO_ROOT, "csvs", "CycPeptMPDB-4D_clean.csv")
    DATA_ROOT = os.path.join(os.path.dirname(REPO_ROOT), "Data", "CycPeptMPDB_4D")
    
    process_structures(CSV_PATH, DATA_ROOT, "Water")
    # process_structures(CSV_PATH, DATA_ROOT, "Hexane")

if __name__ == '__main__':
    main()
