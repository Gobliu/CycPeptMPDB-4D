import re
import os
import pandas as pd
from typing import List, Optional, Generator, Set

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
    Tracks and reports missing files specifically (log-only vs traj-only).
    """
    env_dir = os.path.join(base_data_dir, env_name)
    log_dir = os.path.join(env_dir, "Logs")
    traj_dir = os.path.join(env_dir, "Trajectories")
    str_dir = os.path.join(env_dir, "Structures")

    assert os.path.exists(csv_path), f"CSV path missing: {csv_path}"
    assert os.path.exists(log_dir), f"Log directory missing: {log_dir}"
    assert os.path.exists(traj_dir), f"Trajectory directory missing: {traj_dir}"
    
    os.makedirs(str_dir, exist_ok=True)

    df: pd.DataFrame = pd.read_csv(csv_path, low_memory=False)
    
    env_suffix_map = {"Water": "H2O", "Hexane": "Hexane"}
    env_tag = env_suffix_map.get(env_name, env_name)
    
    # Categorization for reporting
    missing_both: List[str] = []
    log_only: List[str] = []
    traj_only: List[str] = []
    error_files: List[str] = []
    processed_count: int = 0

    for _, row in df.iterrows():
        id_tag = f"{row['Source']}_{row['CycPeptMPDB_ID']}_{env_tag}"
        
        log_path = os.path.join(log_dir, f"{id_tag}.log")
        traj_path = os.path.join(traj_dir, f"{id_tag}_Traj.pdb")
        structure_path = os.path.join(str_dir, f"{id_tag}_Str.pdb")

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
            raw_time_ns = get_cluster_middle_time(log_path)
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

def main():
    CSV_PATH = "/home/liuw/GitHub/CycPeptMPDB-4D/csvs/CycPeptMPDB_Peptide_5publications.csv"
    DATA_ROOT = "/home/liuw/GitHub/Data/CycPeptMPDB_4D"
    
    process_structures(CSV_PATH, DATA_ROOT, "Hexane")

if __name__ == '__main__':
    main()