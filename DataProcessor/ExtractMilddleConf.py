import re

from matplotlib import lines

def get_cluster_middle_time(log_file_path):
    """
    Parses the GROMACS clustering log to find the middle structure 't' value
    of the first cluster.
    """
    with open(log_file_path, 'r') as f:
        # Flag to indicate we've reached the table
        in_table = False
        for line in f:
            # Detect the start of the cluster table
            if "cl. | #st  rmsd | middle rmsd | cluster members" in line:
                in_table = True
                continue
            
            if in_table:
                # Look for the line starting with cluster 1
                # Format: 1 |  54  0.095 |   31.5 .077 | ...
                match = re.search(r"^\s*1\s+\|.*?\|\s+([\d\.]+)\s+[\d\.]+", line)
                if match:
                    return float(match.group(1))
    
    return None

def extract_pdb_by_time(input_pdb, output_pdb, target_time):
    """
    Extracts a specific MODEL from a multi-model PDB file by searching
    for the exact timestamp 't= ...' in the TITLE record.
    """
    with open(input_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
        traj = f_in.readlines()
        model_starts = [i for i, line in enumerate(traj) if line.startswith("TITLE")]
        model_ends = [i for i, line in enumerate(traj) if line.startswith("ENDMDL")]
        assert len(model_starts) == len(model_ends), f"Mismatch in MODEL start and end counts, {len(model_starts)} != {len(model_ends)}"
        # print(traj[model_starts[0]].split())
        start_time = float(traj[model_starts[0]].split()[5])
        target_idx = int((target_time - start_time) / 300.0)  # 300 ps intervals starting from start_time
        print(f"Start time: {start_time}, target time: {target_time}, target index: {target_idx}")
        target_model_lines = traj[model_starts[target_idx]:model_ends[target_idx]+1]
        assert target_time == float(target_model_lines[0].split()[5]), f"Target time does not match the model's time, {target_time} != {float(target_model_lines[0].split()[5])}"
        f_out.writelines(target_model_lines)


if __name__ == '__main__':
    import os
    import pandas as pd

    df = pd.read_csv("/home/liuw/GitHub/CycPeptMPDB-4D/csvs/CycPeptMPDB_Peptide_5publications.csv", low_memory=False)
    missing_files = []
    error_files = []
    for idx, row in df.iterrows():
        log_path = f"/home/liuw/GitHub/Data/CycPeptMPDB_4D/Hexane/Logs/{row['Source']}_{row['CycPeptMPDB_ID']}_Hexane.log"
        traj_path = f"/home/liuw/GitHub/Data/CycPeptMPDB_4D/Hexane/Trajectories/{row['Source']}_{row['CycPeptMPDB_ID']}_Hexane_Tarj.pdb"
        structure_path = f"/home/liuw/GitHub/Data/CycPeptMPDB_4D/Hexane/Structures/{row['Source']}_{row['CycPeptMPDB_ID']}_Hexane_Str.pdb"
        if os.path.exists(structure_path):
            continue
        if not os.path.exists(log_path) or not os.path.exists(traj_path):
            missing_files.append(row['CycPeptMPDB_ID'])
            print('Missing file:', log_path, traj_path, "Counter:", len(missing_files))
            continue
        print(f"Processing {log_path} and {traj_path}")
        # 1. Get the middle structure time from the log file
        target_time = round(get_cluster_middle_time(log_path) * 1000.0)  # Convert ns to ps
        if target_time is None:
            error_files.append(row['CycPeptMPDB_ID'])
            print('Error file:', log_path, "Counter:", len(error_files))
            continue
        extract_pdb_by_time(traj_path, structure_path, target_time)