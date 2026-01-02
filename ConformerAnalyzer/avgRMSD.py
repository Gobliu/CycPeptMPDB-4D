import os
import re
import warnings
import numpy as np
import pandas as pd

# --- inputs ---
df = pd.read_csv('../DataProcessor/Peptide_with_pdb_water.csv')
tgt_dir = '/media/liuwei/1T/MD_CycPeptMPDB/Water/avg_pairwise_rmsd'


for fn in os.listdir(tgt_dir):
    if fn.endswith('.xvg'):
        new_name = re.sub(r'(?:-|__).*?(?=_avgRMSD)', '', fn)
        print(fn, new_name)
        if fn != new_name:
            os.rename(os.path.join(tgt_dir, fn), os.path.join(tgt_dir, new_name))
            print(f"Renamed: {fn} → {new_name}")


def read_first_value(path):
    """Read the first numeric value from the first line of a text file.
    Returns np.nan and emits a warning if the file is missing, empty, or invalid.
    """
    try:
        with open(path, 'r') as f:
            line = f.readline().strip()
            if not line:
                warnings.warn(f'Empty first line in file: {path}', RuntimeWarning)
                return np.nan
            try:
                return float(line.split()[0])
            except Exception:
                warnings.warn(f'Cannot parse first value in file: {path}', RuntimeWarning)
                return np.nan
    except FileNotFoundError:
        warnings.warn(f'File not found: {path}', RuntimeWarning)
        return np.nan


# --- create output columns ---
df['avgRMSD_all'] = np.nan
df['avgRMSD_bb'] = np.nan

# --- iterate rows and fill values ---
for idx, row in df.iterrows():
    s1 = str(row['Source']).strip()
    s2 = str(row['CycPeptMPDB_ID']).strip()

    fname_all = f"{s1}_{s2}_avgRMSD_all.xvg"
    fname_bb = f"{s1}_{s2}_avgRMSD_bb.xvg"   # <-- assume backbone file is *_avgRMSD_bb.xvg

    path_all = os.path.join(tgt_dir, fname_all)
    path_bb = os.path.join(tgt_dir, fname_bb)

    # Read and assign values
    df.at[idx, 'avgRMSD_all'] = read_first_value(path_all)
    df.at[idx, 'avgRMSD_bb'] = read_first_value(path_bb)

# (optional) save result
df.to_csv('../DataProcessor/Peptide_with_pdb_water_with_rmsd.csv', index=False)
