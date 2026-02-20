import os
import pandas as pd
import re
from pathlib import Path

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT.parent / "Data"


def normalize_name(original_name, source, verbose=False):
    """
    Normalize only if source == '2022_Taechalertpaisarn'.
    E.g.,
      'ID.3-Pye3(Ala2Abu5)' -> 'Pye3_Ala2Abu5'
      'ID.1-Pye4'           -> '1Pye4'
      'ID.1'                -> '1Pye1'
    """
    if source != '2022_Taechalertpaisarn':
        return original_name

    match = re.match(r'ID\.(\d+)-([A-Za-z0-9]+)(?:\(([^)]+)\))?', original_name)
    if match:
        id_num = match.group(1)
        name = match.group(2)
        extra = match.group(3)
        if extra:
            new_name = f"{name}_{extra}"
        else:
            new_name = f"{id_num}{name}"
    else:
        match = re.match(r'ID\.(\d+)$', original_name)
        if match:
            id_num = match.group(1)
            new_name = f"{id_num}Pye1"
        else:
            new_name = original_name

    if verbose:
        print(original_name, "->", new_name)
    return new_name


def get_log_path(row):
    key1 = (row['Normalized_Name'], row['Source'])
    key2 = (str(row['CycPeptMPDB_ID']), row['Source'])
    return log_dict.get(key1) or log_dict.get(key2)


# Step 1: Traverse subfolders and collect mapping
log_dict = dict()  # (name, source) → full .log path
# root_dir = str(DATA_DIR / 'pdb_log')
root_dir = str(DATA_DIR / 'water' / 'pdb_log')

for dirpath, _, filenames in os.walk(root_dir):
    for fname in filenames:
        if fname.endswith('.log'):
            prefix = fname[:-4]  # Remove ".log"
            if prefix.startswith('Kelly') or prefix.startswith('Naylor'):
                prefix = prefix[-4:]
            folder_name = os.path.basename(dirpath)
            key = (prefix, folder_name)
            full_path = os.path.join(dirpath, fname)
            log_dict[key] = full_path

# Also create a set of log entry keys for comparison later
log_entries = set(log_dict.keys())

# Step 2: Load CSV and normalize
csv_path = str(SCRIPT_DIR / 'CycPeptMPDB_Peptide_All.csv')
col_list = ['CycPeptMPDB_ID', 'Original_Name_in_Source_Literature', 'Source',
            'Structurally_Unique_ID', 'SMILES', 'PAMPA']
df = pd.read_csv(csv_path, low_memory=False)[col_list]

df['Normalized_Name'] = df.apply(
    lambda row: normalize_name(row['Original_Name_in_Source_Literature'], row['Source']),
    axis=1
)

# Step 3: Match to log_dict and extract path
df['log_path'] = df.apply(get_log_path, axis=1)
df_filtered = df[df['log_path'].notnull()]

# Step 4: Save
df_filtered.to_csv(SCRIPT_DIR / 'Peptide_with_pdb_water.csv', index=False)
print(f"Saved {len(df_filtered)} matched rows to 'Peptide_with_pdb_water.csv'")

# Step 5: Build the set of matched (name, source) keys
filtered_keys = set(zip(df_filtered['Normalized_Name'], df_filtered['Source'])) | \
                set(zip(df_filtered['CycPeptMPDB_ID'].astype(str), df_filtered['Source']))

# Step 6: Find unmatched .log entries
unmatched_entries = log_entries - filtered_keys

# Step 7: Report
for entry in sorted(unmatched_entries):
    print(entry)
print(f"\n{len(unmatched_entries)} log entries not found in the CSV:")
