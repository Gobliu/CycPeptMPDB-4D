import pandas as pd

# --- Load both CSVs ---
df_all = pd.read_csv('CycPeptMPDB_Peptide_All.csv', low_memory=False)
df_rmsd = pd.read_csv('Peptide_with_pdb_water_with_rmsd.csv')

# --- Check common columns ---
print("Columns in All file:", df_all.columns.tolist())
print("Columns in RMSD file:", df_rmsd.columns.tolist())

# --- Identify merge key ---
# Usually this is "Structurally_Unique_ID" or "CycPeptMPDB_ID"
key = 'CycPeptMPDB_ID'  # change if needed

# --- Select only necessary columns from the All file ---
df_all_sub = df_all[[key, 'Monomer_Length']]

# --- Merge ---
merged = pd.merge(df_rmsd, df_all_sub, on=key, how='left')

# --- Save to new file ---
merged.to_csv('Peptide_with_pdb_water_with_rmsd_with_length.csv', index=False)

print("✅ Merged successfully! Output saved as Peptide_with_pdb_water_with_rmsd_with_length.csv")
