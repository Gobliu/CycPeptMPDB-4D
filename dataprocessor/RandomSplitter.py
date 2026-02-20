import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT.parent / "Data"


def random_splitter(df, repeat_split=10):
    df['PAMPA'] = df['PAMPA'].clip(lower=-8, upper=-4)
    for i in range(repeat_split):
        # Split data into train (80%) and temp (20%)
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=123 * i ** 2)

        # Split temp into validation (50%) and test (50%)
        valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=123 * i ** 2)

        print("Train size:", len(train_df))
        print("Validation size:", len(valid_df))
        print("Test size:", len(test_df))

        train_df.loc[:, f'split{i}'] = 'train'
        valid_df.loc[:, f'split{i}'] = 'valid'
        test_df.loc[:, f'split{i}'] = 'test'

        # Concatenate DataFrames6
        df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    df.to_csv(SCRIPT_DIR / 'Random_Split_With_PDB.csv', index=False)


def df_prune(csv_path, pdb_dir):
    df = pd.read_csv(csv_path)

    def pdb_exists(row):
        pdb_path = os.path.join(pdb_dir, f"{row['Source']}_{row['CycPeptMPDB_ID']}.pdb")
        return os.path.exists(pdb_path)
    print('Before prune, df length', len(df))
    df = df[df.apply(pdb_exists, axis=1)].reset_index(drop=True)
    print('After prune, df length', len(df))
    return df


if __name__ == '__main__':
    df_ = df_prune(csv_path=str(SCRIPT_DIR / 'Peptide_with_pdb.csv'), pdb_dir=str(DATA_DIR / 'Hexene'))
    random_splitter(df_)
