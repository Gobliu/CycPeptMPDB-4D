import os
import pandas as pd
import shutil


def check_exist_rename_move(source_dir, target_dir):
    df = pd.read_csv("/home/liuw/GitHub/CycPeptMPDB-4D/csvs/CycPeptMPDB_Peptide_5publications.csv", low_memory=False)

    missing_files = []
    for idx, row in df.iterrows():

        # ===== For Hexane Trajectories =====
        # if row['Source'] == '2015_Wang':
        #     pdb_path = os.path.join(source_dir, "2015_Wang",f"{row['Original_Name_in_Source_Literature']}_100frames.pdb")
        # elif row['Source'] == '2016_Furukawa':
        #     pdb_path = os.path.join(source_dir, "2016_Furukawa",f"{row['Original_Name_in_Source_Literature']}_100frames_princ.pdb")
        # elif row['Source'] == '2018_Naylor':
        #     pdb_path = os.path.join(source_dir, "2018_Naylor",f"Naylor_{row['CycPeptMPDB_ID']}_100frames.pdb")
        # elif row['Source'] == '2020_Townsend':
        #     pdb_path = os.path.join(source_dir, "2020_Townsend",f"2020_Townsend_{row['CycPeptMPDB_ID']}-{row['Original_Name_in_Source_Literature']}_100frames.pdb")
        # elif row['Source'] == '2021_Kelly':
        #     pdb_path = os.path.join(source_dir, "2021_Kelly",f"2021_Kelly_{row['CycPeptMPDB_ID']}_100frames.pdb")
        # else:
        #     continue

        # target_path = os.path.join(target_dir, f"{row['Source']}_{row['CycPeptMPDB_ID']}.pdb")

        # ===== For Hexane Logs =====
        if row['Source'] == '2015_Wang':
            pdb_path = os.path.join(source_dir, "2015_Wang",f"{row['Original_Name_in_Source_Literature']}.log")
        elif row['Source'] == '2016_Furukawa':
            pdb_path = os.path.join(source_dir, "2016_Furukawa",f"{row['Original_Name_in_Source_Literature']}.log")
        elif row['Source'] == '2018_Naylor':
            pdb_path = os.path.join(source_dir, "2018_Naylor",f"Naylor_{row['CycPeptMPDB_ID']}.log")
        elif row['Source'] == '2020_Townsend':
            pdb_path = os.path.join(source_dir, "2020_Townsend",f"{row['Original_Name_in_Source_Literature']}.log")
        elif row['Source'] == '2021_Kelly':
            pdb_path = os.path.join(source_dir, "2021_Kelly",f"Kelly_{row['CycPeptMPDB_ID']}.log")
        else:
            continue

        target_path = os.path.join(target_dir, f"{row['Source']}_{row['CycPeptMPDB_ID']}_Hexane.log")

        if os.path.exists(target_path):
            continue

        if not os.path.exists(pdb_path):
            missing_files.append(row['CycPeptMPDB_ID'])
            print('Missing file:', pdb_path, "Counter:", len(missing_files))
            continue

        print(f'Moving {pdb_path} -> {target_path}')
        shutil.move(pdb_path, target_path)
    print(f"Total missing files: {len(missing_files)}")

    #     pdb_path = f"{row['log_path'][:-4]}_1st_cluster.pdb"
    #     if not os.path.exists(pdb_path):
    #         pdb_path = f"{row['log_path'][:-4]}_1st_frame.pdb"
    #     if not os.path.exists(pdb_path):
    #         # print('File does not exist', row['log_path'])
    #         missing_files.append(row['CycPeptMPDB_ID'])
    #         continue

    #     match_flag = pdb_checker(pdb_path, row['SMILES'])

    #     if match_flag:
    #         source = row['Source'].strip()
    #         filename = f"{source}_{row['CycPeptMPDB_ID']}.pdb"
    #         dest_path = os.path.join(match_pdb_dir.strip(), filename)
    #         print(f'Moving {pdb_path} -> {dest_path}')
    #         shutil.move(pdb_path, dest_path)
    #     else:
    #         print(f"!!! Mismatch {row['log_path']}")
    #         mismatches.append((idx, row['SMILES']))
    #         dest_path = os.path.join(mismatch_pdb_dir.strip(), pdb_path.split('/')[-1])
    #         shutil.copy2(pdb_path, dest_path)
    #         gen_path = dest_path[:-16] + '_rdkit.pdb'
    #         smiles2pdb(row['SMILES'], gen_path)

    # print(f"Total mismatches: {len(mismatches)}")

def rename_files_in_folder(folder_path, extension, suffix):
    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file has a .pdb extension
        if file_name.endswith(extension):
            # Construct the full path of the original file
            old_file_path = os.path.join(folder_path, file_name)
            
            # Add "Hexane_Tarj" before the .pdb extension
            new_file_name = file_name.replace(extension, f"{suffix}{extension}")
            new_file_path = os.path.join(folder_path, new_file_name)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {old_file_path} -> {new_file_path}")


if __name__ == '__main__':
    # df = pd.read_csv("/home/liuw/GitHub/CycPeptMPDB-4D/csvs/CycPeptMPDB_Peptide_All.csv", low_memory=False)
    # # Filter the DataFrame to keep only rows where "Source" is in the specified list
    # valid_sources = ["2015_Wang", "2016_Furukawa", "2018_Naylor", "2020_Townsend", "2021_Kelly"]
    # df = df[df["Source"].isin(valid_sources)]
    # df.to_csv("/home/liuw/GitHub/CycPeptMPDB-4D/csvs/CycPeptMPDB_Peptide_5publications.csv", index=False)

    check_exist_rename_move(
        source_dir='/home/liuw/GitHub/Data/CycPeptMPDB_4D/Hexane/Logs',
        target_dir='/home/liuw/GitHub/Data/CycPeptMPDB_4D/Hexane/Logs'
    )

    # rename_files_in_folder(
    #     folder_path='/home/liuw/GitHub/Data/CycPeptMPDB_4D/Hexane/Trajectories',
    #     extension=".pdb",
    #     suffix="_Hexane_Tarj"
    # )