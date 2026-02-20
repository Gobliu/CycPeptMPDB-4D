import logging
import os
import sys
import pickle

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
from CPMP.featurization.data_utils import featurize_mol, get_atom_features


def load_data_from_pdb(dataset_path, pdb_dir, remove_h, add_dummy_node=True, one_hot_formal_charge=False):
    """Load and featurize data stored in a CSV file.

    Args:
        dataset_path (str): A path to the CSV file containing the data. It should have two columns:
                            the first one contains SMILES strings of the compounds,
                            the second one contains labels.
        pdb_path (str): A path to the folder of pdb files.
        remove_h (bool): If True, remove all Hs after reading mol with RDkit.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.
        use_data_saving (bool): If True, saved features will be loaded from the dataset directory; if no feature file
                                is present, the features will be saved after calculations. Defaults to True.

    Returns:
        A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
        and y is a list of the corresponding labels.
    """

    data_df = pd.read_csv(dataset_path)

    data_x = data_df.apply(lambda row: f"{pdb_dir}/{row['source']}_{row['id']}.pdb", axis=1)

    data_y = data_df.iloc[:, 1].values

    if data_y.dtype == np.float64:
        data_y = data_y.astype(np.float32)

    x_all, y_all = load_data_from_pdb_(data_x, data_y, remove_h,
                                       add_dummy_node=add_dummy_node,
                                       one_hot_formal_charge=one_hot_formal_charge)

    return x_all, y_all


import subprocess
import os
import logging
from rdkit import Chem


def load_data_from_pdb_(x_pdbs, labels, remove_h, add_dummy_node=True, one_hot_formal_charge=False):
    """
    Load and featurize data from a list of PDB file paths and corresponding labels.

    Args:
        x_pdbs (list[str]): List of PDB file paths.
        labels (list[float]): List of corresponding labels.
        add_dummy_node (bool): Whether to add a dummy node to the graph. Default is True.
        one_hot_formal_charge (bool): Whether to one-hot encode formal charges. Default is False.

    Returns:
        tuple:
            - X (list): Each element is [atom_features, adjacency_matrix, distance_matrix]
            - y (list): Each element is [label]
    """
    def try_obabel_fallback(pdb_path):
        tmp_mol_path = pdb_path.replace('.pdb', '_tmp_obabel.mol')
        try:
            subprocess.run(
                ["obabel", pdb_path, "-O", tmp_mol_path, "--gen3d"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            mol = Chem.MolFromMolFile(tmp_mol_path, removeHs=remove_h, sanitize=True)
            if mol is None:
                logging.error(f"[RDKit] Failed to parse MOL from Open Babel output: {tmp_mol_path}")
            else:
                logging.info(f"[Open Babel] Successfully recovered molecule from {pdb_path}")
            return mol
        except subprocess.CalledProcessError as e:
            logging.error(f"[Open Babel] Conversion failed for {pdb_path}\n{e.stderr}")
            return None
        finally:
            if os.path.exists(tmp_mol_path):
                os.remove(tmp_mol_path)

    x_all, y_all = [], []

    for pdb, label in zip(x_pdbs, labels):
        mol = Chem.MolFromPDBFile(pdb, removeHs=remove_h, sanitize=True)
        if mol is None:
            logging.warning(f"[RDKit] Failed to parse {pdb}, trying Open Babel fallback...")
            mol = try_obabel_fallback(pdb)
        if mol is None:
            logging.error(f"[ERROR] Skipping {pdb} - cannot parse even after Open Babel fallback.")
            continue

        afm, adj, dist = featurize_mol(mol, add_dummy_node, one_hot_formal_charge)
        x_all.append([afm, adj, dist])
        y_all.append([label])

    return x_all, y_all

