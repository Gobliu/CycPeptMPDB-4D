import os
import shutil
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser


def pdb_checker(pdb, smiles):
    element_dict_pdb = get_composition_from_pdb(pdb)
    element_dict_pdb = dict(sorted(element_dict_pdb.items()))
    element_dict_smiles = get_composition_from_smiles(smiles)
    element_dict_smiles = dict(sorted(element_dict_smiles.items()))
    match_flag = element_dict_smiles == element_dict_pdb
    if not match_flag:
        print('!!!Not match', 'From pdb', element_dict_pdb, 'From smiles', element_dict_smiles)
    return element_dict_smiles == element_dict_pdb


def get_composition_from_pdb(pdb_path: str):
    comp = defaultdict(int)
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                atom_name = line[12:16].strip()  # Columns 13-16 (0-based index 12-16)
                # Extract element from atom name (first non-digit character)
                element = parse_element_from_atom_name(atom_name)
                comp[element] += 1
    return dict(comp)


def get_composition_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES:", smiles)
    mol_h = Chem.AddHs(mol)  # include implicit hydrogens
    comp = defaultdict(int)
    for atom in mol_h.GetAtoms():
        comp[atom.GetSymbol()] += 1
    return dict(comp)


def parse_element_from_atom_name(atom_name):
    # Remove digits and spaces
    clean_name = ''.join([c for c in atom_name if not c.isdigit()]).strip()

    # Handle special cases (CL, BR, etc.)
    if clean_name.startswith('CL'):
        return 'Cl'
    elif clean_name.startswith('BR'):
        return 'Br'     # not confirmed yet
    elif clean_name.startswith('NA'):
        return 'NA'     # not confirmed yet
    # Add other 2-letter elements as needed (MG, ZN, etc.)

    # Default to first character (C, N, O, etc.)
    return clean_name[0] if clean_name else '?'
