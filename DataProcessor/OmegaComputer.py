import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT.parent / "Data"

# ── Constants ────────────────────────────────────────────────────────────────

COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05, 'Cl': 0.99, 'F': 0.57,
}
BOND_TOLERANCE = 0.4


# ── Parsers ──────────────────────────────────────────────────────────────────

def read_sdf(file_path):
    """Parse multi-conformation SDF/MOL file."""
    conf_list = []
    with open(file_path, 'r') as f:
        for line in f:
            if 'RDKit' in line:
                if len(conf_list) > 2:
                    assert len(conf_list[0]) == len(conf_list[-1])
                conf_list.append([])
            tokens = line.strip().split()
            if len(tokens) == 16:
                conf_list[-1].append(tokens[:4])
    return conf_list


def read_pdb(file_path):
    """Parse single/multi-MODEL PDB using fixed-width columns."""
    conf_list = []
    with open(file_path, 'r') as f:
        for _, line in enumerate(f, 1):
            if line.startswith('MODEL'):
                if conf_list:
                    assert len(conf_list[0]) == len(conf_list[-1]), "Inconsistent atom counts"
                conf_list.append([])
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            if not conf_list:
                conf_list.append([])

            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            element = line[76:78].strip() or line[12:16].strip()[0]
            conf_list[-1].append([x, y, z, element])
    return conf_list


def count_extra_backbone_atoms(file_path):
    """Count extra backbone atoms from non-standard residues in a PDB file.

    - BHF (beta-amino acid): backbone N-CA-CB-C has 4 atoms instead of 3 (+1)
    - TNH (threonine ester): ring closure via CA-CB-OG1 adds 1 extra atom (+1)
    """
    residues = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                res_num = int(line[22:26])
                if res_num not in residues:
                    residues[res_num] = line[17:20].strip()
            if line.startswith('ENDMDL'):
                break
    return sum(1 for name in residues.values() if name in ('BHF', 'TNH'))


# ── Molecular graph ──────────────────────────────────────────────────────────

def infer_bonds(conf):
    num_atoms = len(conf)
    atom_types = []
    bonds = []
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = sum((float(conf[i][k]) - float(conf[j][k])) ** 2 for k in range(3)) ** 0.5
            threshold = COVALENT_RADII[conf[i][-1]] + COVALENT_RADII[conf[j][-1]] + BOND_TOLERANCE
            if dist < threshold:
                bonds.append((i, j))
        atom_types.append(conf[i][-1])
    return bonds, atom_types


def build_graph(atom_types, bonds):
    G = nx.Graph()
    for i, atom in enumerate(atom_types):
        G.add_node(i, element=atom)
    G.add_edges_from(bonds)
    return G


# ── Backbone detection ───────────────────────────────────────────────────────

def _classify_atom(graph, atom_types, idx):
    """Classify backbone atom: N, CA (alpha-carbon), CO (carbonyl), or X."""
    if atom_types[idx] == 'N':
        return 'N'
    neighbor_O_count = sum(1 for n in graph.neighbors(idx) if atom_types[n] == 'O')
    if neighbor_O_count == 0:
        return 'CA'
    elif neighbor_O_count == 1:
        return 'CO'
    return 'X'


def find_backbone_cycle(graph, atom_types, residue_len, n_extra=0, max_cycles=500_000):
    """
    Find backbone omega dihedral atom sets from the molecular graph.
    Iterates cycles lazily with a cap to avoid hanging on complex molecules.

    n_extra: number of extra backbone atoms from non-standard residues
             (e.g. BHF +1 each, TNH +1 each) that increase the cycle length.
    """
    loop_len = residue_len * 3 + n_extra
    tgt_cycles = []
    cycle_count = 0
    for c in nx.simple_cycles(nx.DiGraph(graph)):
        cycle_count += 1
        if len(c) == loop_len:
            tgt_cycles.append(c)
        if cycle_count >= max_cycles:
            print(f"  Warning: hit cycle limit ({max_cycles}), stopping enumeration")
            break

    if not tgt_cycles:
        return []

    omega_atom_set = []
    for c in tgt_cycles:
        for i in range(len(c)):
            names = [_classify_atom(graph, atom_types, c[(i + k) % loop_len]) for k in range(4)]
            if names == ['CA', 'N', 'CO', 'CA']:
                omega_atom_set.append([c[(i + k) % loop_len] for k in range(4)])

    assert len(omega_atom_set) > 0, 'No omega atom set found'
    assert len(omega_atom_set) <= residue_len, 'Incorrect number of omega atom set'
    return omega_atom_set


# ── Torsion angle ────────────────────────────────────────────────────────────

def torsion_angle(conf, atom_sets):
    angles = []
    for a0, a1, a2, a3 in atom_sets:
        p0 = np.asarray(conf[a0][:3], dtype=float)
        p1 = np.asarray(conf[a1][:3], dtype=float)
        p2 = np.asarray(conf[a2][:3], dtype=float)
        p3 = np.asarray(conf[a3][:3], dtype=float)

        b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
        n0 = np.cross(b0, b1); n0 /= np.linalg.norm(n0)
        n1 = np.cross(b1, b2); n1 /= np.linalg.norm(n1)
        b2 /= np.linalg.norm(b2)

        angle = np.degrees(np.arctan2(np.dot(np.cross(n0, b2), n1), np.dot(n0, n1))) % 360
        angles.append(angle)
    return angles


# ── Distribution functions ───────────────────────────────────────────────────

def _init_irregular_log(log_name):
    """Clear the irregular backbone log at the start of each run."""
    open(log_name, 'w').close()
    return set()


def _accumulate_histogram(hist_total, angle_list, bins, range_degrees):
    hist, bin_edges = np.histogram(angle_list, bins=bins, range=range_degrees)
    hist_total += hist
    return bin_edges


def _save_histogram(hist_total, bin_edges, output_pt):
    torch.save({
        'hist_total': torch.tensor(hist_total, dtype=torch.float32),
        'bin_edges': torch.tensor(bin_edges, dtype=torch.float32),
    }, output_pt)
    print(f"Histogram saved to: {output_pt}")


def omega_distribution_cremp(bins=360, range_degrees=(0, 360)):
    hist_total = np.zeros(bins)
    bin_edges = None
    log_name = str(REPO_ROOT / 'logs' / f'Irregular_Backbone_CREMP.txt')
    irregular = _init_irregular_log(log_name)

    sdf_dir = str(DATA_DIR / 'sdf_and_json')
    df = pd.read_csv(DATA_DIR / 'summary_cycpeptmpdb.csv')

    for _, row in df.iterrows():
        sdf_path = f"{sdf_dir}/{row.sequence}.sdf"
        if sdf_path in irregular:
            continue
        c_list = read_sdf(sdf_path)
        bonds, atom_types = infer_bonds(c_list[0])
        graph = build_graph(atom_types, bonds)
        backbone_set = find_backbone_cycle(graph, atom_types, residue_len=row.num_monomers)
        if not backbone_set:
            with open(log_name, 'a') as f:
                f.write(f'{sdf_path}\n')
            continue

        angle_list = []
        for conf in c_list:
            angle_list += torsion_angle(conf, backbone_set)
        bin_edges = _accumulate_histogram(hist_total, angle_list, bins, range_degrees)

    _save_histogram(hist_total, bin_edges, str(REPO_ROOT / 'pts' / 'omega_histogram_cremp.pt'))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return hist_total, bin_centers, bin_edges


def omega_distribution_cycpeptmpdb(bins=360, range_degrees=(0, 360)):
    hist_total = np.zeros(bins)
    bin_edges = None
    log_name = str(REPO_ROOT / 'logs' / f'Irregular_Backbone_CycPeptMPDB.txt')
    irregular = _init_irregular_log(log_name)

    for env, suffix in [('water', '_H2O'), ('vacuum', ''), ('chloroform', '_CHCl3')]:
        mol_dir = str(DATA_DIR / '3d_data_cycpeptmpdb' / 'content' / 'data' / env)
        df = pd.read_csv(SCRIPT_DIR / 'CycPeptMPDB_Peptide_All.csv')
        for _, row in df.iterrows():
            mol_path = f"{mol_dir}/CycPeptMPDB_ID_{row.CycPeptMPDB_ID}{suffix}.mol"
            if mol_path in irregular or not os.path.exists(mol_path):
                continue
            c_list = read_sdf(mol_path)
            bonds, atom_types = infer_bonds(c_list[0])
            graph = build_graph(atom_types, bonds)
            backbone_set = find_backbone_cycle(graph, atom_types, residue_len=row.Monomer_Length)
            if not backbone_set:
                with open(log_name, 'a') as f:
                    f.write(f'{mol_path}\n')
                continue

            angle_list = []
            for conf in c_list:
                angle_list += torsion_angle(conf, backbone_set)
            bin_edges = _accumulate_histogram(hist_total, angle_list, bins, range_degrees)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return hist_total, bin_centers, bin_edges


def omega_distribution_4d(env, env_suffix, pdb_dir, csv_path, bins=360, range_degrees=(0, 360)):
    hist_total = np.zeros(bins)
    bin_edges = None

    pdb_dir = os.path.abspath(pdb_dir)
    csv_path = os.path.abspath(csv_path)
    print(f"[{env}] PDB directory: {pdb_dir}")
    print(f"[{env}] CSV path:      {csv_path}")

    log_name = str(REPO_ROOT / 'logs' / f'Irregular_Backbone_4D_{env}.txt')
    irregular = _init_irregular_log(log_name)

    df = pd.read_csv(csv_path, low_memory=False)
    for idx, row in df.iterrows():
        pdb_path = f"{pdb_dir}/{row.Source}_{row.CycPeptMPDB_ID}_{env_suffix}_Str.pdb"
        if pdb_path in irregular or not os.path.exists(pdb_path):
            continue

        t0 = time.time()
        n_extra = count_extra_backbone_atoms(pdb_path)
        c_list = read_pdb(pdb_path)
        if not c_list or not c_list[0]:
            print(f"[{idx}] {env} {row.Source}_{row.CycPeptMPDB_ID}: empty PDB, skipping")
            quit()
        bonds, atom_types = infer_bonds(c_list[0])
        graph = build_graph(atom_types, bonds)
        backbone_set = find_backbone_cycle(graph, atom_types,
            residue_len=row.Monomer_Length_in_Main_Chain, n_extra=n_extra)
        if not backbone_set:
            with open(log_name, 'a') as f:
                f.write(f'{pdb_path}\n')
            continue

        angle_list = []
        for conf in c_list:
            angle_list += torsion_angle(conf, backbone_set)
        bin_edges = _accumulate_histogram(hist_total, angle_list, bins, range_degrees)
        print(f"[{idx}] {env} {row.Source}_{row.CycPeptMPDB_ID}: {time.time() - t0:.1f}s, omegas={len(angle_list)}")
        sys.stdout.flush()

    _save_histogram(hist_total, bin_edges, str(REPO_ROOT / 'pts' / f'omega_histogram_4d_{env.lower()}.pt'))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return hist_total, bin_centers, bin_edges


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_DIR_4D = DATA_DIR / "CycPeptMPDB_4D"
    CSV_PATH = str(REPO_ROOT / "csvs" / "CycPeptMPDB-4D_clean.csv")
    ENV_SUFFIX_MAP = {"Water": "H2O", "Hexane": "Hexane"}

    # hist_total, bin_centers, bin_edges = omega_distribution_cremp()
    # hist_total, bin_centers, bin_edges = omega_distribution_cycpeptmpdb()

    for env, suffix in ENV_SUFFIX_MAP.items():
        hist_total, bin_centers, bin_edges = omega_distribution_4d(
            env, suffix,
            str(DATA_DIR_4D / env / "Structures"),
            CSV_PATH,
        )

    # plt.figure(figsize=(8, 5))
    # plt.bar(bin_centers, hist_total, width=(bin_edges[1] - bin_edges[0]), align='center', edgecolor='k')
    # plt.xlabel("$\\omega$ Torsion Angle (degrees)")
    # plt.ylabel("Frequency")
    # plt.title(f"Distribution of $\\omega$ Angles — {env}")
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # plt.show()