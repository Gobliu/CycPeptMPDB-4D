import os

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import torch


covalent_radii = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05, 'Cl': 0.99, 'F': 0.57
    # Add more elements if needed
}
# CA-N 1.492; C-H 1.22; N-CO 1.28; CO-O 1.22; CA-CB 1.55


def bond_threshold(a1, a2, tolerance=0.4):
    return covalent_radii[a1] + covalent_radii[a2] + tolerance


def read_sdf(file_path):
    """can cope with multi-conf"""
    conf_list = []
    with open(file_path, 'r') as sdf:
        for line in sdf:
            if 'RDKit' in line:
                if len(conf_list) > 2:
                    assert len(conf_list[0]) == len(conf_list[-1]), f'model {len(conf_list)} has diff number of atoms'
                conf_list.append([])
            line = line.strip().split()
            if len(line) == 16:
                conf_list[-1].append(line[:4])
    # print(len(conf_list), len(conf_list[0]))
    # print(conf_list)
    return conf_list


def read_pdb(file_path):
    """single conf only"""
    conf_list = []
    with open(file_path) as f:
        for line in f:
            if 'MODEL' in line:
                if len(conf_list) > 2:
                    assert len(conf_list[0]) == len(conf_list[-1]), f'model {len(conf_list)} has diff number of atoms'
                conf_list.append([])
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            line = line.strip().split()
            # conf_list.append(line[5:8])
            conf_list[-1].append(line[5:8] + [line[2][0]])
    # print(len(conf_list), conf_list)
    return conf_list


def infer_bonds(conf):
    num_atoms = len(conf)
    atom_types = []
    bonds = []
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            # print(conf[i], conf[j])
            dist = ((float(conf[i][0]) - float(conf[j][0]))**2 +
                    (float(conf[i][1]) - float(conf[j][1]))**2 +
                    (float(conf[i][2]) - float(conf[j][2]))**2)**0.5
            # print(dist)
            if dist < bond_threshold(conf[i][-1], conf[j][-1]):
                bonds.append((i, j))
                # print(conf[i][-1], conf[j][-1], dist)
        atom_types.append(conf[i][-1])
    # print(len(bonds), len(atom_types))
    return bonds, atom_types


def build_graph(atom_types, bonds):
    G = nx.Graph()
    for i, atom in enumerate(atom_types):
        G.add_node(i, element=atom)
    for i, j in bonds:
        G.add_edge(i, j)
    return G


def get_atom_name(graph, atom_types, idx):
    if atom_types[idx] == 'N':
        return 'N'
    neighbors = list(graph.neighbors(idx))
    neighbor_elements = [atom_types[n] for n in neighbors]
    if neighbor_elements.count('O') == 0:
        return 'CA'
    elif neighbor_elements.count('O') == 1:
        return 'CO'
    else:
        return 'X'


# Only for Circle shape, Lariat shape will be treated as irregular-backbone
def find_backbone_cycle(graph, atom_types, residue_len):
    cycles = list(nx.simple_cycles(nx.DiGraph(graph)))
    loop_len = residue_len * 3
    tgt_cycles = [c for c in cycles if len(c) == loop_len]
    if len(tgt_cycles) == 0:
        print(sorted([len(c) for c in cycles], reverse=True))
        return []
    omega_atom_set = []
    for c in tgt_cycles:
        for i in range(len(c)):
            if get_atom_name(graph, atom_types, c[i]) != 'CA':
                continue
            if get_atom_name(graph, atom_types, c[(i+1) % loop_len]) != 'N':
                continue
            if get_atom_name(graph, atom_types, c[(i+2) % loop_len]) != 'CO':
                continue
            if get_atom_name(graph, atom_types, c[(i+3) % loop_len]) != 'CA':
                continue
            omega_atom_set.append([c[i], c[(i+1) % loop_len], c[(i+2) % loop_len], c[(i+3) % loop_len]])
    # print(omega_atom_set)
    assert len(omega_atom_set) <= residue_len, 'Incorrect number of omega atom set'     # can be less
    return omega_atom_set


def torsion_angle(conf, atom_sets):
    angle_list = []
    for one_set in atom_sets:
        assert len(one_set) == 4, f'One atom set for torsion angle should have 4 atoms'
        p0 = np.asarray([float(i) for i in conf[one_set[0]][:3]])
        p1 = np.asarray([float(i) for i in conf[one_set[1]][:3]])
        p2 = np.asarray([float(i) for i in conf[one_set[2]][:3]])
        p3 = np.asarray([float(i) for i in conf[one_set[3]][:3]])

        b0 = p1 - p0
        b1 = p2 - p1
        b2 = p3 - p2

        n0 = np.cross(b0, b1)
        n1 = np.cross(b1, b2)

        n0 /= np.linalg.norm(n0)
        n1 /= np.linalg.norm(n1)
        b2 /= np.linalg.norm(b2)

        m1 = np.cross(n0, b2)

        x = np.dot(n0, n1)
        y = np.dot(m1, n1)

        angle = np.degrees(np.arctan2(y, x)) % 360
        # print(angle)
        angle_list.append(angle)
    return angle_list


def omega_distribution_cremp(bins=360, range_degrees=(0, 360)):
    hist_total = np.zeros(bins)
    bin_edges = None

    try:
        with open('Irregular_Backbone_CREMP.txt', 'r') as log:
            irregular_backbone_sdf = [line.strip() for line in log]
    except FileNotFoundError:
        irregular_backbone_sdf = []

    # Go through all the files ending with .sdf under ip_dir
    sdf_dir = '/media/liuwei/1T/sdf_and_json'
    df = pd.read_csv('/media/liuwei/1T/summary_cycpeptmpdb.csv')
    for idx, row in df.iterrows():
        sdf_path = f"{sdf_dir}/{row.sequence}.sdf"
        if sdf_path in irregular_backbone_sdf:
            continue
        # sdf_path = "/media/liuwei/1T/sdf_and_json/meV.d(N->O)Val.meV.d(N->O)Val.meV.d(N->O)Val.sdf"
        print(f'dealing with {idx} sdf:', sdf_path)
        # if not filename.endswith(".sdf"):
        #     continue
        # sdf_path = os.path.join(ip_dir, filename)
        c_list = read_sdf(sdf_path)  # read all conformations

        # Use the first conformation to build bond graph and detect backbone
        bonds, atom_types = infer_bonds(c_list[0])
        graph = build_graph(atom_types, bonds)
        backbone_set = find_backbone_cycle(graph, atom_types, residue_len=row.num_monomers)
        if len(backbone_set) == 0:
            with open('Irregular_Backbone_CREMP.txt', 'a') as log:
                log.write(f'{sdf_path}\n')
            continue

        angle_list = []
        # Compute torsion angle (ω) for each conformation
        for conf in c_list:
            angle_list += torsion_angle(conf, backbone_set)
        hist, bin_edges = np.histogram(angle_list, bins=bins, range=range_degrees)
        hist_total += hist  # accumulate

    hist_tensor = torch.tensor(hist_total, dtype=torch.float32)
    edges_tensor = torch.tensor(bin_edges, dtype=torch.float32)

    # Save to a .pt file
    torch.save({'hist_total': hist_tensor, 'bin_edges': edges_tensor}, 'omega_histogram_cremp.pt')
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return hist_total, bin_centers, bin_edges


def omega_distribution_cycpeptmpdb(bins=360, range_degrees=(0, 360)):
    hist_total = np.zeros(bins)
    bin_edges = None

    try:
        with open('Irregular_Backbone_CycPeptMPDB.txt', 'r') as log:
            irregular_backbone_mol = [line.strip() for line in log]
    except FileNotFoundError:
        irregular_backbone_mol = []

    for env, suffix in zip(['water', 'vacuum', 'chloroform'], ['_H2O', '', '_CHCl3']):
        # Go through all the files ending with .sdf under ip_dir
        mol_dir = f'/media/liuwei/1T/3d_data_cycpeptmpdb/content/data/{env}'
        df = pd.read_csv('../DataProcessor/CycPeptMPDB_Peptide_All.csv')
        for idx, row in df.iterrows():
            mol_path = f"{mol_dir}/CycPeptMPDB_ID_{row.CycPeptMPDB_ID}{suffix}.mol"
            if mol_path in irregular_backbone_mol:
                continue
            if not os.path.exists(mol_path):
                continue
            print(f'dealing with {idx} mol:', mol_path)
            c_list = read_sdf(mol_path)  # read all conformations

            # Use the first conformation to build bond graph and detect backbone
            bonds, atom_types = infer_bonds(c_list[0])
            graph = build_graph(atom_types, bonds)

            # Only for Circle shape, Lariat shape will be treated as irregular-backbone
            backbone_set = find_backbone_cycle(graph, atom_types, residue_len=row.Monomer_Length)

            if len(backbone_set) == 0 and mol_path not in irregular_backbone_mol:
                with open('Irregular_Backbone_CycPeptMPDB.txt', 'a') as log:
                    log.write(f'{mol_path}\n')
                continue

            angle_list = []
            # Compute torsion angle (ω) for each conformation
            for conf in c_list:
                angle_list += torsion_angle(conf, backbone_set)
            hist, bin_edges = np.histogram(angle_list, bins=bins, range=range_degrees)
            hist_total += hist  # accumulate

    hist_tensor = torch.tensor(hist_total, dtype=torch.float32)
    edges_tensor = torch.tensor(bin_edges, dtype=torch.float32)

    # Save to a .pt file
    # torch.save({'hist_total': hist_tensor, 'bin_edges': edges_tensor}, 'omega_histogram_cycpeptmpdb_chcl3.pt')
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return hist_total, bin_centers, bin_edges


def omega_distribution_mdtraj(bins=360, range_degrees=(0, 360)):
    hist_total = np.zeros(bins)

    log_name = 'Irregular_Backbone_MDTraj.txt'
    try:
        with open(log_name, 'r') as log:
            irregular_backbone_mol = [line.strip() for line in log]
    except FileNotFoundError:
        irregular_backbone_mol = []

    pdb_dir = '../../../Data/Hexane'
    df = pd.read_csv('../DataProcessor/CycPeptMPDB_Peptide_All.csv', low_memory=False)
    for idx, row in df.iterrows():
        pdb_path = f"{pdb_dir}/{row.Source}_{row.CycPeptMPDB_ID}.pdb"
        if pdb_path in irregular_backbone_mol:
            continue
        if not os.path.exists(pdb_path):
            # print(pdb_path)
            continue
        print(f'dealing with {idx} mol:', pdb_path)
        c_list = read_pdb(pdb_path)  # read all conformations

        # Use the first conformation to build bond graph and detect backbone
        bonds, atom_types = infer_bonds(c_list[0])
        graph = build_graph(atom_types, bonds)
        backbone_set = find_backbone_cycle(graph, atom_types, residue_len=row.Monomer_Length_in_Main_Chain)
        if len(backbone_set) == 0 and pdb_path not in irregular_backbone_mol:
            quit()
            with open(log_name, 'a') as log:
                log.write(f'{pdb_path}\n')
            continue

    #     angle_list = []
    #     # Compute torsion angle (ω) for each conformation
    #     for conf in c_list:
    #         angle_list += torsion_angle(conf, backbone_set)
    #     # if max(angle_list) > 200:
    #     #     exit(pdb_path)
    #     hist, bin_edges = np.histogram(angle_list, bins=bins, range=range_degrees)
    #     hist_total += hist  # accumulate
    #
    # hist_tensor = torch.tensor(hist_total, dtype=torch.float32)
    # edges_tensor = torch.tensor(bin_edges, dtype=torch.float32)
    #
    # # Save to a .pt file
    # torch.save({'hist_total': hist_tensor, 'bin_edges': edges_tensor}, 'omega_histogram_mdpdb_hexen.pt')
    # bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    #
    # return hist_total, bin_centers, bin_edges


if __name__ == "__main__":
    # hist_total, bin_centers, bin_edges = omega_distribution_cremp()
    # hist_total, bin_centers, bin_edges = omega_distribution_cycpeptmpdb()
    hist_total, bin_centers, bin_edges = omega_distribution_mdtraj()

    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers, hist_total, width=(bin_edges[1] - bin_edges[0]), align='center', edgecolor='k')
    plt.xlabel("ω Torsion Angle (degrees)")
    plt.ylabel("Frequency")
    plt.title("Distribution of ω Angles Across Cyclic Peptides")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
