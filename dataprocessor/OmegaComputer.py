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

    # Collect patterns, deduplicating by the N atom index (atom 2 in CA-CO-N-CA).
    # Multiple simple cycles of the same length can share backbone atoms when
    # side-chain rings create alternative paths, causing the same peptide bond
    # to appear more than once.
    seen_n = {}
    for c in tgt_cycles:
        for i in range(len(c)):
            names = [_classify_atom(graph, atom_types, c[(i + k) % loop_len]) for k in range(4)]
            if names == ['CA', 'CO', 'N', 'CA']:
                atom_set = [c[(i + k) % loop_len] for k in range(4)]
                n_idx = atom_set[2]
                if n_idx not in seen_n:
                    seen_n[n_idx] = atom_set
    omega_atom_set = list(seen_n.values())

    assert len(omega_atom_set) > 0, 'No omega atom set found'
    assert len(omega_atom_set) <= residue_len, f'Incorrect number of omega atom set: {len(omega_atom_set)} > {residue_len}'
    return omega_atom_set


# ── Torsion angle ────────────────────────────────────────────────────────────

def torsion_angle(conf, atom_sets):
    angles = []
    for a0, a1, a2, a3 in atom_sets:
        p0 = np.asarray(conf[a0][:3], dtype=float)
        p1 = np.asarray(conf[a1][:3], dtype=float)
        p2 = np.asarray(conf[a2][:3], dtype=float)
        p3 = np.asarray(conf[a3][:3], dtype=float)

        # atom_set order is [CA(i), C(i), N(i+1), CA(i+1)], so:
        #   p0=CA(i), p1=C(i), p2=N(i+1), p3=CA(i+1)
        v01 = p1 - p0  # CA(i)   -> C(i)
        v12 = p2 - p1  # C(i)    -> N(i+1)   [the peptide bond axis]
        v23 = p3 - p2  # N(i+1)  -> CA(i+1)
        # Normals to each successive plane
        n_012 = np.cross(v01, v12)  # plane of CA(i)-C(i)-N(i+1)
        n_123 = np.cross(v12, v23)  # plane of C(i)-N(i+1)-CA(i+1)
        v12_hat = v12 / np.linalg.norm(v12)  # unit vector along C(i)->N(i+1)

        # Praxitelous/Blondel formula: atan2((n_012 x v12_hat)·n_123, n_012·n_123)
        angle = np.degrees(np.arctan2(np.dot(np.cross(n_012, v12_hat), n_123), np.dot(n_012, n_123))) % 360
        angles.append(angle)
    return angles


# ── Distribution functions ───────────────────────────────────────────────────

def _init_irregular_log(log_name):
    """Clear the irregular backbone log at the start of each run (creating logs/ if absent)."""
    Path(log_name).parent.mkdir(parents=True, exist_ok=True)
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
    df = pd.read_csv(REPO_ROOT / "csvs" / 'cremp_sequences.csv', low_memory=False)

    for _, row in df.iterrows():
        sdf_path = f"{sdf_dir}/{row.sequence}.sdf"
        if sdf_path in irregular:
            continue
        c_list = read_sdf(sdf_path)
        bonds, atom_types = infer_bonds(c_list[0])
        graph = build_graph(atom_types, bonds)
        residue_len = len(row.sequence.split('.'))  # monomer count = dot-separated tokens
        backbone_set = find_backbone_cycle(graph, atom_types, residue_len=residue_len)
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


def _load_n_extra_from_4d():
    """Map CycPeptMPDB_ID -> n_extra (count of BHF/TNH backbone-extending residues).

    n_extra is read from residue names in the 4D representative structures; the raw
    .mol files carry no residue names, so it cannot be derived from them directly.
    """
    df4 = pd.read_csv(REPO_ROOT / "csvs" / 'CycPeptMPDB-4D.csv', low_memory=False)
    str_dir = DATA_DIR / 'CycPeptMPDB_4D' / 'June2026' / 'Water' / 'Structures'
    n_extra_by_id = {}
    for _, r in df4.iterrows():
        pdb = str_dir / f"{str(r.Source).strip()}_{r.CycPeptMPDB_ID}_H2O_Str.pdb"
        if pdb.exists():
            n_extra_by_id[r.CycPeptMPDB_ID] = count_extra_backbone_atoms(str(pdb))
    return n_extra_by_id


def omega_distribution_cycpeptmpdb(bins=360, range_degrees=(0, 360)):
    hist_total = np.zeros(bins)
    bin_edges = None
    log_name = str(REPO_ROOT / 'logs' / f'Irregular_Backbone_CycPeptMPDB.txt')
    irregular = _init_irregular_log(log_name)

    # Backbone ring length = Monomer_Length_in_Main_Chain (not the full Monomer_Length,
    # which is wrong for Lariats) + n_extra backbone-extending atoms (BHF/TNH).
    n_extra_by_id = _load_n_extra_from_4d()

    missing_count = 0
    irregular_count = 0
    df = pd.read_csv(REPO_ROOT / "csvs" / 'CycPeptMPDB_Peptide_All.csv',
                     low_memory=False, encoding='utf-8-sig')
    for env, suffix in [('water', '_H2O'), ('vacuum', ''), ('chloroform', '_CHCl3')]:
        mol_dir = str(DATA_DIR / 'cycpeptmpdb_3d' / 'content' / 'data' / env)
        for _, row in df.iterrows():
            mol_path = f"{mol_dir}/CycPeptMPDB_ID_{row.CycPeptMPDB_ID}{suffix}.mol"
            if not os.path.exists(mol_path):
                missing_count += 1
                continue
            if mol_path in irregular:
                continue
            c_list = read_sdf(mol_path)
            bonds, atom_types = infer_bonds(c_list[0])
            graph = build_graph(atom_types, bonds)
            residue_len = int(row.Monomer_Length_in_Main_Chain
                              if pd.notna(row.Monomer_Length_in_Main_Chain)
                              else row.Monomer_Length)
            n_extra = n_extra_by_id.get(row.CycPeptMPDB_ID, 0)
            backbone_set = find_backbone_cycle(graph, atom_types,
                                               residue_len=residue_len, n_extra=n_extra)
            if not backbone_set:
                irregular_count += 1
                with open(log_name, 'a') as f:
                    f.write(f'{mol_path}\n')
                continue

            angle_list = []
            for conf in c_list:
                angle_list += torsion_angle(conf, backbone_set)
            bin_edges = _accumulate_histogram(hist_total, angle_list, bins, range_degrees)

    print(f"Missing files:   {missing_count}")
    print(f"Irregular backbone (no cycle found): {irregular_count}")
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


# ── Test / inspection helper ─────────────────────────────────────────────────

def inspect_pdb_omegas(pdb_path, residue_len, n_extra=None):
    """Print residue name + omega angle for each backbone peptide bond in MODEL 0.

    Each omega is labelled by the residue that owns the N atom in the
    CA→N→CO→CA dihedral (i.e. the acceptor residue of the peptide bond).

    Usage::
        inspect_pdb_omegas("path/to/peptide.pdb", residue_len=6)
    """
    pdb_path = str(pdb_path)

    # Parse atom metadata from first MODEL only
    atom_meta = []   # list of (res_num, res_name, atom_name, x, y, z, element)
    with open(pdb_path) as f:
        in_first_model = False
        past_first_model = False
        for line in f:
            if line.startswith('MODEL'):
                if in_first_model:
                    past_first_model = True
                    break
                in_first_model = True
                continue
            if past_first_model:
                break
            if line.startswith('ENDMDL'):
                break
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            res_name = line[17:20].strip()
            res_num  = int(line[22:26])
            atom_name = line[12:16].strip()
            x, y, z  = float(line[30:38]), float(line[38:46]), float(line[46:54])
            element  = line[76:78].strip() or atom_name[0]
            atom_meta.append((res_num, res_name, atom_name, x, y, z, element))

    # Build conf and atom_types for the existing helpers
    conf       = [[m[3], m[4], m[5], m[6]] for m in atom_meta]
    atom_types = [m[6] for m in atom_meta]

    if n_extra is None:
        # Auto-count from residue names in this PDB
        seen = {}
        for res_num, res_name, *_ in atom_meta:
            seen[res_num] = res_name
        n_extra = sum(1 for name in seen.values() if name in ('BHF', 'TNH'))

    bonds = infer_bonds(conf)[0]
    graph = build_graph(atom_types, bonds)
    omega_sets = find_backbone_cycle(graph, atom_types, residue_len, n_extra)

    if not omega_sets:
        print("No backbone cycle found.")
        return

    angles = torsion_angle(conf, omega_sets)

    # omega dihedral atom order: CA(i)→C(i)→N(i+1)→CA(i+1)  [IUPAC order]
    # atom index 2 in each set is N(i+1) — use its residue as the label
    print(f"{'#':<4}  {'Res':>4}  {'ResName':<8}  {'omega (deg)':>12}")
    print("-" * 36)
    for k, (atom_set, angle) in enumerate(zip(omega_sets, angles), 1):
        n_idx = atom_set[2]
        res_num, res_name = atom_meta[n_idx][0], atom_meta[n_idx][1]
        print(f"{k:<4}  {res_num:>4}  {res_name:<8}  {angle:>12.2f}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_DIR_4D = DATA_DIR / "CycPeptMPDB_4D"
    CSV_PATH = str(REPO_ROOT / "csvs" / "CycPeptMPDB-4D.csv")
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

    # inspect_pdb_omegas(DATA_DIR_4D / "Water" / "Structures" / "2015_Wang_1048_H2O_Str.pdb", residue_len=6)