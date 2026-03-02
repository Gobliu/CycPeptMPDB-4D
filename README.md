# CycPeptMPDB-4D

A 4D conformational database of cyclic peptides with membrane permeability data.
CycPeptMPDB-4D extends [CycPeptMPDB](https://www.biosino.org/CycPeptMPDB/) by
adding MD-derived 3D conformations in two solvent environments (hexane and water)
for 5,160 cyclic peptides, along with computed molecular descriptors.

- **Publication:** (Link to be added upon publication)
- **Dataset:** [Zenodo](https://doi.org/10.5281/zenodo.18754430)

## Dataset overview

Each peptide has been simulated in both hexane and water. The dataset provides:

- **Trajectories** (100 frames per peptide per environment)
- **Representative structures** extracted by conformational clustering
- **Molecular descriptors** computed from the trajectories (RMSD, surface area, MMPBSA, dihedral angles)
- **PAMPA permeability** values from the original CycPeptMPDB

### Data directory layout

```
CycPeptMPDB-4D/
├── Water/                      (5,160 peptides)
│   ├── Trajectories/           *.pdb (100 frames each)
│   ├── Structures/             representative conformations
│   └── Logs/                   clustering logs
├── Hexane/                     (5,160 peptides)
│   ├── Trajectories/
│   ├── Structures/
│   └── Logs/
└── CycPeptMPDB-4D.csv          metadata & molecular descriptors
```

File naming pattern: `{Source}_{CycPeptMPDB_ID}_{Env}_Traj.pdb` / `..._Str.pdb`

### Main CSV columns

| Column | Description |
|---|---|
| `CycPeptMPDB_ID` | Unique identifier from the original CycPeptMPDB |
| `Source` | Reference literature for the peptide data |
| `Original_Name_in_Source_Literature` | Original peptide designation in the source literature |
| `Structurally_Unique_ID` | Identifier grouping structurally identical peptides across sources |
| `PAMPA` | Experimental membrane permeability (log cm/s) |
| `Monomer_Length` | Total number of monomers in the peptide |
| `Monomer_Length_in_Main_Chain` | Monomers in the main-chain cycle (differs from `Monomer_Length` for lariat-shaped peptides) |
| `Molecule_Shape` | Structural topology (e.g., Circle, Lariat) |
| `Water_avgRMSD_All` | Average RMSD of all atoms in water (nm) |
| `Water_avgRMSD_BackBone` | Average RMSD of backbone atoms in water (nm) |
| `Hexane_avgRMSD_All` | Average RMSD of all atoms in hexane (nm) |
| `Hexane_avgRMSD_BackBone` | Average RMSD of backbone atoms in hexane (nm) |
| `Desolvation_Free_Energy` | Desolvation free energy via MMPBSA (kcal/mol) |
| `Water_3D_SASA` | Solvent Accessible Surface Area in water (nm²) |
| `Water_3D_PSA` | Polar Surface Area in water (nm²) |
| `Water_3D_NPSA` | Non-Polar Surface Area in water (nm²) |
| `Hexane_3D_SASA` | Solvent Accessible Surface Area in hexane (nm²) |
| `Hexane_3D_PSA` | Polar Surface Area in hexane (nm²) |
| `Hexane_3D_NPSA` | Non-Polar Surface Area in hexane (nm²) |

## Setup

Clone this repository and download the dataset into a sibling `Data/` folder:

```
parent_directory/
├── CycPeptMPDB-4D/              ← this repo (git clone)
└── Data/
    └── CycPeptMPDB_4D/          ← downloaded dataset
        ├── Water/
        ├── Hexane/
        └── ...
```

All scripts in this repo expect the dataset at `../Data/CycPeptMPDB_4D/`
relative to the repository root.

## Repository structure

```
CycPeptMPDB-4D/                  (this repo)
├── csvs/                        CSV data files
│   └── CycPeptMPDB-4D.csv       main dataset
├── train_se3.py                 usage example: SE(3)-Transformer training
├── dataprocessor/               scripts used to build the dataset (not needed for general use)
├── plots/                       scripts used to generate figures
└── pts/                         precomputed dihedral histograms
```

The `dataprocessor/` and `plots/` directories contain internal scripts used to
generate and validate the dataset. Most users only need `csvs/CycPeptMPDB-4D.csv`
and the PDB files. See `train_se3.py` for a complete usage example.

## Usage example: SE(3)-Transformer for PAMPA prediction

`train_se3.py` demonstrates using CycPeptMPDB-4D to train an
[SE(3)-Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/DGLPyTorch/DrugDiscovery/SE3Transformer)
that predicts PAMPA permeability from 3D structures.

### Prerequisites

```bash
# 1. Create and activate the conda environment (Python 3.11 + PyTorch 2.4.1 + CUDA 12.4)
conda env create -f environment.yml
conda activate cycpeptmpdb-4d

# 2. Install the NVIDIA SE3-Transformer
git clone --depth 1 https://github.com/NVIDIA/DeepLearningExamples.git /tmp/DeepLearningExamples
pip install -e /tmp/DeepLearningExamples/DGLPyTorch/DrugDiscovery/SE3Transformer
```

### Training

```bash
# Train with defaults (hexane, 100 epochs)
python train_se3.py

# Train on water conformations with a larger model
python train_se3.py --env water --num_layers 7 --num_degrees 4 --num_channels 32 --num_heads 8 --epochs 200
```

Run `python train_se3.py --help` for all available options.
