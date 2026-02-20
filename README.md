# CycPeptMPDB-4D

A 4D conformational database of cyclic peptides with membrane permeability data.
CycPeptMPDB-4D extends [CycPeptMPDB](https://www.biosino.org/CycPeptMPDB/) by
adding MD-derived 3D conformations in two solvent environments (hexane and water)
for 5,160 cyclic peptides, along with computed molecular descriptors.

- **Publication:** (Link to be added upon publication)
- **Dataset:** (Link coming soon)

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
| `CycPeptMPDB_ID` | Unique peptide identifier |
| `Source` | Literature source |
| `SMILES` | Molecular structure |
| `PAMPA` | Log membrane permeability |
| `Monomer_Length` | Number of residues |
| `Water_avgRMSD_All`, `Water_avgRMSD_BackBone` | Average RMSD in water (all atoms / backbone) |
| `Hexane_avgRMSD_All`, `Hexane_avgRMSD_BackBone` | Average RMSD in hexane |
| `Water_3D_SASA`, `Water_3D_NPSA`, `Water_3D_PSA` | Solvent-accessible surface areas in water |
| `Hexane_3D_SASA`, `Hexane_3D_NPSA`, `Hexane_3D_PSA` | Surface areas in hexane |

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
pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html
pip install e3nn==0.3.3 biopython

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
