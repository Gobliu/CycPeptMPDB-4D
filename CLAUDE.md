# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CycPeptMPDB-4D is a 4D conformational database of cyclic peptides with membrane permeability (PAMPA) data. It extends CycPeptMPDB with MD-derived 3D conformations in hexane and water for 5,160 peptides.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate cycpeptmpdb-4d

# SE3-Transformer (required for train_se3.py)
git clone --depth 1 https://github.com/NVIDIA/DeepLearningExamples.git /tmp/DeepLearningExamples
pip install -e /tmp/DeepLearningExamples/DGLPyTorch/DrugDiscovery/SE3Transformer
```

Key dependencies: Python 3.11, PyTorch 2.4.1 + CUDA 12.4, DGL, BioPython, e3nn==0.3.3, torchdata==0.9.0.

## Running the Training Example

```bash
python train_se3.py                          # hexane (default), 100 epochs
python train_se3.py --env water              # water conformations
python train_se3.py --help                   # all options
```

## Data Layout

The dataset lives in a **sibling** directory, not inside this repo:
```
parent_directory/
├── CycPeptMPDB-4D/          ← this repo
│   └── csvs/                CSV data files (main: CycPeptMPDB-4D.csv)
└── Data/
    └── CycPeptMPDB_4D/      ← downloaded from Zenodo
        ├── Water/            Trajectories/, Structures/, Logs/
        └── Hexane/           Trajectories/, Structures/, Logs/
```

## Path Convention (IMPORTANT)

**Never use relative `../` path strings.** Always anchor paths using `Path(__file__).resolve().parent`:

```python
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent          # for scripts inside subdirectories
CSV_DIR = REPO_ROOT / "csvs"
DATA_DIR = REPO_ROOT.parent / "Data" / "CycPeptMPDB_4D"
```

## Architecture

### `dataprocessor/` — Data pipeline scripts

Each script works on the raw simulation output (PDB trajectories, GROMACS logs) in the external Data directory — extracting structures, computing dihedrals, or validating completeness. Key scripts:
- **OmegaComputer.py** — computes backbone omega dihedral angles from PDB trajectories using NetworkX bond graphs; uses `Monomer_Length_in_Main_Chain` for backbone cycle detection
- **ExtractMiddleConf.py** — extracts representative (middle) conformations from trajectory clusters
- **MissingDataChecker.py** — validates completeness of the dataset
- **MissingPeptidesExporter.py** — writes `csvs/missing_peptides.csv` (in-scope PAMPA peptides not simulated)

> **Descriptor columns are static.** The SASA/NPSA/PSA, avgRMSD, avgGR, and desolvation columns in `CycPeptMPDB-4D.csv` were produced by a one-off merge from GROMACS metric files that have since been removed; the dedicated readers (ReadSA/ReadMMPBSA/ReadAvgRMSD) were deleted along with them. Regenerating those columns means re-running GROMACS analysis on the trajectories.

### `dataprocessor/utils.py` — Shared alias matching

Peptide entries from different sources use different naming schemes:
- **Kelly/Naylor** entries: matched by `CycPeptMPDB_ID` (integer)
- **Townsend** entries: strip `2020_Townsend_XXXX-` prefix before matching `Original_Name_in_Source_Literature`
- MMPBSA files strip trailing replica suffix `_\d+` for non-Townsend entries
- Some 2022_Taechalertpaisarn peptides have no match (expected)

### `train_se3.py` — Usage example

SE(3)-Transformer that predicts PAMPA from 3D structures. Converts PDB → DGL radius graph (8 Å cutoff), uses one-hot atom types + Gaussian RBF edge features. PAMPA targets are clipped to [-8, -4] and normalized to [-1, 1].

### `plots/` — Figure generation scripts

Plotters for RMSD distributions, omega angles, TPSA, coverage, and NOE distances.

### `pts/` — Precomputed dihedral histograms

## Plotting Style

- Use colorblind-friendly colors (Wong palette): blue `#0072B2`, vermillion `#D55E00`
- Large fonts: titles ~18, axis labels ~16, legends ~14, tick labels ~13
- Sparse x-axis labels — avoid dense/overlapping text

## Function Ordering

Within a script, put higher-level / more general functions on top, detail/helper functions below. The reader should understand the big picture first.

## Column Notes

- `Monomer_Length` vs `Monomer_Length_in_Main_Chain`: differ only for Lariat-shape peptides
- `Molecule_Shape`: structural topology (Circle, Lariat, etc.)
