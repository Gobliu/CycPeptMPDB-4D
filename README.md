# CycPeptMPDB-4D

# Training SE(3)-Transformer for PAMPA Prediction

Predict PAMPA membrane permeability of cyclic peptides from their 3D molecular
structures using NVIDIA's [SE(3)-Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/DGLPyTorch/DrugDiscovery/SE3Transformer).

## Prerequisites

| Package | Version | Purpose |
|---|---|---|
| PyTorch | >= 2.0 | Core framework |
| DGL (CUDA) | 2.1 | Graph neural network library |
| e3nn | 0.3.3 | Spherical harmonics / equivariant basis |
| BioPython | >= 1.80 | PDB file parsing |
| NVIDIA SE3-Transformer | 1.2 | Model architecture (installed from source) |

### Install dependencies

```bash
# DGL with CUDA support (required — CPU-only DGL will fail on .to('cuda'))
pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html

# e3nn (must be 0.3.3 for SE3-Transformer compatibility)
pip install e3nn==0.3.3

# NVIDIA SE3-Transformer from source
git clone --depth 1 https://github.com/NVIDIA/DeepLearningExamples.git /tmp/DeepLearningExamples
pip install -e /tmp/DeepLearningExamples/DGLPyTorch/DrugDiscovery/SE3Transformer

# BioPython for PDB parsing
pip install biopython
```

> **Note for PyTorch >= 2.3:** DGL's graphbolt module may fail to load its C++
> library. If you see `FileNotFoundError: Cannot find DGL C++ graphbolt library`,
> edit the file `<site-packages>/dgl/graphbolt/__init__.py` and wrap the final
> `load_graphbolt()` call in a try/except:
> ```python
> try:
>     load_graphbolt()
> except (FileNotFoundError, ImportError, OSError):
>     pass
> ```
>
> Similarly, if e3nn fails with a `weights_only` error, edit
> `<site-packages>/e3nn/o3/_wigner.py` and change the `torch.load(...)` call to
> `torch.load(..., weights_only=False)`.

## Quick start

```bash
# Train with defaults (hexane, small model, 100 epochs)
python train_se3.py

# Train on water conformations
python train_se3.py --env water

# Full run with a larger model
python train_se3.py --env hexane --num_layers 7 --num_degrees 4 --num_channels 32 --num_heads 8 --epochs 200
```

## Data pipeline

### Input: PDB structures

The script reads 3D coordinates from PDB files located at:

| `--env` | Directory | Filename pattern |
|---|---|---|
| `hexane` (default) | `../Data/CycPeptMPDB_4D/Hexane/Structures/` | `*_<ID>_Hexane_Str.pdb` |
| `water` | `../Data/CycPeptMPDB_4D/Water/Structures/` | `*_<ID>_H2O_Str.pdb` |

Each PDB is converted to a molecular graph:
- **Nodes**: one per heavy atom (H excluded by default), with a 6-dim one-hot
  feature vector encoding element type (C, N, O, S, H, other).
- **Edges**: all atom pairs within a radius cutoff (default 8 A), with 16-dim
  Gaussian RBF features encoding the interatomic distance.
- **Relative positions**: stored in `graph.edata['rel_pos']`, used internally
  by the SE(3)-Transformer to compute equivariant bases.

### Target: PAMPA

Raw PAMPA values from `csvs/CycPeptMPDB-4D_clean.csv` are processed as:

1. **Clip** to `[clip_min, clip_max]` (default `[-8, -4]`). The 265 samples
   (5.1%) with PAMPA < -8 are clamped — these are rare outliers in the long
   tail.
2. **Linearly map** to `[-1, 1]`: `target = 2 * (clipped - clip_min) / (clip_max - clip_min) - 1`

Validation and test MAE are always reported on the **original PAMPA scale**
(inverse-mapped) for interpretability.

### Data split

Default: 80% train / 10% validation / 10% test, controlled by `--val_frac` and
`--test_frac`. Split is deterministic (seeded by `--seed`).

## Model configurations

The SE(3)-Transformer model is fully configurable. Here are four reference
configurations:

| Name | `--num_layers` | `--num_degrees` | `--num_channels` | `--num_heads` | Parameters | Use case |
|---|---|---|---|---|---|---|
| **Small** (default) | 4 | 3 | 16 | 4 | 578K | Quick experiments, debugging |
| **Medium** | 5 | 3 | 32 | 8 | 2.9M | Good speed/accuracy tradeoff |
| **Large** | 7 | 4 | 32 | 8 | 9.3M | Production training |
| **XLarge** | 7 | 4 | 64 | 8 | 37M | Maximum capacity, needs more data |

Example commands:

```bash
# Small (default) — ~15s/epoch on RTX 5090
python train_se3.py

# Medium
python train_se3.py --num_layers 5 --num_degrees 3 --num_channels 32 --num_heads 8

# Large
python train_se3.py --num_layers 7 --num_degrees 4 --num_channels 32 --num_heads 8

# XLarge (may need smaller batch size)
python train_se3.py --num_layers 7 --num_degrees 4 --num_channels 64 --num_heads 8 --batch_size 16
```

### What each model parameter controls

- **`--num_layers`**: Depth of the network (number of SE(3) attention blocks).
  More layers = more message-passing rounds = better at capturing long-range
  interactions in the molecular graph. Start with 4, try 5-7.
- **`--num_degrees`**: Number of spherical harmonic degrees in hidden features
  (0, 1, ..., num_degrees-1). Degree 0 = scalar, 1 = vector, 2 = rank-2
  tensor, etc. Higher degrees capture more complex angular dependencies. 3 is
  usually sufficient; 4 adds modest benefit at significant cost.
- **`--num_channels`**: Channel width per degree in hidden layers. Directly
  controls model capacity. 16 is minimal, 32 is solid, 64 is large.
- **`--num_heads`**: Number of attention heads. Must divide `num_channels //
  channels_div`. 4-8 heads work well.
- **`--channels_div`**: Divides channels before the attention layer (default 2).
  Lower values = more parameters in attention, higher = more efficient.
- **`--pooling`**: Graph-level aggregation (`avg` or `max`). `avg` is the
  default and generally works well; `max` can help if a few key atoms dominate
  the property.

## Hyperparameter tuning guide

### Training hyperparameters

| Flag | Default | Range to try | Notes |
|---|---|---|---|
| `--lr` | 2e-4 | 5e-5 to 1e-3 | Most impactful. Reduce if loss is unstable. |
| `--batch_size` | 32 | 16-64 | Larger = faster epochs, may need lr scaling. |
| `--weight_decay` | 1e-5 | 0 to 1e-4 | L2 regularisation. Increase if overfitting. |
| `--epochs` | 100 | 100-300 | Early stopping (patience=20) will halt if needed. |
| `--patience` | 20 | 10-50 | Epochs without val improvement before stopping. |
| `--seed` | 42 | any int | Change seed to assess variance across splits. |

### Graph construction

| Flag | Default | Range to try | Notes |
|---|---|---|---|
| `--cutoff` | 8.0 | 5.0-10.0 | Radius cutoff in Angstroms. Smaller = fewer edges = faster. Larger = captures more non-bonded interactions. |
| `--heavy_only` | True | True/False | Include H atoms with `--no_heavy_only`. Roughly 2x more atoms = 4x more edges = slower, but may help. |

### Target normalisation

| Flag | Default | Notes |
|---|---|---|
| `--clip_min` | -8.0 | Lower bound for PAMPA clipping. |
| `--clip_max` | -4.0 | Upper bound. Together they define the mapping to [-1, 1]. |

To disable clipping entirely and map the full PAMPA range (-10 to -4):
```bash
python train_se3.py --clip_min -10 --clip_max -4
```

## Output

### Checkpoints

Best model is saved to `checkpoints/best_model.pt` (configurable with
`--save_dir`). The checkpoint contains:

```python
{
    "epoch": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "val_mae": float,        # best validation MAE (real PAMPA scale)
    "clip_min": float,       # needed to convert predictions back to PAMPA
    "clip_max": float,
    "args": dict,            # full training configuration for reproducibility
}
```

### Logging

Every epoch logs:

```
Epoch   1/100 | train_loss 0.4752 | val_loss 0.2839 | val_MAE 0.5677 | lr 2.0e-04 | 15.1s
```

- `train_loss` / `val_loss`: MAE in normalised [-1, 1] space.
- `val_MAE`: MAE in **real PAMPA scale** (log units). This is the number to
  compare across experiments.
- `lr`: current learning rate (reduced on plateau).

## Recommended experiment workflow

### 1. Baseline
```bash
python train_se3.py --env hexane --epochs 100
```

### 2. Scale up the model
```bash
python train_se3.py --env hexane --num_layers 7 --num_degrees 4 --num_channels 32 --num_heads 8 --epochs 200
```

### 3. Compare solvent environments
```bash
python train_se3.py --env hexane --save_dir checkpoints/hexane
python train_se3.py --env water  --save_dir checkpoints/water
```

### 4. Sweep learning rate
```bash
for lr in 5e-5 1e-4 2e-4 5e-4 1e-3; do
    python train_se3.py --lr $lr --save_dir checkpoints/lr_${lr} 2>&1 | tail -3
done
```

### 5. Try different cutoffs
```bash
for c in 5.0 6.0 8.0 10.0; do
    python train_se3.py --cutoff $c --save_dir checkpoints/cutoff_${c} 2>&1 | tail -3
done
```

### 6. Assess variance across random seeds
```bash
for s in 42 123 456 789 1024; do
    python train_se3.py --seed $s --save_dir checkpoints/seed_${s} 2>&1 | tail -3
done
```

## Full argument reference

```
python train_se3.py --help
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--csv` | Path | `csvs/CycPeptMPDB-4D_clean.csv` | Path to CSV with PAMPA targets |
| `--env` | str | `hexane` | Solvent environment: `hexane` or `water` |
| `--structures` | Path | auto | Override PDB directory (auto-resolved from `--env`) |
| `--cutoff` | float | 8.0 | Radius cutoff for graph edges (Angstroms) |
| `--heavy_only` | flag | True | Exclude hydrogen atoms |
| `--no_heavy_only` | flag | - | Include hydrogen atoms |
| `--clip_min` | float | -8.0 | PAMPA clip lower bound |
| `--clip_max` | float | -4.0 | PAMPA clip upper bound |
| `--num_layers` | int | 4 | SE(3) attention layers |
| `--num_degrees` | int | 3 | Spherical harmonic degrees |
| `--num_channels` | int | 16 | Hidden channels per degree |
| `--num_heads` | int | 4 | Attention heads |
| `--channels_div` | int | 2 | Channel division factor |
| `--pooling` | str | `avg` | Graph pooling: `avg` or `max` |
| `--norm` | flag | True | Normalisation between attention blocks |
| `--use_layer_norm` | flag | True | Layer norm in MLPs |
| `--epochs` | int | 100 | Maximum training epochs |
| `--batch_size` | int | 32 | Batch size |
| `--lr` | float | 2e-4 | Learning rate |
| `--weight_decay` | float | 1e-5 | AdamW weight decay |
| `--num_workers` | int | 4 | DataLoader workers |
| `--seed` | int | 42 | Random seed |
| `--val_frac` | float | 0.1 | Validation set fraction |
| `--test_frac` | float | 0.1 | Test set fraction |
| `--patience` | int | 20 | Early stopping patience (0 to disable) |
| `--save_dir` | Path | `checkpoints/` | Checkpoint output directory |
