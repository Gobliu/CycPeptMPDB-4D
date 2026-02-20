#!/usr/bin/env python3
"""
Train an SE(3)-Transformer to predict PAMPA permeability from cyclic peptide
3D structures (PDB files).

Usage:
    python train_se3.py                             # hexane (default)
    python train_se3.py --env water                 # water conformations
    python train_se3.py --epochs 200 --lr 1e-3
    python train_se3.py --clip_min -8 --clip_max -4 # clip PAMPA then map to [-1, 1]
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Bio.PDB import PDBParser
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from se3_transformer.model.fiber import Fiber
from se3_transformer.model.transformer import SE3TransformerPooled

# ---------------------------------------------------------------------------
# Paths (script-relative)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
CSV_PATH = REPO_ROOT / "csvs" / "CycPeptMPDB-4D_clean.csv"
DATA_DIR = REPO_ROOT.parent / "Data" / "CycPeptMPDB_4D"

# Environment-specific settings: (structures subdir, PDB filename suffix regex)
ENV_CONFIG = {
    "hexane": ("Hexane", r"_(\d+)_Hexane_Str\.pdb$"),
    "water":  ("Water",  r"_(\d+)_H2O_Str\.pdb$"),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Atom-level feature encoding
# ---------------------------------------------------------------------------
ATOM_TYPES = ["C", "N", "O", "S", "H"]  # covers >99 % of cyclic peptide atoms
NUM_ATOM_TYPES = len(ATOM_TYPES) + 1     # +1 for "other"
NODE_FEAT_DIM = NUM_ATOM_TYPES           # one-hot atom type


def one_hot_atom(element: str) -> List[float]:
    """One-hot encode atom element type."""
    vec = [0.0] * NUM_ATOM_TYPES
    el = element.strip().upper()
    if el in ATOM_TYPES:
        vec[ATOM_TYPES.index(el)] = 1.0
    else:
        vec[-1] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Gaussian RBF for edge features
# ---------------------------------------------------------------------------
NUM_RBF = 16  # number of radial basis functions


def gaussian_rbf(distances: Tensor, cutoff: float, num_rbf: int = NUM_RBF) -> Tensor:
    """Expand distances into Gaussian radial basis functions.

    Args:
        distances: (E,) pairwise distances
        cutoff: radius cutoff used for graph construction
        num_rbf: number of RBF centres

    Returns:
        (E, num_rbf) tensor of RBF-expanded distances
    """
    centres = torch.linspace(0.0, cutoff, num_rbf, device=distances.device)
    width = 0.5 * (cutoff / num_rbf)
    return torch.exp(-((distances.unsqueeze(-1) - centres) ** 2) / (2 * width**2))


EDGE_FEAT_DIM = NUM_RBF  # Gaussian RBF expansion

# ---------------------------------------------------------------------------
# PDB → DGL graph conversion
# ---------------------------------------------------------------------------
_pdb_parser = PDBParser(QUIET=True)


def pdb_to_graph(
    pdb_path: str | Path,
    cutoff: float = 8.0,
    heavy_only: bool = True,
) -> Tuple[dgl.DGLGraph, Dict[str, Tensor], Dict[str, Tensor]]:
    """Parse a PDB file and build a radius graph for SE3-Transformer.

    Args:
        pdb_path: path to the PDB file
        cutoff: distance cutoff in Angstroms for building edges
        heavy_only: if True, exclude hydrogen atoms (speeds up training)

    Returns:
        graph:      DGLGraph with edata['rel_pos'] set
        node_feats: {'0': Tensor [N, NODE_FEAT_DIM, 1]}
        edge_feats: {'0': Tensor [E, EDGE_FEAT_DIM, 1]}
    """
    structure = _pdb_parser.get_structure("mol", str(pdb_path))

    elements: List[str] = []
    coords: List[List[float]] = []
    for atom in structure.get_atoms():
        el = atom.element.strip().upper()
        if heavy_only and el == "H":
            continue
        elements.append(el)
        coords.append(atom.get_vector().get_array().tolist())

    coords_t = torch.tensor(coords, dtype=torch.float32)  # (N, 3)
    num_atoms = len(elements)

    # Build radius graph: connect atoms within cutoff
    dist_matrix = torch.cdist(coords_t, coords_t)  # (N, N)
    mask = (dist_matrix < cutoff) & (dist_matrix > 0)
    src, dst = torch.where(mask)

    graph = dgl.graph((src, dst))

    # Relative positions (required by SE3-Transformer for basis computation)
    rel_pos = coords_t[dst] - coords_t[src]
    graph.edata["rel_pos"] = rel_pos

    # Node features: one-hot atom type -> (N, NODE_FEAT_DIM, 1)
    node_feat_list = [one_hot_atom(el) for el in elements]
    node_feats = {"0": torch.tensor(node_feat_list, dtype=torch.float32).unsqueeze(-1)}

    # Edge features: Gaussian RBF of distance -> (E, EDGE_FEAT_DIM, 1)
    distances = rel_pos.norm(dim=-1)
    edge_feats = {"0": gaussian_rbf(distances, cutoff).unsqueeze(-1)}

    return graph, node_feats, edge_feats


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CycPeptDataset(Dataset):
    """Dataset that pairs PDB structures with PAMPA permeability targets."""

    def __init__(
        self,
        csv_path: Path,
        structure_dir: Path,
        pdb_pattern: str,
        cutoff: float = 8.0,
        heavy_only: bool = True,
        clip_min: float = -8.0,
        clip_max: float = -4.0,
    ):
        super().__init__()
        self.structure_dir = structure_dir
        self.cutoff = cutoff
        self.heavy_only = heavy_only
        self.clip_min = clip_min
        self.clip_max = clip_max

        df = pd.read_csv(csv_path)

        # Build a mapping from CycPeptMPDB_ID → PDB filename
        pdb_files = {f.name: f for f in structure_dir.glob("*.pdb")}
        id_to_pdb: Dict[int, Path] = {}
        for fname, fpath in pdb_files.items():
            m = re.search(pdb_pattern, fname)
            if m:
                id_to_pdb[int(m.group(1))] = fpath

        # Keep only rows that have a matching PDB file
        valid_ids = []
        valid_targets = []
        valid_paths = []
        for _, row in df.iterrows():
            pid = int(row["CycPeptMPDB_ID"])
            if pid in id_to_pdb:
                valid_ids.append(pid)
                valid_targets.append(float(row["PAMPA"]))
                valid_paths.append(id_to_pdb[pid])

        raw_targets = torch.tensor(valid_targets, dtype=torch.float32)

        # Clip to [clip_min, clip_max] then linearly map to [-1, 1]
        clipped = raw_targets.clamp(clip_min, clip_max)
        self.targets = 2.0 * (clipped - clip_min) / (clip_max - clip_min) - 1.0
        n_clipped = (raw_targets < clip_min).sum().item() + (raw_targets > clip_max).sum().item()

        self.ids = valid_ids
        self.pdb_paths = valid_paths
        log.info(
            "Dataset: %d samples (matched %d/%d IDs to PDB files, %d clipped to [%.1f, %.1f])",
            len(self.ids), len(self.ids), len(df), n_clipped, clip_min, clip_max,
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        graph, node_feats, edge_feats = pdb_to_graph(
            self.pdb_paths[idx],
            cutoff=self.cutoff,
            heavy_only=self.heavy_only,
        )
        target = self.targets[idx]
        return graph, node_feats, edge_feats, target


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------
def collate_fn(
    samples: List[Tuple[dgl.DGLGraph, dict, dict, Tensor]],
) -> Tuple[dgl.DGLGraph, Dict[str, Tensor], Dict[str, Tensor], Tensor]:
    """Batch a list of (graph, node_feats, edge_feats, target) into one."""
    graphs, nf_list, ef_list, targets = zip(*samples)

    batched_graph = dgl.batch(graphs)
    batched_node_feats = {"0": torch.cat([nf["0"] for nf in nf_list], dim=0)}
    batched_edge_feats = {"0": torch.cat([ef["0"] for ef in ef_list], dim=0)}
    targets = torch.stack(targets)

    return batched_graph, batched_node_feats, batched_edge_feats, targets


# ---------------------------------------------------------------------------
# Training / validation helpers
# ---------------------------------------------------------------------------
def _norm_to_real(preds: Tensor, clip_min: float, clip_max: float) -> Tensor:
    """Map predictions from [-1, 1] back to [clip_min, clip_max] PAMPA scale."""
    return (preds + 1.0) / 2.0 * (clip_max - clip_min) + clip_min


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    clip_min: float,
    clip_max: float,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    for batch in loader:
        graph, node_feats, edge_feats, targets = batch
        graph = graph.to(device)
        node_feats = {k: v.to(device) for k, v in node_feats.items()}
        edge_feats = {k: v.to(device) for k, v in edge_feats.items()}
        targets = targets.to(device)  # already in [-1, 1]

        preds = model(graph, node_feats, edge_feats)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item() * len(targets)
        n_samples += len(targets)

    return total_loss / n_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    clip_min: float,
    clip_max: float,
) -> Tuple[float, float]:
    """Returns (normalised_loss, real-scale MAE)."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n_samples = 0
    for batch in loader:
        graph, node_feats, edge_feats, targets = batch
        graph = graph.to(device)
        node_feats = {k: v.to(device) for k, v in node_feats.items()}
        edge_feats = {k: v.to(device) for k, v in edge_feats.items()}
        targets = targets.to(device)

        preds = model(graph, node_feats, edge_feats)
        loss = loss_fn(preds, targets)

        # Convert both to real PAMPA scale for interpretable MAE
        preds_real = _norm_to_real(preds, clip_min, clip_max)
        targets_real = _norm_to_real(targets, clip_min, clip_max)
        mae = (preds_real - targets_real).abs().sum().item()

        total_loss += loss.item() * len(targets)
        total_mae += mae
        n_samples += len(targets)

    return total_loss / n_samples, total_mae / n_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SE(3)-Transformer for PAMPA prediction")

    # Data
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Path to CycPeptMPDB-4D_clean.csv")
    parser.add_argument("--env", type=str, default="hexane", choices=list(ENV_CONFIG.keys()),
                        help="Solvent environment: hexane or water")
    parser.add_argument("--structures", type=Path, default=None,
                        help="Override PDB structures directory (default: auto from --env)")
    parser.add_argument("--cutoff", type=float, default=8.0, help="Radius cutoff for graph construction (Angstroms)")
    parser.add_argument("--heavy_only", action="store_true", default=True, help="Use only heavy atoms (drop H)")
    parser.add_argument("--no_heavy_only", action="store_true", help="Include hydrogen atoms")
    parser.add_argument("--clip_min", type=float, default=-8.0, help="PAMPA clip lower bound")
    parser.add_argument("--clip_max", type=float, default=-4.0, help="PAMPA clip upper bound")

    # Model
    parser.add_argument("--num_layers", type=int, default=4, help="Number of SE3 attention layers")
    parser.add_argument("--num_degrees", type=int, default=3, help="Number of degrees for hidden features")
    parser.add_argument("--num_channels", type=int, default=16, help="Hidden channels per degree")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--channels_div", type=int, default=2, help="Channels division factor")
    parser.add_argument("--pooling", type=str, default="avg", choices=["avg", "max"])
    parser.add_argument("--norm", action="store_true", default=True, help="Normalization after attention blocks")
    parser.add_argument("--use_layer_norm", action="store_true", default=True)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--test_frac", type=float, default=0.1, help="Test fraction")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (0 to disable)")
    parser.add_argument("--save_dir", type=Path, default=REPO_ROOT / "checkpoints")

    args = parser.parse_args()
    if args.no_heavy_only:
        args.heavy_only = False

    # Resolve env → structures dir and PDB regex
    env_subdir, pdb_pattern = ENV_CONFIG[args.env]
    if args.structures is None:
        args.structures = DATA_DIR / env_subdir / "Structures"

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s  |  Env: %s  |  Structures: %s", device, args.env, args.structures)

    # ── Dataset ─────────────────────────────────────────────────────────
    full_dataset = CycPeptDataset(
        csv_path=args.csv,
        structure_dir=args.structures,
        pdb_pattern=pdb_pattern,
        cutoff=args.cutoff,
        heavy_only=args.heavy_only,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )

    # Split
    n = len(full_dataset)
    n_test = int(n * args.test_frac)
    n_val = int(n * args.val_frac)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    log.info("Split: train=%d  val=%d  test=%d", n_train, n_val, n_test)
    log.info("PAMPA targets clipped to [%.1f, %.1f] → normalised to [-1, 1]",
             args.clip_min, args.clip_max)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # ── Model ───────────────────────────────────────────────────────────
    model = SE3TransformerPooled(
        fiber_in=Fiber({0: NODE_FEAT_DIM}),
        fiber_out=Fiber({0: args.num_degrees * args.num_channels}),
        fiber_edge=Fiber({0: EDGE_FEAT_DIM}),
        num_degrees=args.num_degrees,
        num_channels=args.num_channels,
        output_dim=1,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        channels_div=args.channels_div,
        pooling=args.pooling,
        norm=args.norm,
        use_layer_norm=args.use_layer_norm,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %s", f"{n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6,
    )
    loss_fn = nn.L1Loss()  # MAE

    # ── Training loop ───────────────────────────────────────────────────
    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_val_mae = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, args.clip_min, args.clip_max,
        )
        val_loss, val_mae = evaluate(model, val_loader, loss_fn, device, args.clip_min, args.clip_max)
        scheduler.step(val_mae)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        log.info(
            "Epoch %3d/%d | train_loss %.4f | val_loss %.4f | val_MAE %.4f | lr %.1e | %.1fs",
            epoch, args.epochs, train_loss, val_loss, val_mae, lr, elapsed,
        )

        # Checkpoint best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            ckpt_path = args.save_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae": val_mae,
                    "clip_min": args.clip_min,
                    "clip_max": args.clip_max,
                    "args": vars(args),
                },
                ckpt_path,
            )
            log.info("  ↳ Saved best model (val_MAE=%.4f)", val_mae)
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                log.info("Early stopping at epoch %d (patience=%d)", epoch, args.patience)
                break

    # ── Test evaluation ─────────────────────────────────────────────────
    ckpt = torch.load(args.save_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_mae = evaluate(model, test_loader, loss_fn, device, args.clip_min, args.clip_max)
    log.info("=" * 60)
    log.info("Test MAE: %.4f  (best val MAE: %.4f at epoch %d)", test_mae, ckpt["val_mae"], ckpt["epoch"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()
