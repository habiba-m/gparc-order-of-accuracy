"""
Training script for G-PARC on advection benchmarks.

Usage:
    python train.py --case gaussian2d
    python train.py --case tophat1d --epochs 200 --hidden_dim 128

The script trains on single-step (s_t → s_{t+1}) pairs. For multi-step
autoregressive training (K > 1 steps, as used in the paper), the rollout
function in gparc/model.py can be called inside a custom training loop.
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from gparc.model import GPARC


# ── Data loading ─────────────────────────────────────────────────────────────

def load_split(split_dir: Path) -> list:
    """Flatten all simulation .pt files in a directory into a list of snapshots."""
    snapshots = []
    for path in sorted(split_dir.glob("*.pt")):
        sim = torch.load(path, weights_only=False)
        snapshots.extend(sim)
    return snapshots


# ── Metrics ──────────────────────────────────────────────────────────────────

def rrmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    num   = (pred - target).pow(2).mean().sqrt()
    denom = target.pow(2).mean().sqrt() + 1e-8
    return (num / denom).item()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> GPARC:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    case     = args.case

    print(f"Loading '{case}' dataset …")
    train_data = load_split(data_dir / case / "train")
    val_data   = load_split(data_dir / case / "val")
    print(f"  Train snapshots: {len(train_data):,}  |  Val: {len(val_data):,}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_data,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = GPARC(
        state_dim      = 1,
        static_dim     = 4,
        hidden_dim     = args.hidden_dim,
        source_out_dim = args.source_dim,
        n_gnn_layers   = args.gnn_layers,
        n_mlp_layers   = args.mlp_layers,
        dt             = 0.01,
        integrator     = args.integrator,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"G-PARC parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.MSELoss()

    save_dir  = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"gparc_{case}_best.pt"

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred  = model(batch)
            loss  = criterion(pred, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred  = model(batch)
                val_loss += criterion(pred, batch.y).item()
        val_loss /= len(val_loader)

        scheduler.step()

        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"train {train_loss:.4e} | val {val_loss:.4e} | lr {lr_now:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    print(f"\nBest val loss: {best_val_loss:.4e}")
    print(f"Checkpoint saved to {save_path}")

    # Restore best weights before returning
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    return model


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train G-PARC on advection benchmarks")

    # Data
    parser.add_argument("--data_dir",   default="data/advection_data",
                        help="Root directory of the advection dataset")
    parser.add_argument("--case",       default="gaussian2d",
                        choices=["gaussian2d", "tophat1d"],
                        help="Which advection benchmark to train on")
    parser.add_argument("--save_dir",   default="checkpoints",
                        help="Directory to save model checkpoints")

    # Training
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)

    # Architecture
    parser.add_argument("--hidden_dim",  type=int, default=64)
    parser.add_argument("--source_dim",  type=int, default=32)
    parser.add_argument("--gnn_layers",  type=int, default=2)
    parser.add_argument("--mlp_layers",  type=int, default=3)
    parser.add_argument("--integrator",  default="euler",
                        choices=["euler", "heun", "rk4"])

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
