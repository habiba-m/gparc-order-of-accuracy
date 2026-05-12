"""
train.py - G-PARC Advection Training
Train G-PARC to predict one timestep of  u_t + v.grad(u) = 0.
Change PROBLEM at the top to switch between gaussian and disc.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from pathlib import Path

sys.path.insert(0, str(Path.home() / "G-PARC"))
from differentiator.hop import SolveGradientsLST, AdvectionMLS
from differentiator.mappingandrecon import MappingAndRecon
from utilities.featureextractor import GraphConvFeatureExtractorV2
from utilities.embed import SimulationConditionedLayerNorm

PROBLEM   = "disc"      # "gaussian" or "disc"
TRAIN_DIR = f"data/{PROBLEM}/train"
VAL_DIR   = f"data/{PROBLEM}/val"
OUT_DIR   = f"outputs/{PROBLEM}"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS    = 200
LR = 3e-4 if PROBLEM == "gaussian" else 1e-4
SEQ_LEN   = 4       # autoregressive steps per training update
HIDDEN    = 64      # feature extractor hidden/output channels

# DATASET
class AdvectionDataset(IterableDataset):
    def __init__(self, directory):
        files = sorted(Path(directory).glob("*.pt"))
        print(f"  Loading {len(files)} sims into RAM from {directory}...", flush=True)
        self.sims = [torch.load(f, weights_only=False) for f in files]
        n_windows = sum(max(0, len(s) - SEQ_LEN + 1) for s in self.sims)
        print(f"  Done. {n_windows} training windows total.", flush=True)

    def __iter__(self):
        for sim in self.sims:
            for i in range(len(sim) - SEQ_LEN + 1):
                yield [sim[i + k] for k in range(SEQ_LEN)]

# MODEL
class GPARCAdvection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_solver = SolveGradientsLST()
        self.advection   = AdvectionMLS(self.grad_solver)
        self.extractor   = GraphConvFeatureExtractorV2(
            in_channels=2, hidden_channels=HIDDEN, out_channels=HIDDEN,
            num_layers=4, use_layer_norm=True, use_relative_pos=True,
        )
        self.film = SimulationConditionedLayerNorm(
            normalized_shape=HIDDEN, global_dim=3
        )
        self.mar = MappingAndRecon(
            n_base_features=HIDDEN, n_mask_channel=1,
            output_channel=1, heads=4, zero_init=True,
        )

    def init_mls(self, sample: Data):
        pos = sample.x[:, :2].to(next(self.parameters()).device)
        ei  = sample.edge_index.to(pos.device)
        self.grad_solver(Data(pos=pos, edge_index=ei),
                         torch.zeros(pos.shape[0], 1, device=pos.device))
        print("  MLS cache initialised", flush=True)

    def step(self, data: Data) -> torch.Tensor:
        pos  = data.x[:, :2]
        u    = data.x[:, 2:3]
        vel  = data.x[:, 3:5]
        dt   = data.global_delta_t.flatten()[0].item()
        gp   = data.global_params
        ei   = data.edge_index
        mesh = Data(pos=pos, edge_index=ei, mesh_id=data.mesh_id)

        adv   = self.advection(u, vel, mesh)
        feats = self.extractor(pos, ei, pos=pos)
        feats = self.film(feats, gp)
        du_dt = self.mar(feats, adv, ei)
        #return u + dt * du_dt
        return torch.clamp(u + dt * du_dt, 0.0, 1.0)

    def forward(self, sequence):
        preds = []
        u_cur = None
        for i, data in enumerate(sequence):
            if i > 0:
                data = Data(
                    x=torch.cat([data.x[:, :2], u_cur.detach(), data.x[:, 3:5]], dim=1),
                    y=data.y,
                    edge_index=data.edge_index,
                    global_params=data.global_params,
                    global_delta_t=data.global_delta_t,
                    mesh_id=data.mesh_id,
                )
            u_cur = self.step(data)
            preds.append(u_cur)
        return preds

def loss_fn(u_pred, data):
    u_next = data.y[:, 0:1]
    if PROBLEM == "gaussian":
        # Delta supervision: amplifies the tiny per-step signal ~100x
        u_cur = data.x[:, 2:3]
        return F.mse_loss(u_pred - u_cur, u_next - u_cur)
    else:
        # Direct supervision for disc: boundary nodes already have Δu=±1
        # delta supervision destabilises training
        return F.mse_loss(u_pred, u_next)
        
# TRAIN / VAL
def train_one_epoch(model, loader, optimiser, device):
    model.train()
    total, n = 0.0, 0
    for seq in loader:
        seq = [d.to(device) for d in seq]
        optimiser.zero_grad(set_to_none=True)
        preds = model(seq)
        loss  = sum(loss_fn(p, d) for p, d in zip(preds, seq)) / len(preds)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for seq in loader:
        seq  = [d.to(device) for d in seq]
        preds = model(seq)
        loss  = sum(loss_fn(p, d) for p, d in zip(preds, seq)) / len(preds)
        if torch.isfinite(loss):
            total += loss.item()
            n += 1
    return total / max(n, 1)

# MAIN
if __name__ == "__main__":
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)

    print(f"Problem : {PROBLEM}  |  Device : {device}", flush=True)

    train_loader = DataLoader(AdvectionDataset(TRAIN_DIR), batch_size=None)
    val_loader   = DataLoader(AdvectionDataset(VAL_DIR),   batch_size=None)

    model = GPARCAdvection().to(device)

    print("Initialising MLS...", flush=True)
    first = next(iter(DataLoader(AdvectionDataset(TRAIN_DIR), batch_size=None)))
    model.init_mls(first[0].to(device))

    print(f"Params  : {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"Starting {EPOCHS} epochs...", flush=True)

    optimiser = AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimiser, T_max=EPOCHS, eta_min=LR * 0.01)
    best_val  = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimiser, device)
        val_loss   = validate(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch:4d}/{EPOCHS}  train={train_loss:.3e}  val={val_loss:.3e}", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{OUT_DIR}/best_model.pth")
            print(f"  -> saved best (val={best_val:.3e})", flush=True)

    torch.save(model.state_dict(), f"{OUT_DIR}/final_model.pth")
    print(f"\nDone. Best val: {best_val:.3e}  |  Saved to {OUT_DIR}/", flush=True)