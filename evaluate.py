

from __future__ import annotations
import sys
import time
import numpy as np
import torch
from pathlib import Path
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path.home() / "G-PARC"))

from differentiator.hop import SolveGradientsLST, AdvectionMLS
from differentiator.mappingandrecon import MappingAndRecon
from utilities.featureextractor import GraphConvFeatureExtractorV2
from utilities.embed import SimulationConditionedLayerNorm

from numerical.convergence import ConvergenceStudy, ConvergenceResult, l2_error, linf_error


# Constants — must match generate_data.py exactly

L       = 2.0 * np.pi
VX, VY  = 1.0, 0.5
T_END   = 2.0
N_SNAPS = 20
DELTA_T = T_END / (N_SNAPS - 1)   # timestep used during training
HIDDEN  = 64
CFL     = 0.4

RESOLUTIONS = [256, 128, 64, 32, 16]

GAUSS_CHECKPOINT = "output/gaussian/best_model.pth"
DISC_CHECKPOINT  = "output/gaussian/disc/best_model.pth"

# Fixed IC parameters for convergence study (centred, mid-range)
G_X0, G_Y0, G_SIGMA = np.pi, np.pi, 0.5
D_X0, D_Y0, D_RADIUS = np.pi, np.pi, 0.6


# Grid helpers  (ij indexing — x varies along axis 0, y along axis 1)


def make_grid(N: int):
    """Return (X, Y, h) for N×N cell-left grid on [0, L)²."""
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return X, Y, L / N


def make_edge_index(N: int) -> torch.Tensor:
    """4-connected periodic cardinal edge index for N×N grid, shape [2, 4N²]."""
    idx = np.arange(N * N).reshape(N, N)
    rows, cols = [], []
    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        rows.append(idx.flatten())
        cols.append(np.roll(idx, shift=(-di, -dj), axis=(0, 1)).flatten())
    return torch.tensor(
        np.stack([np.concatenate(rows), np.concatenate(cols)]), dtype=torch.long
    )



# Analytical solutions


def gaussian_exact(X, Y, t, x0, y0, sigma):
    xc = (x0 + VX * t) % L;  yc = (y0 + VY * t) % L
    dx = X - xc;  dx -= L * np.round(dx / L)
    dy = Y - yc;  dy -= L * np.round(dy / L)
    return np.exp(-(dx**2 + dy**2) / sigma**2).astype(np.float32)


def disc_exact(X, Y, t, x0, y0, radius):
    xc = (x0 + VX * t) % L;  yc = (y0 + VY * t) % L
    dx = X - xc;  dx -= L * np.round(dx / L)
    dy = Y - yc;  dy -= L * np.round(dy / L)
    return (dx**2 + dy**2 <= radius**2).astype(np.float32)


# Classical baselines  (ij convention: vx acts on axis=0, vy on axis=1)


def upwind_2d(u0, h, dt, n_steps):
    u = u0.copy()
    ax = VX * dt / h;  ay = VY * dt / h
    for _ in range(n_steps):
        u = u - ax * (u - np.roll(u, 1, axis=0)) - ay * (u - np.roll(u, 1, axis=1))
    return u


def lax_wendroff_2d(u0, h, dt, n_steps):
    u = u0.copy()
    nx = VX * dt / h;  ny = VY * dt / h
    for _ in range(n_steps):
        up = np.roll(u, -1, axis=0);  um = np.roll(u, 1, axis=0)
        u  = u - 0.5 * nx * (up - um) + 0.5 * nx**2 * (up - 2*u + um)
        up = np.roll(u, -1, axis=1);  um = np.roll(u, 1, axis=1)
        u  = u - 0.5 * ny * (up - um) + 0.5 * ny**2 * (up - 2*u + um)
    return u


# G-PARC model  

class GPARCAdvection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_solver = SolveGradientsLST()
        self.advection   = AdvectionMLS(self.grad_solver)
        self.extractor   = GraphConvFeatureExtractorV2(
            in_channels=2, hidden_channels=HIDDEN, out_channels=HIDDEN,
            num_layers=4, use_layer_norm=True, use_relative_pos=True,
        )
        self.film = SimulationConditionedLayerNorm(normalized_shape=HIDDEN, global_dim=3)
        self.mar  = MappingAndRecon(
            n_base_features=HIDDEN, n_mask_channel=1,
            output_channel=1, heads=4, zero_init=True,
        )

    def step(self, data: Data) -> torch.Tensor:
        pos   = data.x[:, :2]
        u     = data.x[:, 2:3]
        vel   = data.x[:, 3:5]
        dt    = data.global_delta_t.flatten()[0].item()
        mesh  = Data(pos=pos, edge_index=data.edge_index, mesh_id=data.mesh_id)
        adv   = self.advection(u, vel, mesh)
        feats = self.extractor(pos, data.edge_index, pos=pos)
        feats = self.film(feats, data.global_params)
        du_dt = self.mar(feats, adv, data.edge_index)
        return torch.clamp(u + dt * du_dt, 0.0, 1.0)


def load_model(checkpoint: str, device: torch.device) -> GPARCAdvection:
    model = GPARCAdvection().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    model.eval()
    return model


@torch.no_grad()
def gparc_rollout(model: GPARCAdvection, u0_np: np.ndarray, N: int,
                device: torch.device) -> np.ndarray:
    """Roll out G-PARC from u0_np [N,N] for N_SNAPS-1 steps. Returns field [N,N]."""
    X, Y, h = make_grid(N)
    pos      = torch.from_numpy(
        np.stack([X.flatten() / L, Y.flatten() / L], axis=1).astype(np.float32)
    ).to(device)
    ei       = make_edge_index(N).to(device)
    gp       = torch.tensor([VX, VY, DELTA_T], dtype=torch.float32, device=device)
    g_dt     = torch.tensor([DELTA_T], dtype=torch.float32, device=device)
    mesh_id  = torch.tensor(N, dtype=torch.long, device=device)
    vx_col   = torch.full((N * N, 1), VX, dtype=torch.float32, device=device)
    vy_col   = torch.full((N * N, 1), VY, dtype=torch.float32, device=device)

    u = torch.from_numpy(u0_np.flatten().astype(np.float32)).unsqueeze(1).to(device)

    for _ in range(N_SNAPS - 1):
        data = Data(
            x=torch.cat([pos, u, vx_col, vy_col], dim=1),
            edge_index=ei,
            global_params=gp,
            global_delta_t=g_dt,
            mesh_id=mesh_id,
        )
        u = model.step(data)

    return u.cpu().numpy().reshape(N, N)



# Convergence runners


def run_study(problem: str, ic_fn, exact_fn, resolutions: list[int],
            gparc_model=None, device=None) -> ConvergenceStudy:
    study = ConvergenceStudy(problem=problem)

    for N in resolutions:
        X, Y, h = make_grid(N)
        cell_vol = h * h
        u0     = ic_fn(X, Y, 0.0)
        u_ex   = exact_fn(X, Y, T_END)

        dt_cfl = CFL * h / max(abs(VX), abs(VY))
        n_st   = int(np.ceil(T_END / dt_cfl));  dt = T_END / n_st

        t0 = time.time()
        u_up = upwind_2d(u0, h, dt, n_st)
        study.add(ConvergenceResult("upwind", N, h,
                                    l2_error(u_up, u_ex, cell_vol),
                                    linf_error(u_up, u_ex),
                                    (time.time() - t0) * 1000))

        t0 = time.time()
        u_lw = lax_wendroff_2d(u0, h, dt, n_st)
        study.add(ConvergenceResult("lax_wendroff", N, h,
                                    l2_error(u_lw, u_ex, cell_vol),
                                    linf_error(u_lw, u_ex),
                                    (time.time() - t0) * 1000))

        if gparc_model is not None:
            t0 = time.time()
            u_gp = gparc_rollout(gparc_model, u0, N, device)
            study.add(ConvergenceResult("g_parc", N, h,
                                        l2_error(u_gp, u_ex, cell_vol),
                                        linf_error(u_gp, u_ex),
                                        (time.time() - t0) * 1000))

        print(f"  N={N:4d}  h={h:.4f}  done", flush=True)

    return study



# Main


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n", flush=True)
    out = Path("figures");  out.mkdir(exist_ok=True)

    # Gaussian 
    print("── 2D Gaussian (smooth) ─────")
    gauss_model = load_model(GAUSS_CHECKPOINT, device)
    print(f"  Loaded {GAUSS_CHECKPOINT}")

    gauss_study = run_study(
        problem  = "2D Gaussian (smooth)",
        ic_fn    = lambda X, Y, t: gaussian_exact(X, Y, t, G_X0, G_Y0, G_SIGMA),
        exact_fn = lambda X, Y, t: gaussian_exact(X, Y, t, G_X0, G_Y0, G_SIGMA),
        resolutions  = RESOLUTIONS,
        gparc_model  = gauss_model,
        device       = device,
    )
    gauss_study.print_table()
    gauss_study.to_csv(str(out / "results_2d.csv"))
    print(f"  Saved {out / 'results_2d.csv'}\n")

    # Disc (non-smooth)
    print("── 2D Disc (non-smooth) ────")
    disc_model = load_model(DISC_CHECKPOINT, device)
    print(f"  Loaded {DISC_CHECKPOINT}")

    disc_study = run_study(
        problem  = "2D disc (non-smooth)",
        ic_fn    = lambda X, Y, t: disc_exact(X, Y, t, D_X0, D_Y0, D_RADIUS),
        exact_fn = lambda X, Y, t: disc_exact(X, Y, t, D_X0, D_Y0, D_RADIUS),
        resolutions  = RESOLUTIONS,
        gparc_model  = disc_model,
        device       = device,
    )
    disc_study.print_table()
    disc_study.to_csv(str(out / "results_disc.csv"))
    print(f"  Saved {out / 'results_disc.csv'}\n")

    print("Done. Run:  cd numerical && python plot.py")


if __name__ == "__main__":
    main()
