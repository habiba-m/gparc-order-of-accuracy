"""
data_gen.py  —  G-PARC Advection Dataset Generator
====================================================
Generates training, validation, and test data for two advection problems:

  1. 2D Gaussian  :  smooth initial condition on a periodic 2D grid
  2. 1D Tophat    :  non-smooth (discontinuous) IC on a periodic 1D line
                     embedded in 2D so the same model architecture applies

PDE solved (analytically, no numerical error):
    u_t + v · ∇u = 0     (linear advection, constant velocity)

Data format per timestep  (compatible with G-PARC / train_and_validate_weighted):
    data.x          [N, 5]   static + dynamic  =  [x, y, vx, vy,  u_t]
    data.y          [N, 1]   target            =  u_{t+1}
    data.pos        [N, 2]   spatial coords    =  (x, y)
    data.edge_index [2, E]   graph connectivity (4-neighbor periodic)
    data.mesh_id    scalar   geometry-based unique ID  (used by MLS cache)
    data.h          scalar   representative mesh spacing

mesh_id design
--------------
The MLS operators (SolveGradientsLST, SolveWeightLST2d) cache computed
weights by mesh_id.  Because MLS weights depend ONLY on node positions and
graph topology, every simulation on the SAME (nx, ny) grid should share the
same mesh_id so weights are computed once and reused.

  2D grid :  mesh_id = nx * 10_000 + ny          (e.g. 64×64 → 640064)
  1D line :  mesh_id = 10_000_000 + n            (e.g. n=256 → 10000256)

These ranges do not overlap and are stable across train / val / test splits.
"""

import math
from pathlib import Path
from typing import List, Tuple

import torch
from torch_geometric.data import Data


# Global constants

DTYPE = torch.float32

BASE_DIR = Path("advection_data")

# Time stepping
DT = 0.01
T_FINAL = 1.0  # 100 timestep transitions per simulation

# 2D domain
Lx, Ly = 2.0 * math.pi, 2.0 * math.pi

# Gaussian IC parameters
GAUSS_CENTER = (1.5, 2.0)
GAUSS_SIGMA = 0.4  # wide enough that the bump is well-resolved even at 16×16

# 1D domain
L1D = 2.0 * math.pi

# Tophat IC parameters
TOPHAT_CENTER = 1.0
TOPHAT_WIDTH = 1.0  # compact support, creates genuine discontinuity

# Split configurations

# --- 2D Gaussian ---
# Training meshes: two resolutions to expose the model to different h values
TRAIN_GAUSS_MESHES = [(64, 64), (80, 80)]

# Training velocities: diverse angles and speeds to prevent velocity memorisation
TRAIN_GAUSS_VELS = [
    (0.8, 0.2),
    (0.5, 0.7),
    (-0.6, 0.4),
    (0.9, -0.3),
]

VAL_GAUSS_MESHES = [(64, 64)]
VAL_GAUSS_VELS = [(0.6, 0.6)]

# Test meshes span ~4× in h to get a clean log-log convergence slope.
# (64,64) is deliberately in training so we can verify in-distribution behaviour.
# (48,48), (32,32), (24,24), (16,16) are progressively coarser unseen resolutions.
TEST_GAUSS_MESHES = [(64, 64), (48, 48), (32, 32), (24, 24), (16, 16)]
TEST_GAUSS_VELS = [(0.8, 0.2), (0.6, 0.6), (-0.5, 0.8)]

# --- 1D Tophat ---
TRAIN_TOPHAT_N = [256]
TRAIN_TOPHAT_VELS = [0.3, 0.5, 0.7, 0.9, -0.3, -0.5, -0.7, -0.9]

VAL_TOPHAT_N = [256]
VAL_TOPHAT_VELS = [0.6]

# Coarser test grids expose whether convergence degrades near discontinuities
TEST_TOPHAT_N = [256, 128, 64, 32]
TEST_TOPHAT_VELS = [0.7, -0.8, 1.0]


# Helpers


def periodic_distance(x: torch.Tensor, x0: float, L: float) -> torch.Tensor:
    """
    Signed distance from x to x0 on the periodic domain [0, L).
    Returns the shortest-path signed displacement in [-L/2, L/2).
    """
    return (x - x0 + 0.5 * L) % L - 0.5 * L


def make_times(dt: float, t_final: float) -> torch.Tensor:
    """Uniform time grid from 0 to t_final (inclusive)."""
    return torch.arange(0.0, t_final + 1e-12, dt, dtype=DTYPE)


def mesh_id_2d(nx: int, ny: int) -> int:
    """
    Deterministic mesh_id for a 2D (nx × ny) periodic grid.
    Unique for any (nx, ny) with nx, ny < 10_000.
    """
    return nx * 10_000 + ny


def mesh_id_1d(n: int) -> int:
    """
    Deterministic mesh_id for a 1D periodic line of n nodes.
    Offset by 10_000_000 to avoid overlap with 2D ids.
    """
    return 10_000_000 + n


# Graph builders


def build_2d_periodic_grid(nx: int, ny: int, lx: float, ly: float) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Build an (nx × ny) uniform periodic Cartesian grid.

    Connectivity: 4-neighbor (north / south / east / west) with periodic
    wrap-around.  Undirected — every edge appears in both directions.

    Returns
    -------
    pos        [N, 2]   node positions, N = nx * ny
    edge_index [2, E]   undirected 4-neighbor edges
    h          float    max(lx/nx, ly/ny) representative mesh spacing
    """
    xs = torch.linspace(0.0, lx, nx + 1, dtype=DTYPE)[:-1]
    ys = torch.linspace(0.0, ly, ny + 1, dtype=DTYPE)[:-1]

    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    pos = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # [N, 2]

    def node(i: int, j: int) -> int:
        return i * ny + j

    edges = []
    for i in range(nx):
        for j in range(ny):
            a = node(i, j)
            b = node((i + 1) % nx, j)  # east neighbour (periodic)
            c = node(i, (j + 1) % ny)  # north neighbour (periodic)
            edges += [(a, b), (b, a), (a, c), (c, a)]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]
    h = max(lx / nx, ly / ny)
    return pos, edge_index, h


def build_1d_periodic_line(n: int, lx: float) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Build a 1D periodic line of n nodes embedded in 2D (y = 0 for all nodes).

    The y = 0 embedding lets the 1D problem reuse the same 4-channel static
    feature layout [x, y=0, vx, vy=0] as the 2D case, so a single model
    architecture handles both problems.

    Connectivity: 2-neighbor (left / right) with periodic wrap-around.

    Returns
    -------
    pos        [N, 2]   positions with y-column all zeros
    edge_index [2, E]
    h          float    lx / n
    """
    xs = torch.linspace(0.0, lx, n + 1, dtype=DTYPE)[:-1]
    pos = torch.stack([xs, torch.zeros_like(xs)], dim=1)  # [N, 2]

    edges = []
    for i in range(n):
        j = (i + 1) % n
        edges += [(i, j), (j, i)]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    h = lx / n
    return pos, edge_index, h


# Exact solutions


def gaussian_solution_2d(
    pos: torch.Tensor,
    t: float,
    vx: float,
    vy: float,
    lx: float,
    ly: float,
    x0: float,
    y0: float,
    sigma: float,
) -> torch.Tensor:
    """
    Exact solution of periodic 2D linear advection with Gaussian IC.

    The bump centre advects as  (x0 + vx·t) mod lx,  (y0 + vy·t) mod ly,
    preserving the Gaussian shape exactly (no diffusion).

    Returns
    -------
    u  [N, 1]  field values at time t
    """
    x = pos[:, 0]
    y = pos[:, 1]

    xc = (x0 + vx * t) % lx
    yc = (y0 + vy * t) % ly

    dx = periodic_distance(x, xc, lx)
    dy = periodic_distance(y, yc, ly)

    u = torch.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))
    return u.unsqueeze(1)  # [N, 1]


def tophat_solution_1d(
    pos: torch.Tensor,
    t: float,
    vx: float,
    lx: float,
    x0: float,
    width: float,
) -> torch.Tensor:
    """
    Exact solution of periodic 1D linear advection with tophat IC.

    The tophat is 1 inside [xc - width/2, xc + width/2] (periodically),
    0 outside.  The sharp edges create genuine discontinuities that probe
    whether the model introduces implicit numerical dissipation.

    Returns
    -------
    u  [N, 1]  binary field values at time t
    """
    x = pos[:, 0]
    xc = (x0 + vx * t) % lx
    dx = periodic_distance(x, xc, lx)
    u = (dx.abs() <= 0.5 * width).to(DTYPE)
    return u.unsqueeze(1)  # [N, 1]


# Snapshot packager


def make_data_object(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    static_feats: torch.Tensor,
    u_t: torch.Tensor,
    u_tp1: torch.Tensor,
    h: float,
    case_name: str,
    mesh_id_val: int,
) -> Data:
    """
    Package one (u_t → u_{t+1}) transition into a PyG Data object.

    Feature layout
    --------------
    data.x  =  [static_feats | u_t]   shape [N, S+1]
               static_feats = [x, y, vx, vy]   (S = 4)
               This is the layout the derivative solver and integrators expect:
                   columns 0-3  →  static  (positions + velocity)
                   column  4    →  dynamic (scalar field u)

    data.y  =  u_{t+1}                shape [N, 1]
               Target for loss computation.

    Parameters
    ----------
    mesh_id_val : int
        Geometry-based unique ID.  All snapshots from the same physical mesh
        (same nx, ny or same n) share this ID so MLS weights are cached once.
    """
    data = Data(
        x=torch.cat([static_feats, u_t], dim=1),  # [N, 5]
        y=u_tp1,  # [N, 1]
        pos=pos,  # [N, 2]
        edge_index=edge_index,
    )
    data.mesh_id = torch.tensor([mesh_id_val], dtype=torch.long)
    data.h = torch.tensor([h], dtype=DTYPE)
    data.case = case_name
    return data


# Simulation generators


def generate_gaussian_simulation(
    nx: int,
    ny: int,
    vx: float,
    vy: float,
    dt: float = DT,
    t_final: float = T_FINAL,
) -> List[Data]:
    """
    Generate all (u_t, u_{t+1}) transitions for a 2D Gaussian simulation.

    Static features per node:  [x, y, vx, vy]   (4 channels)
    Dynamic feature per node:  [u]               (1 channel)

    The velocity (vx, vy) is constant across all nodes and timesteps —
    it is a simulation parameter, not something that evolves.  Storing it
    as a static feature lets the model condition on it without any special
    treatment.

    Parameters
    ----------
    nx, ny : int    Grid resolution along x and y.
    vx, vy : float  Constant advection velocity.

    Returns
    -------
    sim : List[Data]  One Data object per timestep transition (len = T/dt).
    """
    pos, edge_index, h = build_2d_periodic_grid(nx, ny, Lx, Ly)
    times = make_times(dt, t_final)

    mid = mesh_id_2d(nx, ny)

    static_feats = torch.cat(
        [
            pos,  # x, y
            torch.full((pos.shape[0], 1), vx, dtype=DTYPE),  # vx
            torch.full((pos.shape[0], 1), vy, dtype=DTYPE),  # vy
        ],
        dim=1,
    )  # [N, 4]

    sim = []
    for k in range(len(times) - 1):
        t = float(times[k].item())
        tp1 = float(times[k + 1].item())

        u_t = gaussian_solution_2d(pos, t, vx, vy, Lx, Ly, GAUSS_CENTER[0], GAUSS_CENTER[1], GAUSS_SIGMA)
        u_tp1 = gaussian_solution_2d(pos, tp1, vx, vy, Lx, Ly, GAUSS_CENTER[0], GAUSS_CENTER[1], GAUSS_SIGMA)

        data = make_data_object(
            pos=pos,
            edge_index=edge_index,
            static_feats=static_feats,
            u_t=u_t,
            u_tp1=u_tp1,
            h=h,
            case_name="gaussian2d",
            mesh_id_val=mid,
        )
        # Store resolution for inspection / filtering downstream
        data.nx = torch.tensor([nx], dtype=torch.long)
        data.ny = torch.tensor([ny], dtype=torch.long)
        sim.append(data)

    return sim


def generate_tophat_simulation(
    n: int,
    vx: float,
    dt: float = DT,
    t_final: float = T_FINAL,
) -> List[Data]:
    """
    Generate all (u_t, u_{t+1}) transitions for a 1D Tophat simulation.

    Uses a 1D line embedded in 2D (y = 0, vy = 0) so the static feature
    layout is identical to the 2D Gaussian case: [x, y=0, vx, vy=0].

    Parameters
    ----------
    n   : int   Number of grid points.
    vx  : float Advection speed (signed).

    Returns
    -------
    sim : List[Data]
    """
    pos, edge_index, h = build_1d_periodic_line(n, L1D)
    times = make_times(dt, t_final)

    mid = mesh_id_1d(n)

    static_feats = torch.cat(
        [
            pos[:, 0:1],  # x
            pos[:, 1:2],  # y = 0
            torch.full((pos.shape[0], 1), vx, dtype=DTYPE),  # vx
            torch.zeros((pos.shape[0], 1), dtype=DTYPE),  # vy = 0
        ],
        dim=1,
    )  # [N, 4]

    sim = []
    for k in range(len(times) - 1):
        t = float(times[k].item())
        tp1 = float(times[k + 1].item())

        u_t = tophat_solution_1d(pos, t, vx, L1D, TOPHAT_CENTER, TOPHAT_WIDTH)
        u_tp1 = tophat_solution_1d(pos, tp1, vx, L1D, TOPHAT_CENTER, TOPHAT_WIDTH)

        data = make_data_object(
            pos=pos,
            edge_index=edge_index,
            static_feats=static_feats,
            u_t=u_t,
            u_tp1=u_tp1,
            h=h,
            case_name="tophat1d",
            mesh_id_val=mid,
        )
        data.n = torch.tensor([n], dtype=torch.long)
        sim.append(data)

    return sim


# Writers


def save_simulation(sim: List[Data], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sim, out_path)


def write_gaussian_split(
    split_dir: Path,
    meshes: List[Tuple[int, int]],
    velocities: List[Tuple[float, float]],
):
    """Write all (mesh, velocity) combinations for one split to disk."""
    sim_id = 0
    for nx, ny in meshes:
        for vx, vy in velocities:
            sim = generate_gaussian_simulation(nx, ny, vx, vy)
            out_path = split_dir / f"gaussian2d_sim_{sim_id:04d}.pt"
            save_simulation(sim, out_path)
            mid = mesh_id_2d(nx, ny)
            print(f"  saved {out_path}  " f"[mesh={nx}×{ny}, v=({vx},{vy}), mesh_id={mid}, " f"steps={len(sim)}]")
            sim_id += 1


def write_tophat_split(
    split_dir: Path,
    resolutions: List[int],
    velocities: List[float],
):
    """Write all (resolution, velocity) combinations for one split to disk."""
    sim_id = 0
    for n in resolutions:
        for vx in velocities:
            sim = generate_tophat_simulation(n, vx)
            out_path = split_dir / f"tophat1d_sim_{sim_id:04d}.pt"
            save_simulation(sim, out_path)
            mid = mesh_id_1d(n)
            print(f"  saved {out_path}  " f"[n={n}, vx={vx}, mesh_id={mid}, steps={len(sim)}]")
            sim_id += 1


# Top-level build


def build_all_datasets(base_dir: Path = BASE_DIR):
    """
    Generate all training / validation / test splits for both cases.

    Directory layout produced:
        advection_data/
          gaussian2d/
            train/   gaussian2d_sim_0000.pt  ...
            val/     gaussian2d_sim_0000.pt  ...
            test/    gaussian2d_sim_0000.pt  ...  (multiple resolutions)
          tophat1d/
            train/   tophat1d_sim_0000.pt  ...
            val/     ...
            test/    ...

    Each .pt file is a List[Data] — one per timestep transition.
    """
    print("=" * 60)
    print("Generating 2D Gaussian datasets")
    print("=" * 60)

    print("\n[gaussian2d / train]")
    write_gaussian_split(base_dir / "gaussian2d" / "train", TRAIN_GAUSS_MESHES, TRAIN_GAUSS_VELS)

    print("\n[gaussian2d / val]")
    write_gaussian_split(base_dir / "gaussian2d" / "val", VAL_GAUSS_MESHES, VAL_GAUSS_VELS)

    print("\n[gaussian2d / test]  — multiple resolutions for convergence study")
    write_gaussian_split(base_dir / "gaussian2d" / "test", TEST_GAUSS_MESHES, TEST_GAUSS_VELS)

    print("\n" + "=" * 60)
    print("Generating 1D Tophat datasets")
    print("=" * 60)

    print("\n[tophat1d / train]")
    write_tophat_split(base_dir / "tophat1d" / "train", TRAIN_TOPHAT_N, TRAIN_TOPHAT_VELS)

    print("\n[tophat1d / val]")
    write_tophat_split(base_dir / "tophat1d" / "val", VAL_TOPHAT_N, VAL_TOPHAT_VELS)

    print("\n[tophat1d / test]  — multiple resolutions for convergence study")
    write_tophat_split(base_dir / "tophat1d" / "test", TEST_TOPHAT_N, TEST_TOPHAT_VELS)

    print(f"\nDone.  Dataset root: {base_dir.resolve()}")


# Inspection utilities


def inspect_pt_file(path: str):
    """
    Quick sanity check for a saved simulation file.

    Expected output:
        Gaussian  →  u min ~ 0, u max ~ 1, smooth bump visible in first 5 rows
        Tophat    →  u values exactly 0 or 1 (binary)
    """
    sim = torch.load(path, weights_only=False)
    d0 = sim[0]

    print(f"\n{'='*50}")
    print(f"File  : {path}")
    print(f"Steps : {len(sim)}")
    print(f"Case  : {getattr(d0, 'case', 'unknown')}")
    print(f"h     : {d0.h.item():.5f}")
    print(f"mesh_id: {d0.mesh_id.item()}")
    print(f"x  shape : {d0.x.shape}    (expected [N, 5])")
    print(f"y  shape : {d0.y.shape}    (expected [N, 1])")
    print(f"pos shape: {d0.pos.shape}  (expected [N, 2])")
    print(f"edge_index: {d0.edge_index.shape}")
    print(f"u(t)   min/max : {d0.x[:, -1].min():.4f} / {d0.x[:, -1].max():.4f}")
    print(f"u(t+1) min/max : {d0.y[:, 0].min():.4f} / {d0.y[:, 0].max():.4f}")
    print(f"First 5 rows of x:\n{d0.x[:5]}")
    print(f"First 5 rows of y:\n{d0.y[:5]}")


def get_sample_data(case: str = "gaussian2d", base_dir: Path = BASE_DIR) -> Data:
    """
    Load the very first snapshot from the first training simulation.

    Used by GPARC_Advection to initialise MLS weights before training:
        sample = get_sample_data()
        model.derivative_solver.initialize_weights(sample)
    """
    if case == "gaussian2d":
        path = base_dir / "gaussian2d" / "train" / "gaussian2d_sim_0000.pt"
    elif case == "tophat1d":
        path = base_dir / "tophat1d" / "train" / "tophat1d_sim_0000.pt"
    else:
        raise ValueError(f"Unknown case: {case}")

    if not path.exists():
        raise FileNotFoundError(f"Training data not found at {path}.\n" f"Run build_all_datasets() first.")

    sim = torch.load(path, weights_only=False)
    return sim[0]


# Entry point

if __name__ == "__main__":
    build_all_datasets()

    print("\n--- Sanity checks ---")
    inspect_pt_file("advection_data/gaussian2d/train/gaussian2d_sim_0000.pt")
    inspect_pt_file("advection_data/tophat1d/train/tophat1d_sim_0000.pt")
