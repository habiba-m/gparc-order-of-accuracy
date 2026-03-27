import math
from pathlib import Path
from typing import List, Tuple

import torch
from torch_geometric.data import Data


DTYPE = torch.float32
BASE_DIR = Path("advection_data")

# time settings
DT = 0.01
T_FINAL = 1.0

# 2D Gaussian settings
Lx, Ly = 2.0 * math.pi, 2.0 * math.pi
GAUSS_CENTER = (1.5, 2.0)
GAUSS_SIGMA = 0.4

TRAIN_GAUSS_MESHES = [(64, 64), (80, 80)]
VAL_GAUSS_MESHES = [(64, 64)]
TEST_GAUSS_MESHES = [(64, 64), (48, 48), (32, 32), (24, 24), (16, 16)]

TRAIN_GAUSS_VELS = [
    (0.8, 0.2),
    (0.5, 0.7),
    (-0.6, 0.4),
    (0.9, -0.3),
]
VAL_GAUSS_VELS = [
    (0.6, 0.6),
]
TEST_GAUSS_VELS = [
    (0.8, 0.2),
    (0.6, 0.6),
    (-0.5, 0.8),
]

# 1D Tophat settings
L1D = 2.0 * math.pi
TOPHAT_CENTER = 1.0
TOPHAT_WIDTH = 1.0

TRAIN_TOPHAT_N = [256]
VAL_TOPHAT_N = [256]
TEST_TOPHAT_N = [256, 128, 64, 32]

TRAIN_TOPHAT_VELS = [0.3, 0.5, 0.7, 0.9, -0.3, -0.5, -0.7, -0.9]
VAL_TOPHAT_VELS = [0.6]
TEST_TOPHAT_VELS = [0.7, -0.8, 1.0]


# HELPERS
def periodic_distance(x: torch.Tensor, x0: float, L: float) -> torch.Tensor:
    """Smallest periodic signed distance on [0, L)."""
    return (x - x0 + 0.5 * L) % L - 0.5 * L


def make_times(dt: float, t_final: float) -> torch.Tensor:
    return torch.arange(0.0, t_final + 1e-12, dt, dtype=DTYPE)


# GRAPH BUILDERS
def build_2d_periodic_grid(nx: int, ny: int, lx: float, ly: float):
    """
    Returns:
        pos: [N, 2]
        edge_index: [2, E]
        h: representative mesh spacing
    """
    xs = torch.linspace(0.0, lx, nx + 1, dtype=DTYPE)[:-1]
    ys = torch.linspace(0.0, ly, ny + 1, dtype=DTYPE)[:-1]

    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    pos = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    def idx(i: int, j: int) -> int:
        return i * ny + j

    edges = []
    for i in range(nx):
        for j in range(ny):
            a = idx(i, j)
            b = idx((i + 1) % nx, j)
            c = idx(i, (j + 1) % ny)

            # undirected 4-neighbor periodic graph
            edges.append((a, b))
            edges.append((b, a))
            edges.append((a, c))
            edges.append((c, a))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    h = max(lx / nx, ly / ny)
    return pos, edge_index, h


def build_1d_periodic_line(n: int, lx: float):
    """
    1D periodic line embedded in 2D as [x, 0].
    This lets you reuse the same static feature pattern [x, y, vx, vy].
    """
    xs = torch.linspace(0.0, lx, n + 1, dtype=DTYPE)[:-1]
    ys = torch.zeros_like(xs)
    pos = torch.stack([xs, ys], dim=1)

    edges = []
    for i in range(n):
        j = (i + 1) % n
        edges.append((i, j))
        edges.append((j, i))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    h = lx / n
    return pos, edge_index, h


# EXACT SOLUTIONS
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
    Exact periodic advection of a Gaussian bump.
    Returns [N, 1]
    """
    x = pos[:, 0]
    y = pos[:, 1]

    xc = (x0 + vx * t) % lx
    yc = (y0 + vy * t) % ly

    dx = periodic_distance(x, xc, lx)
    dy = periodic_distance(y, yc, ly)

    u = torch.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))
    return u.unsqueeze(1)


def tophat_solution_1d(
    pos: torch.Tensor,
    t: float,
    vx: float,
    lx: float,
    x0: float,
    width: float,
) -> torch.Tensor:
    """
    Exact periodic advection of a non-smooth tophat.
    Returns [N, 1]
    """
    x = pos[:, 0]
    xc = (x0 + vx * t) % lx
    dx = periodic_distance(x, xc, lx)
    u = (dx.abs() <= 0.5 * width).to(DTYPE)
    return u.unsqueeze(1)


# SNAPSHOT PACKAGING
def make_data_object(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    static_feats: torch.Tensor,
    u_t: torch.Tensor,
    u_tp1: torch.Tensor,
    h: float,
    case_name: str,
    mesh_id: int,
) -> Data:
    """
    G-PARCv2-style snapshot:
      x = [static_feats, dynamic_state] = [N, S+1]
      y = next-step dynamic state        = [N, 1]
    """
    data = Data(
        x=torch.cat([static_feats, u_t], dim=1),  # [N, S+1]
        y=u_tp1,  # [N, 1]
        pos=pos,
        edge_index=edge_index,
    )
    data.mesh_id = torch.tensor([mesh_id], dtype=torch.long)
    data.h = torch.tensor([h], dtype=DTYPE)
    data.case = case_name
    return data


# SIMULATION GENERATORS
def generate_gaussian_simulation(
    nx: int,
    ny: int,
    vx: float,
    vy: float,
    mesh_id: int,
    dt: float = DT,
    t_final: float = T_FINAL,
) -> List[Data]:
    """
    Static features: [x, y, vx, vy]   -> 4 channels
    Dynamic feature: [u]              -> 1 channel
    """
    pos, edge_index, h = build_2d_periodic_grid(nx, ny, Lx, Ly)
    times = make_times(dt, t_final)

    static_feats = torch.cat(
        [
            pos,
            torch.full((pos.shape[0], 1), vx, dtype=DTYPE),
            torch.full((pos.shape[0], 1), vy, dtype=DTYPE),
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
            mesh_id=mesh_id,
        )
        data.nx = torch.tensor([nx], dtype=torch.long)
        data.ny = torch.tensor([ny], dtype=torch.long)
        sim.append(data)

    return sim


def generate_tophat_simulation(
    n: int,
    vx: float,
    mesh_id: int,
    dt: float = DT,
    t_final: float = T_FINAL,
) -> List[Data]:
    """
    Still uses 4 static channels for compatibility:
      [x, y, vx, vy] = [x, 0, vx, 0]
    Dynamic feature:
      [u]
    """
    pos, edge_index, h = build_1d_periodic_line(n, L1D)
    times = make_times(dt, t_final)

    static_feats = torch.cat(
        [
            pos[:, 0:1],  # x
            pos[:, 1:2],  # y = 0
            torch.full((pos.shape[0], 1), vx, dtype=DTYPE),
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
            mesh_id=mesh_id,
        )
        data.n = torch.tensor([n], dtype=torch.long)
        sim.append(data)

    return sim


# WRITERS
def save_simulation(sim: List[Data], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sim, out_path)


def write_gaussian_split(split_dir: Path, meshes, velocities, start_mesh_id: int = 0):
    mesh_id = start_mesh_id
    sim_id = 0
    for nx, ny in meshes:
        for vx, vy in velocities:
            sim = generate_gaussian_simulation(nx, ny, vx, vy, mesh_id=mesh_id)
            out_path = split_dir / f"gaussian2d_sim_{sim_id:04d}.pt"
            save_simulation(sim, out_path)
            print(f"saved {out_path}")
            sim_id += 1
            mesh_id += 1


def write_tophat_split(split_dir: Path, resolutions, velocities, start_mesh_id: int = 100000):
    mesh_id = start_mesh_id
    sim_id = 0
    for n in resolutions:
        for vx in velocities:
            sim = generate_tophat_simulation(n, vx, mesh_id=mesh_id)
            out_path = split_dir / f"tophat1d_sim_{sim_id:04d}.pt"
            save_simulation(sim, out_path)
            print(f"saved {out_path}")
            sim_id += 1
            mesh_id += 1


def build_all_datasets(base_dir: Path = BASE_DIR):
    # 2D Gaussian dataset
    write_gaussian_split(base_dir / "gaussian2d" / "train", TRAIN_GAUSS_MESHES, TRAIN_GAUSS_VELS, start_mesh_id=0)
    write_gaussian_split(base_dir / "gaussian2d" / "val", VAL_GAUSS_MESHES, VAL_GAUSS_VELS, start_mesh_id=1000)
    write_gaussian_split(base_dir / "gaussian2d" / "test", TEST_GAUSS_MESHES, TEST_GAUSS_VELS, start_mesh_id=2000)

    # 1D Tophat dataset
    write_tophat_split(base_dir / "tophat1d" / "train", TRAIN_TOPHAT_N, TRAIN_TOPHAT_VELS, start_mesh_id=3000)
    write_tophat_split(base_dir / "tophat1d" / "val", VAL_TOPHAT_N, VAL_TOPHAT_VELS, start_mesh_id=4000)
    write_tophat_split(base_dir / "tophat1d" / "test", TEST_TOPHAT_N, TEST_TOPHAT_VELS, start_mesh_id=5000)

    print(f"\nDone. Wrote dataset under: {base_dir.resolve()}")


# INSPECTION
def inspect_pt_file(path: str):
    sim = torch.load(path, weights_only=False)
    print(f"timesteps: {len(sim)}")
    d0 = sim[0]

    # Gaussian: should see min near 0 and max near 1
    # Tophat: should see exactly 0 and 1
    print("u(t) min/max:", d0.x[:, -1].min().item(), d0.x[:, -1].max().item())

    # Gaussian: should look almost identical (just slightly shifted)
    # Tophat: still binary (0 or 1)
    print("u(t+1) min/max:", d0.y[:, 0].min().item(), d0.y[:, 0].max().item())

    print("x shape:", d0.x.shape)  # [N, 5] = 4 static + 1 dynamic
    print("y shape:", d0.y.shape)  # [N, 1]
    print("pos shape:", d0.pos.shape)  # [N, 2]
    print("edge_index shape:", d0.edge_index.shape)
    print("case:", getattr(d0, "case", None))
    print("h:", getattr(d0, "h", None))
    print("first 5 x rows:\n", d0.x[:5])
    print("first 5 y rows:\n", d0.y[:5])


if __name__ == "__main__":
    build_all_datasets()

    # checks:
    inspect_pt_file("advection_data/gaussian2d/train/gaussian2d_sim_0000.pt")
    inspect_pt_file("advection_data/tophat1d/train/tophat1d_sim_0000.pt")
