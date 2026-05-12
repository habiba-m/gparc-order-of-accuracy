"""
PDE: u_t + v·∇u = 0 on [0, 2π)², periodic BCs.

Two ICs:
  gaussian  u₀ = exp(-‖r-r₀‖²/σ²)   solved with spectral RK4
  disc      u₀ = 1 inside circle      analytical only (no Gibbs ringing)

Each .pt file is a list of PyG Data objects (one per timestep transition):
  x             [N², 5]   [x_norm, y_norm, u, vx, vy]
  y             [N², 3]   [u_next, vx_next, vy_next]
  edge_index    [2, 4N²]  4-connected cardinal, periodic
  global_params [3]       [vx, vy, delta_t]
  mesh_id       int       N  (MLS cache key)
  u_exact       [N², 1]   analytical ground truth  (eval splits only)
"""

import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
import json

# CONFIG

DATA_DIR = Path("data")
L = 2.0 * np.pi  # domain length
VX, VY = 1.0, 0.5  # constant advection velocity
T_END = 2.0
N_SNAPS = 20  # snapshots per sim (including t=0)
CFL = 0.4  # for spectral solver internal stepping

TRAIN_N = 64  # training resolution
EVAL_NS = [256, 128, 64, 48, 32, 24, 16]

# Simulation counts
N_TRAIN_GAUSSIAN = 50
N_VAL_GAUSSIAN = 10
N_TEST_GAUSSIAN = 10
N_TRAIN_DISC = 30
N_VAL_DISC = 10
N_TEST_DISC = 10
N_EVAL = 10  # per resolution, per problem

SEED = 42
rng = np.random.default_rng(SEED)

# Gaussian IC parameter ranges
G_X0 = (np.pi / 2, 3 * np.pi / 2)
G_Y0 = (np.pi / 2, 3 * np.pi / 2)
G_SIGMA = (0.3, 0.8)

# Disc IC parameter ranges
D_X0 = (np.pi / 2, 3 * np.pi / 2)
D_Y0 = (np.pi / 2, 3 * np.pi / 2)
D_RADIUS = (0.4, 1.0)  # radius in physical units (fraction of domain)

# ANALYTICAL SOLUTIONS


def gaussian_exact(X, Y, t, x0, y0, sigma):
    """Advected Gaussian with periodic minimum-image distance."""
    xc = (x0 + VX * t) % L
    yc = (y0 + VY * t) % L
    dx = X - xc
    dx = dx - L * np.round(dx / L)
    dy = Y - yc
    dy = dy - L * np.round(dy / L)
    return np.exp(-(dx**2 + dy**2) / sigma**2).astype(np.float32)


def disc_exact(X, Y, t, x0, y0, radius):
    """
    Advected disc (circular top-hat) with periodic minimum-image distance.
    u = 1.0 inside circle of given radius, 0.0 outside.
    Uses analytical formula directly.
    """
    xc = (x0 + VX * t) % L
    yc = (y0 + VY * t) % L
    dx = X - xc
    dx = dx - L * np.round(dx / L)
    dy = Y - yc
    dy = dy - L * np.round(dy / L)
    return (dx**2 + dy**2 <= radius**2).astype(np.float32)


# SPECTRAL RK4 SOLVER  (Gaussian only)


def _wavenumbers(N):
    return np.fft.fftfreq(N, d=1.0 / N) * (2.0 * np.pi / L)


def _rhs(u_hat, kx, ky):
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return -1j * (VX * KX + VY * KY) * u_hat


def _rk4(u_hat, dt, kx, ky):
    k1 = _rhs(u_hat, kx, ky)
    k2 = _rhs(u_hat + 0.5 * dt * k1, kx, ky)
    k3 = _rhs(u_hat + 0.5 * dt * k2, kx, ky)
    k4 = _rhs(u_hat + dt * k3, kx, ky)
    return u_hat + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def spectral_solve(u0, N, save_times):
    """Spectral RK4 solver. Returns {t: field [N,N]}."""
    kx = _wavenumbers(N)
    ky = _wavenumbers(N)
    dt_c = CFL * (L / N) / (abs(VX) + abs(VY))
    uhat = np.fft.fft2(u0)
    t, idx, snaps = 0.0, 0, {}

    if np.isclose(save_times[idx], 0.0):
        snaps[0.0] = np.real(np.fft.ifft2(uhat)).copy()
        idx += 1

    while t < T_END - 1e-12:
        dt = min(dt_c, T_END - t)
        if idx < len(save_times) and (t + dt) > save_times[idx] + 1e-12:
            dt = save_times[idx] - t
        uhat = _rk4(uhat, dt, kx, ky)
        t += dt
        while idx < len(save_times) and np.isclose(t, save_times[idx], atol=1e-10):
            snaps[save_times[idx]] = np.real(np.fft.ifft2(uhat)).copy()
            idx += 1
    return snaps


# GRAPH CONSTRUCTION  (4-connected cardinal, periodic)
def make_edge_index(N):
    """
    4-connected periodic cardinal graph for N×N grid.
    Node (i,j) = i*N + j  (row-major).
    Returns [2, 4N²] torch.long.
    """
    idx = np.arange(N * N).reshape(N, N)
    rows, cols = [], []
    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        rows.append(idx.flatten())
        cols.append(np.roll(idx, shift=(-di, -dj), axis=(0, 1)).flatten())
    return torch.tensor(np.stack([np.concatenate(rows), np.concatenate(cols)]), dtype=torch.long)


# IC SAMPLING
def sample_gaussian_ics(n):
    x0 = rng.uniform(*G_X0, size=n)
    y0 = rng.uniform(*G_Y0, size=n)
    sigma = rng.uniform(*G_SIGMA, size=n)
    return [dict(type="gaussian", x0=float(x0[i]), y0=float(y0[i]), sigma=float(sigma[i])) for i in range(n)]


def sample_disc_ics(n):
    x0 = rng.uniform(*D_X0, size=n)
    y0 = rng.uniform(*D_Y0, size=n)
    radius = rng.uniform(*D_RADIUS, size=n)
    return [dict(type="disc", x0=float(x0[i]), y0=float(y0[i]), radius=float(radius[i])) for i in range(n)]


# BUILD ONE SIMULATION  →  list of Data objects
def build_sim(N, ic, include_exact=False):
    """
    ic: dict with keys 'type' ('gaussian' or 'disc') plus IC parameters.

    Returns list of N_SNAPS-1 Data objects (consecutive timestep pairs).
    If include_exact=True, attaches data.u_exact [N²,1] at target time.
    """
    x1d = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x1d, x1d, indexing="ij")  # [N, N]

    pos = torch.from_numpy(np.stack([(X / L).flatten(), (Y / L).flatten()], axis=1).astype(np.float32))  # [N², 2]
    edge_index = make_edge_index(N)

    t_snap = np.linspace(0.0, T_END, N_SNAPS)
    delta_t = float(t_snap[1] - t_snap[0])

    gp = torch.tensor([VX, VY, delta_t], dtype=torch.float32)
    g_pressure = torch.tensor([VX], dtype=torch.float32)
    g_density = torch.tensor([VY], dtype=torch.float32)
    g_delta_t = torch.tensor([delta_t], dtype=torch.float32)
    mesh_id = torch.tensor(N, dtype=torch.long)

    vx_nodes = torch.full((N * N, 1), VX, dtype=torch.float32)
    vy_nodes = torch.full((N * N, 1), VY, dtype=torch.float32)

    # Generate snapshots
    ic_type = ic["type"]

    if ic_type == "gaussian":
        # Use spectral solver
        u0 = gaussian_exact(X, Y, 0.0, ic["x0"], ic["y0"], ic["sigma"])
        snaps = spectral_solve(u0, N, t_snap)
        fields = [snaps[t].flatten().astype(np.float32) for t in t_snap]

    elif ic_type == "disc":
        # Use analytical formula directly
        fields = [disc_exact(X, Y, t, ic["x0"], ic["y0"], ic["radius"]).flatten() for t in t_snap]
    else:
        raise ValueError(f"Unknown IC type: {ic_type}")

    # Build Data objects
    data_list = []
    for ti in range(N_SNAPS - 1):
        u_cur = torch.from_numpy(fields[ti]).unsqueeze(1)  # [N², 1]
        u_next = torch.from_numpy(fields[ti + 1]).unsqueeze(1)

        x_feat = torch.cat([pos, u_cur, vx_nodes, vy_nodes], dim=1)  # [N², 5]
        y_feat = torch.cat([u_next, vx_nodes, vy_nodes], dim=1)  # [N², 3]

        d = Data(
            x=x_feat,
            y=y_feat,
            pos=pos,
            edge_index=edge_index,
            global_params=gp,
            global_pressure=g_pressure,
            global_density=g_density,
            global_delta_t=g_delta_t,
            mesh_id=mesh_id,
        )

        if include_exact:
            # Always use the analytical formula as the clean reference
            if ic_type == "gaussian":
                exact = gaussian_exact(X, Y, t_snap[ti + 1], ic["x0"], ic["y0"], ic["sigma"]).flatten()
            else:
                exact = disc_exact(X, Y, t_snap[ti + 1], ic["x0"], ic["y0"], ic["radius"]).flatten()
            d.u_exact = torch.from_numpy(exact).unsqueeze(1)  # [N², 1]

        data_list.append(d)

    return data_list  # 19 Data objects


# SAVE
def save_sim(path, data_list, meta=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_list, path)
    if meta:
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)


# GENERATION
def generate_split(problem, split, ics, N=TRAIN_N):
    """Generate train/val/test split for one problem type."""
    out_dir = DATA_DIR / problem / split
    print(f"\n[{problem}/{split}]  {len(ics)} sims  N={N}")
    for i, ic in enumerate(ics):
        sim = build_sim(N, ic, include_exact=False)
        meta = {**ic, "split": split, "N": N, "vx": VX, "vy": VY, "n_steps": len(sim)}
        save_sim(out_dir / f"sim_{i:04d}.pt", sim, meta)
        ic_str = f"x0={ic['x0']:.2f} y0={ic['y0']:.2f} " f"σ={ic['sigma']:.2f}" if ic["type"] == "gaussian" else f"x0={ic['x0']:.2f} y0={ic['y0']:.2f} r={ic['radius']:.2f}"
        print(f"  [{i+1:2d}/{len(ics)}]  sim_{i:04d}.pt  {ic_str}")


def generate_eval(problem, ics, eval_ns=EVAL_NS):
    """Generate eval set at multiple resolutions with analytical ground truth."""
    for N in eval_ns:
        out_dir = DATA_DIR / problem / "eval" / f"res_{N}"
        print(f"\n[{problem}/eval/res_{N}]  {len(ics)} sims  h={L/N:.4f}")
        for i, ic in enumerate(ics):
            sim = build_sim(N, ic, include_exact=True)
            meta = {**ic, "split": "eval", "N": N, "h": float(L / N), "vx": VX, "vy": VY, "n_steps": len(sim)}
            save_sim(out_dir / f"sim_{i:04d}.pt", sim, meta)
        print(f"  saved  mesh_id={N}  u_exact attached to every Data object")


# SANITY CHECK
def sanity_check():
    print("\n── sanity check ──")
    N = 32

    # Gaussian
    g_ic = dict(type="gaussian", x0=np.pi, y0=np.pi, sigma=0.5)
    g_sim = build_sim(N, g_ic, include_exact=True)
    d = g_sim[0]
    assert d.x.shape == (N * N, 5)
    assert d.y.shape == (N * N, 3)
    assert d.edge_index.shape == (2, 4 * N * N)
    err_g = (d.y[:, 0:1] - d.u_exact).pow(2).mean().sqrt().item()
    print(f"  Gaussian  shapes OK  solver vs exact L2: {err_g:.2e}")

    # Disc
    d_ic = dict(type="disc", x0=np.pi, y0=np.pi, radius=0.6)
    d_sim = build_sim(N, d_ic, include_exact=True)
    d2 = d_sim[0]
    assert d2.x.shape == (N * N, 5)
    assert d2.y.shape == (N * N, 3)
    # Disc uses analytical for both y and u_exact so error = 0
    err_d = (d2.y[:, 0:1] - d2.u_exact).pow(2).mean().sqrt().item()
    print(f"  Disc      shapes OK  y vs exact L2: {err_d:.2e}  (0 expected)")

    # Check disc is binary
    vals = d2.x[:, 2].unique()
    assert set(vals.tolist()).issubset({0.0, 1.0}), "Disc should be binary"
    print(f"  Disc u values: {vals.tolist()}  (binary ✓)")
    print(f"  global_params: {d.global_params.tolist()}")
    print(f"  mesh_id: {d.mesh_id.item()}")
    print("  ✓ all checks passed")


# ENTRY POINT
if __name__ == "__main__":
    print(f"Generating data: train N={TRAIN_N}, eval N={EVAL_NS}")
    sanity_check()

    g_all = sample_gaussian_ics(N_TRAIN_GAUSSIAN + N_VAL_GAUSSIAN + N_TEST_GAUSSIAN)
    g_train = g_all[:N_TRAIN_GAUSSIAN]
    g_val = g_all[N_TRAIN_GAUSSIAN : N_TRAIN_GAUSSIAN + N_VAL_GAUSSIAN]
    g_test = g_all[N_TRAIN_GAUSSIAN + N_VAL_GAUSSIAN :]
    g_eval = g_test[:N_EVAL]

    d_all = sample_disc_ics(N_TRAIN_DISC + N_VAL_DISC + N_TEST_DISC)
    d_train = d_all[:N_TRAIN_DISC]
    d_val = d_all[N_TRAIN_DISC : N_TRAIN_DISC + N_VAL_DISC]
    d_test = d_all[N_TRAIN_DISC + N_VAL_DISC :]
    d_eval = d_test[:N_EVAL]

    generate_split("gaussian", "train", g_train)
    generate_split("gaussian", "val", g_val)
    generate_split("gaussian", "test", g_test)
    generate_eval("gaussian", g_eval)

    generate_split("disc", "train", d_train)
    generate_split("disc", "val", d_val)
    generate_split("disc", "test", d_test)
    generate_eval("disc", d_eval)

    print("\nDone.")
    for problem in ["gaussian", "disc"]:
        for split in ["train", "val", "test"]:
            n = len(list((DATA_DIR / problem / split).glob("*.pt")))
            print(f"  {problem}/{split}: {n} files")
        for N in EVAL_NS:
            n = len(list((DATA_DIR / problem / "eval" / f"res_{N}").glob("*.pt")))
            print(f"  {problem}/eval/res_{N}: {n} files  (h={L/N:.4f})")
