"""
Reference solutions for the linear advection equation.

Provides:
- Analytical solutions (the exact translated initial condition)
- Classical numerical baselines: first-order upwind and Lax-Wendroff

These are used both as ground truth for the convergence study and as
baselines in the comparison table.

The advection equation is:
    2D:  u_t + v . grad(u) = 0
    1D:  u_t + c u_x = 0
The exact solution is pure translation of the IC: u(x, t) = u0(x - v*t).
"""

from __future__ import annotations
import numpy as np

# Initial conditions

def gaussian_2d(x: np.ndarray, y: np.ndarray,
                x0: float = 0.0, y0: float = 0.0,
                sigma: float = 0.1) -> np.ndarray:
    """2D Gaussian centered at (x0, y0) with width sigma."""
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (sigma ** 2))


def tophat_1d(x: np.ndarray, a: float = -0.25, b: float = 0.25) -> np.ndarray:
    """1D tophat (indicator) on [a, b]."""
    return ((x >= a) & (x <= b)).astype(np.float64)



# Analytical solutions -- ground truth

def analytical_2d_gaussian(x: np.ndarray, y: np.ndarray, t: float,
                        v: tuple[float, float] = (1.0, 0.0),
                        x0: float = 0.0, y0: float = 0.0,
                        sigma: float = 0.1,
                        domain: tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
    """
    Analytical solution at time t for the 2D advected Gaussian on a periodic
    square domain [-L, L] x [-L, L] (default L=1).
    """
    L = domain[1] - domain[0]
    # periodic shift of center
    xc = ((x0 + v[0] * t) - domain[0]) % L + domain[0]
    yc = ((y0 + v[1] * t) - domain[0]) % L + domain[0]
    # account for periodic wrap by taking minimum image distance
    dx = (x - xc + L / 2) % L - L / 2
    dy = (y - yc + L / 2) % L - L / 2
    return np.exp(-(dx ** 2 + dy ** 2) / sigma ** 2)


def analytical_1d_tophat(x: np.ndarray, t: float,
                        c: float = 1.0,
                        a: float = -0.25, b: float = 0.25,
                        domain: tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
    """Analytical solution at time t for 1D advected tophat on periodic domain."""
    L = domain[1] - domain[0]
    # shift x by -c*t (modulo domain)
    xs = (x - c * t - domain[0]) % L + domain[0]
    return ((xs >= a) & (xs <= b)).astype(np.float64)

# Classical baselines (1D)

def upwind_1d(u0: np.ndarray, c: float, dx: float, dt: float, n_steps: int) -> np.ndarray:
    """First-order upwind for u_t + c u_x = 0 on a periodic 1D grid.

    Dissipative scheme — expected to give order 1 convergence.
    """
    u = u0.copy()
    if c >= 0:
        for _ in range(n_steps):
            u = u - c * dt / dx * (u - np.roll(u, 1))
    else:
        for _ in range(n_steps):
            u = u - c * dt / dx * (np.roll(u, -1) - u)
    return u


def lax_wendroff_1d(u0: np.ndarray, c: float, dx: float, dt: float, n_steps: int) -> np.ndarray:
    """Lax-Wendroff for u_t + c u_x = 0 on a periodic 1D grid.

    Second-order accurate on smooth solutions; dispersive (oscillates at shocks).
    """
    u = u0.copy()
    nu = c * dt / dx
    for _ in range(n_steps):
        up = np.roll(u, -1)
        um = np.roll(u, 1)
        u = u - 0.5 * nu * (up - um) + 0.5 * nu ** 2 * (up - 2 * u + um)
    return u



# Classical baselines (2D)

def upwind_2d(u0: np.ndarray, v: tuple[float, float],
            dx: float, dy: float, dt: float, n_steps: int) -> np.ndarray:
    """First-order upwind for 2D linear advection on a periodic grid."""
    u = u0.copy()
    vx, vy = v
    for _ in range(n_steps):
        if vx >= 0:
            fx = vx * dt / dx * (u - np.roll(u, 1, axis=1))
        else:
            fx = vx * dt / dx * (np.roll(u, -1, axis=1) - u)
        if vy >= 0:
            fy = vy * dt / dy * (u - np.roll(u, 1, axis=0))
        else:
            fy = vy * dt / dy * (np.roll(u, -1, axis=0) - u)
        u = u - fx - fy
    return u


def lax_wendroff_2d(u0: np.ndarray, v: tuple[float, float],
                    dx: float, dy: float, dt: float, n_steps: int) -> np.ndarray:
    """2D Lax-Wendroff (dimensional splitting)."""
    u = u0.copy()
    vx, vy = v
    nux = vx * dt / dx
    nuy = vy * dt / dy
    for _ in range(n_steps):
        # x-sweep
        up = np.roll(u, -1, axis=1)
        um = np.roll(u, 1, axis=1)
        u = u - 0.5 * nux * (up - um) + 0.5 * nux ** 2 * (up - 2 * u + um)
        # y-sweep
        up = np.roll(u, -1, axis=0)
        um = np.roll(u, 1, axis=0)
        u = u - 0.5 * nuy * (up - um) + 0.5 * nuy ** 2 * (up - 2 * u + um)
    return u


# ---------------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------------

def make_grid_1d(N: int, domain: tuple[float, float] = (-1.0, 1.0)) -> tuple[np.ndarray, float]:
    """Return (x, dx) for a periodic 1D grid with N cells."""
    L = domain[1] - domain[0]
    dx = L / N
    x = domain[0] + (np.arange(N) + 0.5) * dx
    return x, dx


def make_grid_2d(N: int, domain: tuple[float, float] = (-1.0, 1.0)
                ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (X, Y, dx, dy) for a periodic NxN grid."""
    x, dx = make_grid_1d(N, domain)
    y, dy = make_grid_1d(N, domain)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return X, Y, dx, dy


def stable_dt(dx: float, v_max: float, cfl: float = 0.4) -> float:
    """Return a stable time step for the classical schemes."""
    return cfl * dx / abs(v_max)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1D smoke test
    N = 200
    x, dx = make_grid_1d(N)
    c = 1.0
    t_final = 0.5
    dt = stable_dt(dx, c)
    n_steps = int(np.ceil(t_final / dt))
    dt = t_final / n_steps

    u0 = tophat_1d(x)
    u_up = upwind_1d(u0, c, dx, dt, n_steps)
    u_lw = lax_wendroff_1d(u0, c, dx, dt, n_steps)
    u_ex = analytical_1d_tophat(x, t_final, c=c)

    print(f"[1D tophat smoke test] N={N}, t_final={t_final}")
    print(f"  upwind   L2 error: {np.sqrt(dx * np.sum((u_up - u_ex) ** 2)):.4e}")
    print(f"  Lax-Wen  L2 error: {np.sqrt(dx * np.sum((u_lw - u_ex) ** 2)):.4e}")

    # 2D smoke test
    N = 128
    X, Y, dx, dy = make_grid_2d(N)
    v = (1.0, 0.5)
    t_final = 0.5
    dt = stable_dt(min(dx, dy), max(abs(v[0]), abs(v[1])))
    n_steps = int(np.ceil(t_final / dt))
    dt = t_final / n_steps

    u0 = gaussian_2d(X, Y, sigma=0.15)
    u_up = upwind_2d(u0, v, dx, dy, dt, n_steps)
    u_ex = analytical_2d_gaussian(X, Y, t_final, v=v, sigma=0.15)
    err = np.sqrt(dx * dy * np.sum((u_up - u_ex) ** 2))
    print(f"\n[2D Gaussian smoke test] N={N}, t_final={t_final}")
    print(f"  upwind   L2 error: {err:.4e}")