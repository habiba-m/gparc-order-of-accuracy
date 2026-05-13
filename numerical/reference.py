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


def disc_2d(x: np.ndarray, y: np.ndarray,
            x0: float = 0.0, y0: float = 0.0,
            R: float = 0.3) -> np.ndarray:
    """2D circular disc indicator: 1 inside circle of radius R centred at (x0, y0)."""
    return ((x - x0) ** 2 + (y - y0) ** 2 <= R ** 2).astype(np.float64)


# Analytical solutions -- ground truth

def analytical_2d_gaussian(x: np.ndarray, y: np.ndarray, t: float,
                        v: tuple[float, float] = (1.0, 0.0),
                        x0: float = 0.0, y0: float = 0.0,
                        sigma: float = 0.1,
                        domain: tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
    """Analytical solution at time t for the 2D advected Gaussian on a periodic domain."""
    L = domain[1] - domain[0]
    xc = ((x0 + v[0] * t) - domain[0]) % L + domain[0]
    yc = ((y0 + v[1] * t) - domain[0]) % L + domain[0]
    dx = (x - xc + L / 2) % L - L / 2
    dy = (y - yc + L / 2) % L - L / 2
    return np.exp(-(dx ** 2 + dy ** 2) / sigma ** 2)


def analytical_2d_disc(x: np.ndarray, y: np.ndarray, t: float,
                       v: tuple[float, float] = (1.0, 0.5),
                       x0: float = 0.0, y0: float = 0.0,
                       R: float = 0.3,
                       domain: tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
    """Analytical solution at time t for the 2D advected disc on a periodic domain."""
    L = domain[1] - domain[0]
    xc = ((x0 + v[0] * t) - domain[0]) % L + domain[0]
    yc = ((y0 + v[1] * t) - domain[0]) % L + domain[0]
    dx = (x - xc + L / 2) % L - L / 2
    dy = (y - yc + L / 2) % L - L / 2
    return (dx ** 2 + dy ** 2 <= R ** 2).astype(np.float64)


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
        up = np.roll(u, -1, axis=1)
        um = np.roll(u, 1, axis=1)
        u = u - 0.5 * nux * (up - um) + 0.5 * nux ** 2 * (up - 2 * u + um)
        up = np.roll(u, -1, axis=0)
        um = np.roll(u, 1, axis=0)
        u = u - 0.5 * nuy * (up - um) + 0.5 * nuy ** 2 * (up - 2 * u + um)
    return u


# Grid utilities

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
