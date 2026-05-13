"""
Convergence study runner

Computes L2 error vs mesh spacing for:
- G-PARC model
- First-order upwind baseline
- Lax-Wendroff baseline

Then fits the slope log(e) ~ p * log(h) to recover the empirical order
of accuracy p.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import numpy as np

from numerical.reference import (
    make_grid_2d, stable_dt,
    gaussian_2d, disc_2d,
    analytical_2d_gaussian, analytical_2d_disc,
    upwind_2d, lax_wendroff_2d,
)


# Data structures

@dataclass
class ConvergenceResult:
    """One row of the convergence table."""
    method: str
    N: int
    h: float
    l2_error: float
    linf_error: float
    runtime_ms: float


@dataclass
class ConvergenceStudy:
    """Container holding all results for one problem and fitting utilities."""
    problem: str
    rows: list[ConvergenceResult] = field(default_factory=list)

    def add(self, row: ConvergenceResult):
        self.rows.append(row)

    def methods(self) -> list[str]:
        return sorted({r.method for r in self.rows})

    def fit_order(self, method: str) -> dict:
        """Fit log(error) = p * log(h) + b. Returns slope, intercept, R^2."""
        rs = [r for r in self.rows if r.method == method]
        rs.sort(key=lambda r: r.h)
        if len(rs) < 2:
            return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": len(rs)}
        h = np.array([r.h for r in rs])
        e = np.array([r.l2_error for r in rs])
        valid = e > 0
        if valid.sum() < 2:
            return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": int(valid.sum())}
        lh = np.log(h[valid])
        le = np.log(e[valid])
        slope, intercept = np.polyfit(lh, le, 1)
        pred = slope * lh + intercept
        ss_res = np.sum((le - pred) ** 2)
        ss_tot = np.sum((le - le.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {"slope": float(slope), "intercept": float(intercept),
                "r2": float(r2), "n": int(valid.sum())}

    def to_csv(self, path: str):
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["problem", "method", "N", "h", "l2_error", "linf_error", "runtime_ms"])
            for r in self.rows:
                w.writerow([self.problem, r.method, r.N, r.h,
                            r.l2_error, r.linf_error, r.runtime_ms])

    def print_table(self):
        print(f"\n=== {self.problem} ===")
        for m in self.methods():
            print(f"\n  Method: {m}")
            print(f"  {'N':>6} {'h':>10} {'L2':>14} {'Linf':>14} {'time(ms)':>10}")
            rs = [r for r in self.rows if r.method == m]
            rs.sort(key=lambda r: -r.N)
            for r in rs:
                print(f"  {r.N:>6} {r.h:>10.5f} {r.l2_error:>14.4e} "
                      f"{r.linf_error:>14.4e} {r.runtime_ms:>10.1f}")
            fit = self.fit_order(m)
            print(f"  --> fitted order p = {fit['slope']:.3f}  (R^2 = {fit['r2']:.4f})")


# Error metrics

def l2_error(pred: np.ndarray, ref: np.ndarray, cell_volume: float) -> float:
    """Discrete L2 norm: sqrt(sum (pred - ref)^2 * dV)."""
    return float(np.sqrt(cell_volume * np.sum((pred - ref) ** 2)))


def linf_error(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(pred - ref)))


# Study runners

def run_2d_gaussian_study(resolutions: list[int],
                          v: tuple[float, float] = (1.0, 0.5),
                          sigma: float = 0.5,
                          x0: float = np.pi, y0: float = np.pi,
                          t_final: float = 2.0,
                          domain: tuple[float, float] = (0.0, 2 * np.pi),
                          predict_fn: Optional[Callable] = None,
                          include_classical: bool = True,
                          ) -> ConvergenceStudy:
    """Run convergence study on the 2D Gaussian advection problem."""
    study = ConvergenceStudy(problem="2D Gaussian (smooth)")
    for N in resolutions:
        X, Y, dx, dy = make_grid_2d(N, domain)
        u0 = gaussian_2d(X, Y, x0=x0, y0=y0, sigma=sigma)
        u_exact = analytical_2d_gaussian(X, Y, t_final, v=v, x0=x0, y0=y0,
                                         sigma=sigma, domain=domain)
        cell_vol = dx * dy
        grid = {"X": X, "Y": Y, "dx": dx, "dy": dy, "v": v, "domain": domain}
        if include_classical:
            dt = stable_dt(min(dx, dy), max(abs(v[0]), abs(v[1])))
            n_steps = int(np.ceil(t_final / dt)); dt = t_final / n_steps
            t0 = time.time()
            u_up = upwind_2d(u0, v, dx, dy, dt, n_steps)
            rt = (time.time() - t0) * 1000
            study.add(ConvergenceResult("upwind", N, dx,
                                        l2_error(u_up, u_exact, cell_vol),
                                        linf_error(u_up, u_exact), rt))
            t0 = time.time()
            u_lw = lax_wendroff_2d(u0, v, dx, dy, dt, n_steps)
            rt = (time.time() - t0) * 1000
            study.add(ConvergenceResult("lax_wendroff", N, dx,
                                        l2_error(u_lw, u_exact, cell_vol),
                                        linf_error(u_lw, u_exact), rt))
        if predict_fn is not None:
            t0 = time.time()
            u_model = predict_fn(u0, grid, t_final)
            rt = (time.time() - t0) * 1000
            study.add(ConvergenceResult("g_parc", N, dx,
                                        l2_error(u_model, u_exact, cell_vol),
                                        linf_error(u_model, u_exact), rt))
    return study


def run_2d_disc_study(resolutions: list[int],
                      v: tuple[float, float] = (1.0, 0.5),
                      R: float = 0.6,
                      x0: float = np.pi, y0: float = np.pi,
                      t_final: float = 2.0,
                      domain: tuple[float, float] = (0.0, 2 * np.pi),
                      predict_fn: Optional[Callable] = None,
                      include_classical: bool = True,
                      ) -> ConvergenceStudy:
    """Run convergence study on the 2D disc (non-smooth) advection problem."""
    study = ConvergenceStudy(problem="2D disc (non-smooth)")
    for N in resolutions:
        X, Y, dx, dy = make_grid_2d(N, domain)
        u0 = disc_2d(X, Y, x0, y0, R)
        u_exact = analytical_2d_disc(X, Y, t_final, v=v, x0=x0, y0=y0, R=R, domain=domain)
        cell_vol = dx * dy
        grid = {"X": X, "Y": Y, "dx": dx, "dy": dy, "v": v, "domain": domain}
        if include_classical:
            dt = stable_dt(min(dx, dy), max(abs(v[0]), abs(v[1])))
            n_steps = int(np.ceil(t_final / dt)); dt = t_final / n_steps
            t0 = time.time()
            u_up = upwind_2d(u0, v, dx, dy, dt, n_steps)
            rt = (time.time() - t0) * 1000
            study.add(ConvergenceResult("upwind", N, dx,
                                        l2_error(u_up, u_exact, cell_vol),
                                        linf_error(u_up, u_exact), rt))
            t0 = time.time()
            u_lw = lax_wendroff_2d(u0, v, dx, dy, dt, n_steps)
            rt = (time.time() - t0) * 1000
            study.add(ConvergenceResult("lax_wendroff", N, dx,
                                        l2_error(u_lw, u_exact, cell_vol),
                                        linf_error(u_lw, u_exact), rt))
        if predict_fn is not None:
            t0 = time.time()
            u_model = predict_fn(u0, grid, t_final)
            rt = (time.time() - t0) * 1000
            study.add(ConvergenceResult("g_parc", N, dx,
                                        l2_error(u_model, u_exact, cell_vol),
                                        linf_error(u_model, u_exact), rt))
    return study


# CLI

def main():
    """Run classical baselines for both 2D problems. Run from repo root:
        python -m numerical.convergence
    """
    script_dir = Path(__file__).parent
    out = (script_dir / ".." / "figures").resolve()
    out.mkdir(exist_ok=True)

    resolutions = [256, 128, 64, 32, 16]

    study_gauss = run_2d_gaussian_study(resolutions=resolutions)
    study_gauss.print_table()
    study_gauss.to_csv(str(out / "results_2d.csv"))

    study_disc = run_2d_disc_study(resolutions=resolutions)
    study_disc.print_table()
    study_disc.to_csv(str(out / "results_disc.csv"))


if __name__ == "__main__":
    main()
