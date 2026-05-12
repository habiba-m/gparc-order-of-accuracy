"""
Convergence study runner 

Computes L2 error vs mesh spacing for:
-  G-PARC model
- First-order upwind baseline
- Lax-Wendroff baseline

Then fits the slope log(e) ~ p * log(h) to recover the empirical order
of accuracy p.

This isolates the model details from the convergence analysis.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

from numerical.reference import (
    make_grid_1d, make_grid_2d, stable_dt,
    gaussian_2d, tophat_1d,
    analytical_2d_gaussian, analytical_1d_tophat,
    upwind_1d, lax_wendroff_1d,
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
        # guard against zero error
        valid = e > 0
        if valid.sum() < 2:
            return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": int(valid.sum())}
        lh = np.log(h[valid])
        le = np.log(e[valid])
        slope, intercept = np.polyfit(lh, le, 1)
        # R^2
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



# Error metric -- going to use l2 error for fitting, but also report linf for reference

def l2_error(pred: np.ndarray, ref: np.ndarray, cell_volume: float) -> float:
    """Discrete L2 norm: sqrt(sum (pred - ref)^2 * dV)."""
    return float(np.sqrt(cell_volume * np.sum((pred - ref) ** 2)))


def linf_error(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(pred - ref)))


# Study runners
def run_1d_tophat_study(resolutions: list[int],
                        c: float = 1.0,
                        t_final: float = 0.5,
                        predict_fn: Optional[Callable] = None,
                        include_classical: bool = True
                        ) -> ConvergenceStudy:
    """Run convergence study on the 1D tophat problem.

    predict_fn: callable taking (u0, {'x': x, 'dx': dx, 'c': c}, t_final).
                If None, only the classical baselines are evaluated.
    """
    study = ConvergenceStudy(problem="1D tophat (discontinuous)")

    for N in resolutions:
        x, dx = make_grid_1d(N)
        u0 = tophat_1d(x)
        u_exact = analytical_1d_tophat(x, t_final, c=c)
        grid = {"x": x, "dx": dx, "c": c}

        if include_classical:
            # Upwind
            dt = stable_dt(dx, c)
            n_steps = int(np.ceil(t_final / dt)); dt = t_final / n_steps
            t0 = time.time()
            u_up = upwind_1d(u0, c, dx, dt, n_steps)
            rt = (time.time() - t0) * 1000
            study.add(ConvergenceResult("upwind", N, dx,
                                        l2_error(u_up, u_exact, dx),
                                        linf_error(u_up, u_exact), rt))
            # Lax-Wendroff
            t0 = time.time()
            u_lw = lax_wendroff_1d(u0, c, dx, dt, n_steps)
            rt = (time.time() - t0) * 1000
            study.add(ConvergenceResult("lax_wendroff", N, dx,
                                        l2_error(u_lw, u_exact, dx),
                                        linf_error(u_lw, u_exact), rt))

        if predict_fn is not None:
            t0 = time.time()
            u_model = predict_fn(u0, grid, t_final)
            rt = (time.time() - t0) * 1000
            study.add(ConvergenceResult("g_parc", N, dx,
                                        l2_error(u_model, u_exact, dx),
                                        linf_error(u_model, u_exact), rt))

    return study


def run_2d_gaussian_study(resolutions: list[int],
                        v: tuple[float, float] = (1.0, 0.5),
                        sigma: float = 0.15,
                        t_final: float = 0.5,
                        predict_fn: Optional[Callable] = None,
                        include_classical: bool = True
                        ) -> ConvergenceStudy:
    """Run convergence study on the 2D Gaussian advection problem."""
    study = ConvergenceStudy(problem="2D Gaussian (smooth)")

    for N in resolutions:
        X, Y, dx, dy = make_grid_2d(N)
        u0 = gaussian_2d(X, Y, sigma=sigma)
        u_exact = analytical_2d_gaussian(X, Y, t_final, v=v, sigma=sigma)
        cell_vol = dx * dy
        grid = {"X": X, "Y": Y, "dx": dx, "dy": dy, "v": v}

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
    """Run the classical baselines as a self-test. Plug in G-PARC by editing
    the `predict_fn` import below and adjusting the wrapper module."""
    # ---- Replace this with your model wrapper ----
    predict_fn_2d = None
    predict_fn_1d = None
    # Example:
    # from gparc_wrapper import make_predict_fn_2d, make_predict_fn_1d
    # predict_fn_2d = make_predict_fn_2d("checkpoints/gparc_2d.pt")
    # predict_fn_1d = make_predict_fn_1d("checkpoints/gparc_1d.pt")

    study_2d = run_2d_gaussian_study(
        resolutions=[128, 96, 64, 48, 32],
        predict_fn=predict_fn_2d,
    )
    study_2d.print_table()
    study_2d.to_csv("../figures/results_2d.csv")

    study_1d = run_1d_tophat_study(
        resolutions=[512, 256, 128, 64, 32],
        predict_fn=predict_fn_1d,
    )
    study_1d.print_table()
    study_1d.to_csv("../figures/results_1d.csv")


if __name__ == "__main__":
    main()