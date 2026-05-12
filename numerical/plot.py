"""
Produce all figures used in the report from the convergence CSV files.
Run this after convergence.py has produced results_2d.csv and results_1d.csv.
"""

from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


METHOD_STYLES = {
    "upwind":        dict(label="1st-order upwind",  marker="o", linestyle="--", color="#888888"),
    "lax_wendroff":  dict(label="Lax-Wendroff",      marker="s", linestyle="--", color="#555555"),
    "g_parc":        dict(label="G-PARC (ours)",     marker="D", linestyle="-",  color="#1f77b4"),
}


def load_csv(path: str) -> dict:
    """Returns {method: {'h': [...], 'l2': [...]}}."""
    by_method = defaultdict(lambda: {"h": [], "l2": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            by_method[row["method"]]["h"].append(float(row["h"]))
            by_method[row["method"]]["l2"].append(float(row["l2_error"]))
    for m in by_method:
        order = np.argsort(by_method[m]["h"])
        by_method[m]["h"] = np.array(by_method[m]["h"])[order]
        by_method[m]["l2"] = np.array(by_method[m]["l2"])[order]
    return dict(by_method)


def fit_slope(h, e):
    valid = e > 0
    if valid.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(h[valid]), np.log(e[valid]), 1)[0])


def plot_convergence(csv_path: str, out_path: str, title: str):
    by_method = load_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method, data in by_method.items():
        style = METHOD_STYLES.get(method, dict(label=method, marker="x"))
        ax.loglog(data["h"], data["l2"], **style)
        slope = fit_slope(data["h"], data["l2"])
        # annotate slope at last point
        ax.annotate(f"p={slope:.2f}",
                    xy=(data["h"][-1], data["l2"][-1]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=9, color=style.get("color", "black"))

    # reference slope guides
    h_ref = np.array([min(min(d["h"]) for d in by_method.values()),
                    max(max(d["h"]) for d in by_method.values())])
    e_anchor = max(max(d["l2"]) for d in by_method.values())
    h_anchor = max(max(d["h"]) for d in by_method.values())
    for p, ls in [(1, ":"), (2, "-.")]:
        c = e_anchor * (h_ref / h_anchor) ** p
        ax.loglog(h_ref, c, ls, color="lightgray", linewidth=1,
                label=f"O(h^{p}) reference")

    ax.set_xlabel("mesh spacing $h$")
    ax.set_ylabel("$L^2$ error at final time")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_qualitative_2d(out_path: str):
    """A simple dataset / behavior visualization. Pure illustration; doesn't
    need the trained model. Shows initial, midpoint, and final analytical state
    plus what upwind does to it (visualizing dissipation)."""
    from numerical.reference import make_grid_2d, gaussian_2d, analytical_2d_gaussian, upwind_2d, stable_dt

    N = 64
    X, Y, dx, dy = make_grid_2d(N)
    v = (1.0, 0.5); sigma = 0.15
    u0 = gaussian_2d(X, Y, sigma=sigma)
    times = [0.0, 0.25, 0.5]
    dt = stable_dt(min(dx, dy), max(abs(v[0]), abs(v[1])))

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    for j, t in enumerate(times):
        ex = analytical_2d_gaussian(X, Y, t, v=v, sigma=sigma)
        axes[0, j].imshow(ex, origin="lower", extent=[-1, 1, -1, 1], cmap="viridis", vmin=0, vmax=1)
        axes[0, j].set_title(f"Analytical, t={t}")

        if t == 0.0:
            axes[1, j].imshow(u0, origin="lower", extent=[-1, 1, -1, 1], cmap="viridis", vmin=0, vmax=1)
        else:
            n_steps = int(np.ceil(t / dt)); dtj = t / n_steps
            u_up = upwind_2d(u0, v, dx, dy, dtj, n_steps)
            axes[1, j].imshow(u_up, origin="lower", extent=[-1, 1, -1, 1], cmap="viridis", vmin=0, vmax=1)
        axes[1, j].set_title(f"Upwind, t={t}")

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("2D Gaussian advection: analytical vs first-order upwind (showing dissipation)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_qualitative_1d(out_path: str):
    from numerical.reference import make_grid_1d, tophat_1d, analytical_1d_tophat, upwind_1d, lax_wendroff_1d, stable_dt

    N = 256
    x, dx = make_grid_1d(N)
    c = 1.0; t_final = 0.5
    dt = stable_dt(dx, c); n_steps = int(np.ceil(t_final / dt)); dt = t_final / n_steps

    u0 = tophat_1d(x)
    u_ex = analytical_1d_tophat(x, t_final, c=c)
    u_up = upwind_1d(u0, c, dx, dt, n_steps)
    u_lw = lax_wendroff_1d(u0, c, dx, dt, n_steps)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, u_ex, "k-", linewidth=2, label="Analytical")
    ax.plot(x, u_up, "--", color="#888", label="Upwind (smears)")
    ax.plot(x, u_lw, ":", color="#444", label="Lax-Wendroff (oscillates)")
    ax.set_xlabel("x"); ax.set_ylabel("u")
    ax.set_title(f"1D tophat advection at t={t_final} (N={N})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    out = Path("../figures")
    out.mkdir(exist_ok=True)
    print("Generating figures...")
    if Path("results_2d.csv").exists():
        plot_convergence("results_2d.csv", str(out / "convergence_2d.png"),
                        "Order of accuracy: 2D Gaussian (smooth)")
    if Path("results_1d.csv").exists():
        plot_convergence("results_1d.csv", str(out / "convergence_1d.png"),
                        "Order of accuracy: 1D tophat (discontinuous)")
    plot_qualitative_2d(str(out / "qualitative_2d.png"))
    plot_qualitative_1d(str(out / "qualitative_1d.png"))
    print("Done.")


if __name__ == "__main__":
    main()