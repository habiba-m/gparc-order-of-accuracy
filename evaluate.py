"""
Measures the empirical order of accuracy of the trained G-PARC model
by evaluating single-step prediction error across mesh resolutions.

For each resolution h = 2π/N:
  - Feed the exact state, predict one step forward
  - Compute Rel-L2 and L∞ error against analytical solution
  - Average over 10 held-out simulations

Fit log(error) vs log(h) to extract the convergence slope p:
  slope ≈ 1.0  →  first-order (dissipative, like upwind FD)
  slope < 1.0  →  sub-first-order (strong implicit dissipation)
  slope ≈ 2.0  →  matches MLS theoretical consistency order
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.data import Data

sys.path.insert(0, str(Path.home() / "G-PARC"))
from differentiator.hop import SolveGradientsLST, AdvectionMLS
from differentiator.mappingandrecon import MappingAndRecon
from utilities.featureextractor import GraphConvFeatureExtractorV2
from utilities.embed import SimulationConditionedLayerNorm

# CONFIG
# EVAL_NS   = [256, 128, 64, 32, 16]   # resolutions to evaluate
# EVAL_NS = [64, 32, 16]
EVAL_NS = [64, 48, 32, 24, 16]
L = 2.0 * np.pi  # domain length
HIDDEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GPARCAdvection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_solver = SolveGradientsLST()
        self.advection = AdvectionMLS(self.grad_solver)
        self.extractor = GraphConvFeatureExtractorV2(
            in_channels=2,
            hidden_channels=HIDDEN,
            out_channels=HIDDEN,
            num_layers=4,
            use_layer_norm=True,
            use_relative_pos=True,
        )
        self.film = SimulationConditionedLayerNorm(normalized_shape=HIDDEN, global_dim=3)
        self.mar = MappingAndRecon(
            n_base_features=HIDDEN,
            n_mask_channel=1,
            output_channel=1,
            heads=4,
            zero_init=True,
        )

    def init_mls(self, sample: Data):
        pos = sample.x[:, :2]
        ei = sample.edge_index
        self.grad_solver(Data(pos=pos, edge_index=ei, mesh_id=sample.mesh_id), torch.zeros(pos.shape[0], 1, device=pos.device))

    def step(self, data: Data, clamp: bool = False) -> torch.Tensor:
        pos = data.x[:, :2]
        u = data.x[:, 2:3]
        vel = data.x[:, 3:5]
        dt = data.global_delta_t.flatten()[0].item()
        gp = data.global_params
        ei = data.edge_index
        mesh = Data(pos=pos, edge_index=ei, mesh_id=data.mesh_id)
        adv = self.advection(u, vel, mesh)
        feats = self.extractor(pos, ei, pos=pos)
        feats = self.film(feats, gp)
        du_dt = self.mar(feats, adv, ei)
        out = u + dt * du_dt
        return torch.clamp(out, 0.0, 1.0) if clamp else out


def load_model(path, device):
    model = GPARCAdvection().to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def rollout_and_error(model, sim, device, problem):
    clamp = problem == "disc"

    # Average single-step error across all 19 transitions
    l2_errors, linf_errors = [], []

    for step_data in sim:
        step_data = step_data.to(device)
        # Feed exact current state, predict one step
        inp = Data(
            x=step_data.x,  # exact u at t
            edge_index=step_data.edge_index,
            global_params=step_data.global_params,
            global_delta_t=step_data.global_delta_t,
            mesh_id=step_data.mesh_id,
        )
        u_pred = model.step(inp, clamp=clamp)
        u_exact = step_data.y[:, 0:1].to(device)

        l2 = (torch.norm(u_pred - u_exact) / (torch.norm(u_exact) + 1e-8)).item()
        linf = (u_pred - u_exact).abs().max().item()
        l2_errors.append(l2)
        linf_errors.append(linf)

    return np.mean(l2_errors), np.mean(linf_errors)


def evaluate_resolution(model, problem, N, device):
    eval_dir = Path(f"data/{problem}/eval/res_{N}")
    files = sorted(eval_dir.glob("*.pt"))

    first_sim = torch.load(files[0], weights_only=False)
    model.grad_solver.clear_caches()  # clear stale geometry
    model.init_mls(first_sim[0].to(device))

    l2_errors, linf_errors = [], []
    for f in files:
        sim = torch.load(f, weights_only=False)
        l2, linf = rollout_and_error(model, sim, device, problem)
        l2_errors.append(l2)
        linf_errors.append(linf)

    return np.mean(l2_errors), np.mean(linf_errors)


# CONVERGENCE STUDY
def run_convergence_study(problem, device):
    print(f"\n{'='*55}")
    print(f"  Convergence Study: {problem}")
    print(f"{'='*55}")

    model = load_model(f"outputs/{problem}/best_model.pth", device)

    h_values = []
    l2_errors = []
    linf_errors = []

    print(f"\n{'N':>6}  {'h':>8}  {'Rel-L2':>12}  {'L-inf':>12}")
    print("-" * 45)

    for N in EVAL_NS:
        h = L / N
        l2, linf = evaluate_resolution(model, problem, N, device)
        h_values.append(h)
        l2_errors.append(l2)
        linf_errors.append(linf)
        print(f"{N:>6}  {h:>8.4f}  {l2:>12.4e}  {linf:>12.4e}")

    h_arr = np.array(h_values)
    l2_arr = np.array(l2_errors)
    li_arr = np.array(linf_errors)

    # Fit log(error) = p * log(h) + C  →  slope p = order of accuracy
    log_h = np.log(h_arr)
    p_l2, _ = np.polyfit(log_h, np.log(l2_arr), 1)
    p_linf, _ = np.polyfit(log_h, np.log(li_arr), 1)

    print(f"\n  Empirical order of accuracy:")
    print(f"    Rel-L2  slope: {p_l2:.3f}")
    print(f"    L-inf   slope: {p_linf:.3f}")

    if p_l2 >= 1.8:
        interp = "close to MLS theoretical order (~2)"
    elif p_l2 >= 0.8:
        interp = "approximately first-order (dissipative)"
    elif p_l2 >= 0.3:
        interp = "sub-first-order (strong implicit dissipation)"
    else:
        interp = "near-zero slope (saturated error)"
    print(f"    Interpretation: {interp}")

    return h_arr, l2_arr, li_arr, p_l2, p_linf


# PLOT
def plot_convergence(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"l2": "steelblue", "linf": "firebrick"}

    for ax, (problem, (h, l2, linf, p_l2, p_linf)) in zip(axes, results.items()):
        h_min, h_max = h.min(), h.max()
        ref_h = np.array([h_min * 0.7, h_max * 1.3])

        ax.loglog(ref_h, l2[0] * (ref_h / h_min) ** 1, "k--", lw=1, label="O(h¹) reference")
        ax.loglog(ref_h, l2[0] * (ref_h / h_min) ** 2, "k:", lw=1, label="O(h²) reference")

        ax.loglog(h, l2, "o-", color=colors["l2"], lw=2, ms=7, label=f"Rel-L2  (slope={p_l2:.2f})")
        ax.loglog(h, linf, "s-", color=colors["linf"], lw=2, ms=7, label=f"L-inf   (slope={p_linf:.2f})")

        ax.set_xlim(h_min * 0.7, h_max * 1.5)
        ax.set_xlabel("Mesh spacing  h = 2π/N", fontsize=12)
        ax.set_ylabel("Error", fontsize=12)
        ax.set_title(f"G-PARC Convergence — {problem.capitalize()}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = Path("outputs/convergence_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out}")
    plt.close()


# MAIN
if __name__ == "__main__":
    device = torch.device(DEVICE)
    results = {}

    for problem in ["gaussian", "disc"]:
        h, l2, linf, p_l2, p_linf = run_convergence_study(problem, device)
        results[problem] = (h, l2, linf, p_l2, p_linf)

    plot_convergence(results)

    # Final summary
    print(f"\n{'='*55}")
    print(f"  Summary")
    print(f"{'='*55}")
    print(f"  {'Problem':<12} {'L2 slope':>10} {'Linf slope':>12}")
    print(f"  {'-'*36}")
    for problem, (_, _, _, p_l2, p_linf) in results.items():
        print(f"  {problem:<12} {p_l2:>10.3f} {p_linf:>12.3f}")
    print(f"\n  Reference: O(h¹)=1.0  O(h²)=2.0  MLS~2.0")
    print(f"{'='*55}")
