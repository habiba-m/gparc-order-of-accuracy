"""
G-PARC: Graph Physics-Aware Recurrent Convolutional Network.

Architecture (Eq. 6 in the paper):
  Phi_theta(s, grad_s, lap_s, c) = MLP_theta([s, grad_s, lap_s] || R_theta(s, c))

  1. SolveGradientsLST  -> grad_s  [N, 2*F]  (MLS, no learned parameters)
  2. SolveWeightLST2d   -> weights [E]        (MLS, cached per mesh topology)
     apply_laplacian    -> lap_s   [N, F]
  3. SourceGNN          -> R       [N, source_dim]  (SAGEConv message passing)
  4. FusionMLP          -> ds/dt   [N, F]           (learned PDE right-hand side)
  5. Integrator         -> s_{t+dt} = s_t + integral(Phi dt)  (Euler / Heun / RK4)

The MLS solvers match the production implementation in JackBeerman/G-PARC
(differentiator/hop.py): ridge regularization, per-mesh caching, gradient/
weight clamping, and 2-hop stencil extension for the Laplacian.

NOTE: The neural architecture here (SAGEConv + MLP) is a simplified version
of the full G-PARC model, which uses GAT + SPADE + GraphResNet.  The MLS
physics operators are faithful to the original.
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from .mls import SolveGradientsLST, SolveWeightLST2d, apply_laplacian


# ── Helper modules ───────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Feed-forward MLP with SiLU activations."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SourceGNN(nn.Module):
    """
    GNN approximating the source term R(s, static_feats).

    Uses SAGEConv layers (mean aggregation of neighbor features + self).
    In the full G-PARC model this is replaced by a deeper GAT network with
    SPADE normalization, but SAGEConv captures the same structural inductive bias.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2):
        super().__init__()
        assert n_layers >= 1
        channels = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(
            SAGEConv(channels[i], channels[i + 1]) for i in range(n_layers)
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.act(x)
        return x


# ── Main model ───────────────────────────────────────────────────────────────

class GPARC(nn.Module):
    """
    Graph Physics-Aware Recurrent Convolutional Network.

    Input convention (matches data/data.py):
        data.x        = [static_feats | s_t]   [N, static_dim + state_dim]
        data.y        = s_{t+1}                [N, state_dim]
        data.pos      = node positions          [N, 2]
        data.edge_index                         [2, E]
        data.mesh_id  = simulation identifier   (used for MLS caching)

    Output:
        s_{t+1} prediction  [N, state_dim]

    Args:
        state_dim:      number of dynamic state variables (1 for scalar advection)
        static_dim:     number of static node features (4 for [x, y, vx, vy])
        hidden_dim:     hidden width shared by GNN and MLP layers
        source_out_dim: output dimension of the source-term GNN
        n_gnn_layers:   depth of the source-term GNN
        n_mlp_layers:   depth of the fusion MLP (>= 2)
        dt:             physical timestep Dt used by the integrator
        integrator:     'euler' | 'heun' | 'rk4'
        use_2hop:       whether to use 2-hop stencil extension for Laplacian
                        (recommended for unstructured meshes; default True)
    """

    def __init__(
        self,
        state_dim: int = 1,
        static_dim: int = 4,
        hidden_dim: int = 64,
        source_out_dim: int = 32,
        n_gnn_layers: int = 2,
        n_mlp_layers: int = 3,
        dt: float = 0.01,
        integrator: str = "euler",
        use_2hop: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.static_dim = static_dim
        self.dt = dt
        self.integrator = integrator

        # MLS differential operators (no learned parameters)
        self.grad_solver = SolveGradientsLST()
        self.lap_solver  = SolveWeightLST2d(use_2hop=use_2hop)

        # Source term GNN: input is [static, s]
        self.source_gnn = SourceGNN(
            in_dim=static_dim + state_dim,
            hidden_dim=hidden_dim,
            out_dim=source_out_dim,
            n_layers=n_gnn_layers,
        )

        # Fusion MLP: [s, grad_sx, grad_sy, lap_s, R] -> ds/dt
        # grad_s contributes 2*state_dim features (x and y per field)
        mlp_in = state_dim + 2 * state_dim + state_dim + source_out_dim
        self.fusion_mlp = MLP(mlp_in, hidden_dim, state_dim, n_mlp_layers)

    def clear_mls_cache(self):
        """Clear cached MLS geometry (call when mesh topology changes)."""
        self.grad_solver.clear_cache()
        self.lap_solver.clear_cache()

    # ── Core physics operator ────────────────────────────────────────────────

    def _dsdt(self, s: Tensor, static: Tensor, data: Data) -> Tensor:
        """Evaluate Phi_theta(s, grad_s, lap_s, c) -> ds/dt  [N, state_dim]."""
        N = s.shape[0]

        # MLS spatial operators (cached per mesh topology)
        grad_list = self.grad_solver(data, s)              # list of state_dim [N,2] tensors
        lap_w     = self.lap_solver(data)                  # [E] Laplacian weights

        # Stack gradients: [[N,2], [N,2], ...] -> [N, 2*state_dim]
        # Ordering: [grad_x_f0, grad_y_f0, grad_x_f1, grad_y_f1, ...]
        grad_flat = torch.cat(grad_list, dim=1)            # [N, 2*state_dim]

        lap_s = apply_laplacian(data, s.float(), lap_w).to(s.dtype)  # [N, state_dim]

        # Source term GNN
        R = self.source_gnn(torch.cat([static, s], dim=1), data.edge_index)

        # Fuse and predict ds/dt
        return self.fusion_mlp(torch.cat([s, grad_flat, lap_s, R], dim=1))

    # ── Time integration ─────────────────────────────────────────────────────

    def forward(self, data: Data) -> Tensor:
        """
        Predict s_{t+dt} from data (single autoregressive step).

        Returns:
            s_next: [N, state_dim]
        """
        static = data.x[:, : self.static_dim]
        s      = data.x[:, self.static_dim :]

        if self.integrator == "euler":
            s_next = s + self.dt * self._dsdt(s, static, data)

        elif self.integrator == "heun":
            k1 = self._dsdt(s, static, data)
            # Build intermediate data object with updated state
            data_mid = Data(x=torch.cat([static, s + self.dt * k1], dim=1),
                            pos=data.pos, edge_index=data.edge_index,
                            mesh_id=getattr(data, "mesh_id", None))
            k2 = self._dsdt(s + self.dt * k1, static, data_mid)
            s_next = s + 0.5 * self.dt * (k1 + k2)

        elif self.integrator == "rk4":
            k1 = self._dsdt(s, static, data)
            k2 = self._dsdt(s + 0.5 * self.dt * k1, static,
                            self._swap_state(data, s + 0.5 * self.dt * k1, static))
            k3 = self._dsdt(s + 0.5 * self.dt * k2, static,
                            self._swap_state(data, s + 0.5 * self.dt * k2, static))
            k4 = self._dsdt(s + self.dt * k3, static,
                            self._swap_state(data, s + self.dt * k3, static))
            s_next = s + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        else:
            raise ValueError(f"Unknown integrator '{self.integrator}'. "
                             "Choose from 'euler', 'heun', 'rk4'.")

        return s_next

    @staticmethod
    def _swap_state(data: Data, s_new: Tensor, static: Tensor) -> Data:
        """Return a lightweight Data view with updated dynamic state."""
        return Data(
            x          = torch.cat([static, s_new], dim=1),
            pos        = data.pos,
            edge_index = data.edge_index,
            mesh_id    = getattr(data, "mesh_id", None),
        )

    # ── Rollout utility ──────────────────────────────────────────────────────

    @torch.no_grad()
    def rollout(self, sim: list, device: torch.device | str = "cpu") -> Tensor:
        """
        Autoregressive rollout over a full simulation sequence.

        Args:
            sim:    list of Data objects (one per timestep), as saved by data/data.py
            device: inference device

        Returns:
            preds: [T, N, state_dim] predicted states
                   (index t = prediction for timestep t+1)
        """
        self.eval()
        d0      = sim[0]
        static  = d0.x[:, : self.static_dim].to(device)
        pos     = d0.pos.to(device)
        ei      = d0.edge_index.to(device)
        mesh_id = getattr(d0, "mesh_id", None)
        s       = d0.x[:, self.static_dim :].to(device)

        preds = []
        for _ in sim:
            cur = Data(
                x          = torch.cat([static, s], dim=1),
                pos        = pos,
                edge_index = ei,
                mesh_id    = mesh_id,
            )
            s = self.forward(cur)
            preds.append(s.cpu())

        return torch.stack(preds, dim=0)  # [T, N, state_dim]
