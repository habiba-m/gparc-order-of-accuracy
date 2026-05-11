"""
MLS (Moving Least Squares) differential operators for G-PARC.

Closely follows the production implementation in JackBeerman/G-PARC:
  differentiator/hop.py — SolveGradientsLST, SolveWeightLST2d, apply_laplacian

Key design decisions (from the actual repo):
  - Gradient solver: ridge regularization (1e-8) + linalg.inv, pinv fallback.
    Diagnostic confirmed gradients are exact at all neighbor counts even for 3-nbr
    nodes — the 2×2 moment matrix is always well-conditioned.
  - Laplacian solver: 5×5 polynomial basis needs ≥5 neighbors to be determined.
    On 4-neighbor meshes the cross-term column is zero (rank-4 matrix), which is
    fine for a cardinal grid but bad for boundary/irregular nodes.
    FIX: 2-hop stencil extension adds neighbor-of-neighbor edges when a node has
    fewer than `min_2hop_neighbors` (default 6) direct neighbors.  M̃ is built
    from the augmented stencil (better conditioned), but Laplacian weights are
    only computed for ORIGINAL edges so the stencil size doesn't change.
  - Geometry is cached per mesh_id (or pos.data_ptr() as fallback) so repeated
    forward calls on the same mesh topology pay zero overhead.
"""
from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data


# ── 2-hop stencil extension ──────────────────────────────────────────────────

def compute_2hop_extension(pos: Tensor, edge_index: Tensor, min_neighbors: int = 6) -> Tensor:
    """
    Augment edge_index by adding 2-hop edges at nodes with too few neighbors.

    For nodes with fewer than `min_neighbors` direct neighbors, adds edges to
    the nearest neighbors-of-neighbors so the 5×5 MLS moment matrix becomes
    well-determined.  Precomputed once per mesh; zero per-timestep cost.

    Args:
        pos:           [N, 2] node positions
        edge_index:    [2, E] original directed edges
        min_neighbors: target minimum neighbor count (default 6, overdetermined)

    Returns:
        augmented edge_index [2, E'] (original edges + extra 2-hop edges)
    """
    N = pos.shape[0]
    row_np = edge_index[0].cpu().numpy()
    col_np = edge_index[1].cpu().numpy()
    pos_np = pos.detach().cpu().numpy()

    adj: dict[int, set[int]] = defaultdict(set)
    for e in range(len(row_np)):
        adj[int(row_np[e])].add(int(col_np[e]))

    extra_rows, extra_cols = [], []
    for node_i in range(N):
        if len(adj[node_i]) >= min_neighbors:
            continue
        needed = min_neighbors - len(adj[node_i])
        two_hop = {
            nbr2
            for nbr in adj[node_i]
            for nbr2 in adj[nbr]
            if nbr2 != node_i and nbr2 not in adj[node_i]
        }
        if not two_hop:
            continue
        two_hop_list = list(two_hop)
        pi = pos_np[node_i]
        dists = ((pos_np[two_hop_list] - pi) ** 2).sum(axis=1)
        for k in dists.argsort()[: min(needed, len(two_hop_list))]:
            extra_rows.append(node_i)
            extra_cols.append(two_hop_list[k])

    if not extra_rows:
        return edge_index

    extra = torch.stack([
        torch.tensor(extra_rows, dtype=torch.long, device=edge_index.device),
        torch.tensor(extra_cols, dtype=torch.long, device=edge_index.device),
    ])
    return torch.cat([edge_index, extra], dim=1)


# ── Gradient solver ───────────────────────────────────────────────────────────

class SolveGradientsLST(nn.Module):
    """
    MLS Gradient Solver using a 2×2 linear polynomial basis.

    Matches SolveGradientsLST from JackBeerman/G-PARC differentiator/hop.py.
    Gradients are exact at all neighbor counts (diagnostic confirmed).
    No 2-hop extension needed — the 2×2 system is always well-conditioned.

    Geometry (M_inv, delta_r) is cached per mesh_id for zero per-timestep overhead.
    """

    def __init__(self, grad_limit: float = 30_000.0):
        super().__init__()
        self.grad_limit = grad_limit
        self._geo_cache: dict = {}  # key → (M_inv, delta_r)

    # ── caching ──────────────────────────────────────────────────────────────

    def _cache_key(self, data: Data):
        if hasattr(data, "mesh_id") and data.mesh_id is not None:
            mid = data.mesh_id
            return mid.item() if mid.numel() == 1 else tuple(mid.tolist())
        return data.pos.data_ptr()

    def clear_cache(self):
        self._geo_cache.clear()

    # ── geometry precomputation ───────────────────────────────────────────────

    def _precompute(self, pos: Tensor, edge_index: Tensor):
        """Compute M_inv [N,2,2] and delta_r [E,2] from geometry."""
        N = pos.shape[0]
        row, col = edge_index
        delta_r = (pos[col] - pos[row]).detach().float()  # [E, 2]

        # M_i = Σ_j δr_ij ⊗ δr_ij  (2×2)
        M_edge = torch.bmm(delta_r.unsqueeze(2), delta_r.unsqueeze(1))  # [E,2,2]
        M = torch.zeros(N, 2, 2, device=pos.device, dtype=torch.float32)
        M.index_add_(0, row, M_edge)

        # Ridge regularization for ill-conditioned distorted meshes
        M = M + 1e-8 * torch.eye(2, device=pos.device, dtype=torch.float32).unsqueeze(0)

        try:
            M_inv = torch.linalg.inv(M)
        except RuntimeError:
            M_inv = torch.linalg.pinv(M)

        return M_inv, delta_r

    # ── per-variable gradient ─────────────────────────────────────────────────

    def _grad_one(self, u: Tensor, row: Tensor, col: Tensor,
                  M_inv: Tensor, delta_r: Tensor) -> Tensor:
        """Gradient of a single scalar field u [N,1] → [N,2]."""
        N = M_inv.shape[0]
        du = (u[col] - u[row]).float()               # [E, 1]
        V_edge = delta_r.unsqueeze(2) * du.unsqueeze(1)  # [E, 2, 1]
        V = torch.zeros(N, 2, 1, device=u.device, dtype=torch.float32)
        V.index_add_(0, row, V_edge)
        grads = torch.bmm(M_inv, V).squeeze(2)       # [N, 2]
        return torch.clamp(grads, -self.grad_limit, self.grad_limit).to(u.dtype)

    # ── public API ────────────────────────────────────────────────────────────

    def forward(self, data: Data, field: Tensor) -> list[Tensor]:
        """
        Compute MLS gradient for each channel of `field`.

        Args:
            data:  PyG Data with .pos [N,2] and .edge_index [2,E]
            field: [N, C] state field

        Returns:
            list of C tensors, each [N, 2]  (∂field[:,c]/∂x, ∂field[:,c]/∂y)
        """
        key = self._cache_key(data)
        pos, edge_index = data.pos, data.edge_index

        if key in self._geo_cache:
            M_inv, delta_r = self._geo_cache[key]
            if M_inv.device != pos.device:
                M_inv = M_inv.to(pos.device)
                delta_r = delta_r.to(pos.device)
                self._geo_cache[key] = (M_inv, delta_r)
        else:
            M_inv, delta_r = self._precompute(pos, edge_index)
            self._geo_cache[key] = (M_inv, delta_r)

        row, col = edge_index
        return [self._grad_one(field[:, i: i + 1], row, col, M_inv, delta_r)
                for i in range(field.shape[1])]


# ── Laplacian solver ──────────────────────────────────────────────────────────

class SolveWeightLST2d(nn.Module):
    """
    MLS Laplacian Solver using a 5-term quadratic polynomial basis.

    Matches SolveWeightLST2d from JackBeerman/G-PARC differentiator/hop.py.

    Basis:  [δx, δy, δx·δy, δx², δy²]
    L:      [0,  0,  0,     2,   2  ]   (∇² of the quadratic basis)

    The 5×5 moment matrix needs ≥5 neighbors to be well-conditioned.  On meshes
    with 4-neighbor connectivity (including boundary nodes of unstructured meshes)
    the system is rank-deficient.  Fix: 2-hop stencil extension — M̃ is built
    from augmented edges, but weights are applied on ORIGINAL edges only.

    Weights are cached per mesh_id; cost paid once per unique mesh topology.
    """

    def __init__(self, weight_limit: float = 5_000.0, use_2hop: bool = True,
                 min_2hop_neighbors: int = 6):
        super().__init__()
        self.weight_limit = weight_limit
        self.use_2hop = use_2hop
        self.min_2hop_neighbors = min_2hop_neighbors
        self._weights_cache: dict = {}
        self._edge_aug_cache: dict = {}

    # ── caching ──────────────────────────────────────────────────────────────

    def _cache_key(self, data: Data):
        if hasattr(data, "mesh_id") and data.mesh_id is not None:
            mid = data.mesh_id
            return mid.item() if mid.numel() == 1 else tuple(mid.tolist())
        return data.pos.data_ptr()

    def clear_cache(self):
        self._weights_cache.clear()
        self._edge_aug_cache.clear()

    # ── polynomial basis helpers ──────────────────────────────────────────────

    @staticmethod
    def _basis(delta: Tensor) -> Tensor:
        """Quadratic polynomial basis from relative positions: [δx, δy, δx·δy, δx², δy²]."""
        x, y = delta[:, 0:1], delta[:, 1:2]
        return torch.cat([x, y, x * y, x * x, y * y], dim=1).float()  # [E, 5]

    # ── augmented stencil ─────────────────────────────────────────────────────

    def _get_augmented_edge_index(self, data: Data, key) -> Tensor:
        if key in self._edge_aug_cache:
            aug = self._edge_aug_cache[key]
            if aug.device != data.pos.device:
                aug = aug.to(data.pos.device)
                self._edge_aug_cache[key] = aug
            return aug
        aug = compute_2hop_extension(data.pos, data.edge_index, self.min_2hop_neighbors)
        self._edge_aug_cache[key] = aug
        return aug

    # ── weight computation ────────────────────────────────────────────────────

    def forward(self, data: Data) -> Tensor:
        """
        Compute per-edge Laplacian weights for the original edge_index.

        Args:
            data: PyG Data with .pos [N,2], .edge_index [2,E], .mesh_id (optional)

        Returns:
            weights: [E] per-edge Laplacian weights  (to be used with apply_laplacian)
        """
        key = self._cache_key(data)
        pos = data.pos
        orig_ei = data.edge_index

        if key in self._weights_cache:
            w = self._weights_cache[key]
            if w.device != pos.device:
                w = w.to(pos.device)
                self._weights_cache[key] = w
            return w

        # ── Step 1: build M̃ using augmented (or original) edges ─────────────
        aug_ei = self._get_augmented_edge_index(data, key) if self.use_2hop else orig_ei
        aug_row, aug_col = aug_ei
        H_aug = self._basis((pos[aug_col] - pos[aug_row]).detach())  # [E_aug, 5]

        M_tilde = torch.zeros(pos.shape[0], 5, 5, device=pos.device, dtype=torch.float32)
        M_tilde.index_add_(0, aug_row, torch.bmm(H_aug.unsqueeze(2), H_aug.unsqueeze(1)))

        M_tilde = M_tilde + 1e-7 * torch.eye(5, device=pos.device, dtype=torch.float32).unsqueeze(0)

        try:
            M_tilde_inv = torch.linalg.inv(M_tilde)
        except RuntimeError:
            M_tilde_inv = torch.linalg.pinv(M_tilde)

        # ── Step 2: weights on ORIGINAL edges with M̃_inv from augmented ─────
        # L = ∇²[δx, δy, δxδy, δx², δy²] = [0,0,0,2,2]
        L = torch.zeros(5, dtype=torch.float32, device=pos.device)
        L[3], L[4] = 2.0, 2.0  # ∂²/∂x² of x² = 2, ∂²/∂y² of y² = 2

        # C_i = M̃_i^{-1} L  [N, 5]
        C = torch.bmm(M_tilde_inv, L.unsqueeze(0).unsqueeze(2).expand(pos.shape[0], 5, 1)).squeeze(2)

        orig_row, orig_col = orig_ei
        H_orig = self._basis((pos[orig_col] - pos[orig_row]).detach())  # [E_orig, 5]
        weights = (C[orig_row] * H_orig).sum(dim=1)                     # [E_orig]
        weights = torch.clamp(weights, -self.weight_limit, self.weight_limit)

        self._weights_cache[key] = weights
        return weights


# ── Laplacian application ─────────────────────────────────────────────────────

def apply_laplacian(data: Data, u: Tensor, weights: Tensor) -> Tensor:
    """
    Apply precomputed MLS Laplacian weights to field u.

    ∇²u_i = Σ_j w_ij (u_j - u_i)

    Args:
        data:    PyG Data with .edge_index (original, matching `weights`)
        u:       [N, C] field values
        weights: [E] per-edge weights from SolveWeightLST2d

    Returns:
        lap_u: [N, C]
    """
    row, col = data.edge_index
    N = data.pos.shape[0]
    diff = u[col] - u[row]                        # [E, C]
    lap = torch.zeros(N, u.shape[1], device=u.device, dtype=u.dtype)
    lap.index_add_(0, row, weights.unsqueeze(1) * diff)
    return lap
