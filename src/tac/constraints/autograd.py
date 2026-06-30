from typing import List

import torch

from tac.core.state import State
from tac.constraints.base import Constraint
from tac.constraints.util import _EPS


class AutogradConstraint(Constraint):
    """A generic XPBD constraint defined by a differentiable measure C(pts).

    Subclasses override `C` to return the *raw* scalar measure (a length, an angle,
    a volume, ...). The base captures its value at construction as `rest`, and the
    quantity actually driven to zero is `residual(pts, rest)`, which defaults to
    `C(pts) - rest`. Override `residual` when the comparison isn't a plain
    subtraction (e.g. an angle that must wrap to (-pi, pi]).

    The gradient d(residual)/dx is taken with autograd, so only the forward pass is
    needed. If the measure depends only on relative positions (translation
    invariant) then sum_i d(residual)/dx_i == 0 automatically, so the update
    conserves momentum and is conservative (no spurious energy injection).

    C must be built from differentiable torch ops (no .item(), no numpy).
    """

    constraint_types = ["measured", "simulated"]

    def __init__(self, state: State, point_idx: List[int], alpha: float):
        super().__init__(state, alpha)
        self.point_idx = list(point_idx)
        self.idx_t = torch.as_tensor(self.point_idx, dtype=torch.long, device=state.x_current.device)
        self.offset = 0.0
        self.rest = self.C(state.x_current[:, self.point_idx]).detach()   # raw measure at rest

    def C(self, pts: torch.Tensor) -> torch.Tensor:
        """Raw measure. pts (B, k, 3) -> (B, 1). Override."""
        raise NotImplementedError("Override C().")

    def residual(self, pts: torch.Tensor, rest: torch.Tensor) -> torch.Tensor:
        """Value driven to zero. Default C - rest; override for angle wrap etc."""
        return self.C(pts) - rest

    @property
    def shape(self):
        return (1,)

    def _get_points(self):
        return self.point_idx

    def get_residual(self, lam: torch.Tensor):
        pts = self.state.x_current[:, self.point_idx]
        return self.residual(pts, self.rest).detach() + self.a_tilde * lam

    def _apply(self, lam, dt, **kwargs):
        x = self.state.x_current                              # (B, N, 3)
        w = self.state.inv_mass[:, self.point_idx]            # (B, k) inverse masses

        pts = x[:, self.point_idx].detach().requires_grad_(True)   # (B, k, 3)
        C = self.residual(pts, self.rest)                    # (B, 1)
        (grad,) = torch.autograd.grad(C.sum(), pts)          # (B, k, 3) = dC/dx_i

        C = C.detach()
        g2 = (grad * grad).sum(-1)                            # (B, k) |dC/dx_i|^2
        w_sum = (w * g2).sum(-1, keepdim=True)               # (B, 1)
        d_lambda = - (C + self.a_tilde * lam) / (w_sum + self.a_tilde).clamp_min(_EPS)  # (B, 1)

        # dx_i = w_i * grad_i * d_lambda; index_add_ so repeated indices accumulate.
        x.index_add_(1, self.idx_t, w[..., None] * grad * d_lambda[:, None])
        return lam + d_lambda


class VectorizedAutogradConstraint(Constraint):
    """Batched `AutogradConstraint`: m constraints of the same type, k points each.

    C : pts (B, m, k, 3) -> (B, m, 1); `residual` (default C - rest) is what's driven
    to zero. One autograd pass yields all m gradients, scattered with index_add_; the
    per-point valence under-relaxation keeps shared-point (merged Jacobi) blocks
    stable and reduces to exact Gauss-Seidel for a point-disjoint block.
    """

    constraint_types = ["measured", "simulated"]

    def __init__(self, state: State, point_idx, alpha: float):
        super().__init__(state, alpha)
        idx = torch.as_tensor(point_idx, dtype=torch.long, device=state.x_current.device)
        assert idx.dim() == 2, "point_idx must be (m, k)"
        self.idx = idx                  # (m, k)
        self.m, self.k = idx.shape
        # per-point valence under-relaxation: total incidence across all m*k slots.
        N = state.x_current.shape[1]
        counts = torch.bincount(idx.reshape(-1), minlength=N).clamp_min(1)   # (N,)
        self.scale = (1.0 / counts[idx]).view(1, self.m, self.k, 1)          # (1, m, k, 1)
        self.offset = 0.0
        self.rest = self.C(state.x_current[:, self.idx]).detach()            # raw measure at rest

    def C(self, pts: torch.Tensor) -> torch.Tensor:
        """Raw measure. pts (B, m, k, 3) -> (B, m, 1). Override."""
        raise NotImplementedError("Override C().")

    def residual(self, pts: torch.Tensor, rest: torch.Tensor) -> torch.Tensor:
        """Value driven to zero. Default C - rest; override for angle wrap etc."""
        return self.C(pts) - rest

    @property
    def shape(self):
        return (self.m, 1)

    def _get_points(self):
        return self.idx.reshape(-1).tolist()

    def get_residual(self, lam: torch.Tensor):
        pts = self.state.x_current[:, self.idx]      # (B, m, k, 3)
        return self.residual(pts, self.rest).detach() + self.a_tilde * lam

    def _apply(self, lam, dt, **kwargs):
        x = self.state.x_current                     # (B, N, 3)
        B = x.shape[0]
        w = self.state.inv_mass[:, self.idx]         # (B, m, k)

        pts = x[:, self.idx].detach().requires_grad_(True)   # (B, m, k, 3)
        C = self.residual(pts, self.rest)            # (B, m, 1)
        (grad,) = torch.autograd.grad(C.sum(), pts)  # (B, m, k, 3) = dC/dx per constraint

        C = C.detach()
        g2 = (grad * grad).sum(-1)                   # (B, m, k)
        w_sum = (w * g2).sum(-1, keepdim=True)       # (B, m, 1)
        d_lambda = - (C + self.a_tilde * lam) / (w_sum + self.a_tilde).clamp_min(_EPS)  # (B, m, 1)

        # dx = w * grad * d_lambda, valence-scaled, scattered in one index_add_
        # (idx may repeat across constraints/slots -> accumulate).
        upd = w[..., None] * grad * d_lambda[:, :, None] * self.scale         # (B, m, k, 3)
        x.index_add_(1, self.idx.reshape(-1), upd.reshape(B, self.m * self.k, 3))
        return lam + d_lambda
