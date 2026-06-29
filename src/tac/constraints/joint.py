from typing import List
from tac.core.state import State
import torch
from tac.constraints.util import _normalize, _perp, _EPS
from tac.constraints.base import Constraint


class Joint(Constraint):
    constraint_types = ["measured", "simulated"] 

    def __init__(
            self,
            state: State,
            axis_points: List[int],
            from_point: int,
            to_point: int,
            alpha: float,
        ):
        assert len(axis_points) == 2, "Axis points must be a pair"
        super().__init__(state, alpha)
        self.axis_points = axis_points
        self.from_point = from_point
        self.to_point = to_point
        theta, *_ = self._compute_joint_state(state)
        self.theta = theta
        self.offset = 0.0

    def get_theta(self):
        theta, u, v, n = self._compute_joint_state(self.state)
        return theta

    def get_residual(self, lam: torch.Tensor):
        C = self.get_theta() - self.theta + self.offset
        C = torch.atan2(torch.sin(C), torch.cos(C))
        r = C + self.a_tilde * lam
        return r

    @staticmethod
    def _theta_of(p, q, c_from, c_to):
        """Hinge angle of from/to about the p->q axis. Differentiable in all four
        point positions (p, q, c_from, c_to each (..., 3))."""
        n = _normalize(q - p)
        u = _perp(c_from - p, n)
        v = _perp(c_to - p, n)
        cross = (n * torch.linalg.cross(u, v)).sum(-1, keepdim=True)
        dot = (u * v).sum(-1, keepdim=True)
        return torch.atan2(cross, dot)   # (..., 1)

    def _compute_joint_state(self, state: State) -> torch.Tensor:
        x = state.x_current
        p, q = x[:, self.axis_points[0]], x[:, self.axis_points[1]]
        n = _normalize(q - p)
        c_from = x[:, self.from_point]   # (B, 3)
        c_to   = x[:, self.to_point]     # (B, 3)
        u = _perp(c_from - p, n)
        v = _perp(c_to - p, n)
        cross = (n * torch.linalg.cross(u, v)).sum(-1, keepdim=True)
        dot = (u * v).sum(-1, keepdim=True)
        theta = torch.atan2(cross, dot)
        return theta, u, v, n  # theta (B, 1), u/v/n (B, 3)

    def _apply(self, lam, dt, **kwargs):
        offset = kwargs.get("offset", None)
        if offset is not None:
            self.offset = offset

        # order: [axis0, axis1, from, to] — all four points the angle depends on
        idx = [self.axis_points[0], self.axis_points[1], self.from_point, self.to_point]
        x = self.state.x_current        # (B, N, 3)
        w = self.state.inv_mass[:, idx]  # (B, 4) inverse masses

        # True constraint gradient: dtheta/dx_i for every participating point, via
        # autograd. theta is invariant to translating the four points together, so
        # sum_i dtheta/dx_i == 0 exactly -> linear momentum is conserved with no
        # manual reaction, and (being a genuine gradient) the projection is
        # conservative: it injects no spurious energy. This replaces the old
        # lever-arm gradients + hand-rolled axis reaction, which were not the
        # gradient of any scalar and pumped energy whenever the mechanism moved.
        pts = x[:, idx].detach().requires_grad_(True)        # (B, 4, 3)
        theta = self._theta_of(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3])  # (B, 1)
        (grad,) = torch.autograd.grad(theta.sum(), pts)      # (B, 4, 3)

        C = theta.detach() - self.theta + self.offset
        # wrap to (-pi, pi] so we always take the shortest way round the atan2 cut.
        C = torch.atan2(torch.sin(C), torch.cos(C))          # (B, 1)

        g2 = (grad * grad).sum(-1)                           # (B, 4) |grad_i|²
        w_sum = (w * g2).sum(-1, keepdim=True)               # (B, 1)
        d_lambda = - (C + self.a_tilde * lam) / (w_sum + self.a_tilde).clamp_min(_EPS)  # (B, 1)

        # XPBD update: dx_i = w_i * grad_i * d_lambda
        x[:, idx] += w[..., None] * grad * d_lambda[:, None]  # (B, 4, 3)
        return lam + d_lambda

    def _get_points(self):
        return [self.from_point, self.to_point, self.axis_points[0], self.axis_points[1]]

    @property
    def shape(self):
        return (1, )

    def to_dict(self):
        return {
            "class": "Joint",
            "params": {
                "axis_points": self.axis_points,
                "from_point": self.from_point,
                "to_point": self.to_point,
                "alpha": self.alpha,
            },
        }


class VectorizedJoint(Constraint):
    constraint_types = ["measured", "simulated"]

    def __init__(
            self,
            state: State,
            axis_points: List[List[int]],   # list of m hinge edges, each [a0, a1]
            from_idx: List[int],            # m from-points
            to_idx: List[int],              # m to-points
            alpha: float,
        ):
        assert len(from_idx) == len(to_idx) == len(axis_points), \
            "axis_points, from_idx and to_idx must have the same length"
        assert all(len(ap) == 2 for ap in axis_points), "Each axis must be a pair of points"
        super().__init__(state, alpha)
        self.axis_points = axis_points
        self.from_idx = list(from_idx)
        self.to_idx = list(to_idx)
        self.axis0_idx = [ap[0] for ap in axis_points]
        self.axis1_idx = [ap[1] for ap in axis_points]
        # long index tensors for index_add_ scatter (cached to avoid per-step rebuilds)
        dev = state.x_current.device
        self.from_t  = torch.as_tensor(self.from_idx,  dtype=torch.long, device=dev)
        self.to_t    = torch.as_tensor(self.to_idx,    dtype=torch.long, device=dev)
        self.axis0_t = torch.as_tensor(self.axis0_idx, dtype=torch.long, device=dev)
        self.axis1_t = torch.as_tensor(self.axis1_idx, dtype=torch.long, device=dev)
        # Per-point under-relaxation: a point shared by k joints in this block gets
        # k corrections summed by index_add_ (Jacobi overshoot). Pre-scale each update
        # by 1/valence so the scattered sum averages instead of piling up. count is the
        # TOTAL incidence across all four scatter arrays (from/to/axis0/axis1); a
        # point-disjoint block has valence 1 everywhere -> exact Gauss-Seidel.
        N = state.x_current.shape[1]
        counts = torch.bincount(
            torch.cat([self.from_t, self.to_t, self.axis0_t, self.axis1_t]), minlength=N
        ).clamp_min(1)   # (N,)
        self.from_scale  = (1.0 / counts[self.from_t]).view(1, -1, 1)    # (1, m, 1)
        self.to_scale    = (1.0 / counts[self.to_t]).view(1, -1, 1)
        self.axis0_scale = (1.0 / counts[self.axis0_t]).view(1, -1, 1)
        self.axis1_scale = (1.0 / counts[self.axis1_t]).view(1, -1, 1)
        theta, *_ = self._compute_joint_state(state)
        self.theta = theta[[0]]   # (1, m, 1) rest angle, broadcasts over batch
        self.offset = 0.0

    @classmethod
    def from_constraints(cls, state: State, joints: List[Joint], alpha: float):
        return cls(
            state,
            [joint.axis_points for joint in joints],
            [joint.from_point for joint in joints],
            [joint.to_point for joint in joints],
            alpha
        )

    def _compute_joint_state(self, state: State):
        x = state.x_current
        p = x[:, self.axis0_idx]   # (B, m, 3)
        q = x[:, self.axis1_idx]   # (B, m, 3)
        n = _normalize(q - p)
        c_from = x[:, self.from_idx]   # (B, m, 3)
        c_to   = x[:, self.to_idx]     # (B, m, 3)
        u = _perp(c_from - p, n)
        v = _perp(c_to - p, n)
        cross = (n * torch.linalg.cross(u, v)).sum(-1, keepdim=True)   # (B, m, 1)
        dot   = (u * v).sum(-1, keepdim=True)                          # (B, m, 1)
        theta = torch.atan2(cross, dot)
        return theta, u, v, n   # theta (B, m, 1), u/v/n (B, m, 3)

    def get_theta(self):
        theta, *_ = self._compute_joint_state(self.state)
        return theta   # (B, m, 1)

    def get_residual(self, lam: torch.Tensor):
        C = self.get_theta() - self.theta + self.offset
        C = torch.atan2(torch.sin(C), torch.cos(C))
        r = C + self.a_tilde * lam
        return r

    def _apply(self, lam, dt, **kwargs):
        assert lam.shape == (self.state.batch_size, len(self.from_idx), 1), \
            "Lambda must be a tensor of shape (b, m, 1)"
        offset = kwargs.get("offset", None)
        if offset is not None:
            self.offset = offset   # broadcastable to (B, m, 1)

        x = self.state.x_current    # (B, N, 3)
        w = self.state.inv_mass     # (B, N)

        # True constraint gradient per joint via autograd (see Joint._apply). theta
        # is translation-invariant in its four points, so sum_i dtheta/dx_i == 0 ->
        # momentum-conserving and conservative (no energy injection). Gather the four
        # points of every joint as one leaf: pts (B, m, 4, 3) = [axis0, axis1, from, to].
        pts = torch.stack([
            x[:, self.axis0_idx], x[:, self.axis1_idx],
            x[:, self.from_idx],  x[:, self.to_idx],
        ], dim=2).detach().requires_grad_(True)              # (B, m, 4, 3)
        theta = Joint._theta_of(pts[:, :, 0], pts[:, :, 1], pts[:, :, 2], pts[:, :, 3])  # (B, m, 1)
        (grad,) = torch.autograd.grad(theta.sum(), pts)      # (B, m, 4, 3)

        C = theta.detach() - self.theta + self.offset        # (B, m, 1)
        # wrap to (-pi, pi] so we always take the shortest way round the atan2 cut.
        C = torch.atan2(torch.sin(C), torch.cos(C))

        w4 = torch.stack([
            w[:, self.axis0_idx], w[:, self.axis1_idx],
            w[:, self.from_idx],  w[:, self.to_idx],
        ], dim=2)                                            # (B, m, 4) inverse masses
        g2 = (grad * grad).sum(-1)                           # (B, m, 4) |grad_i|²
        w_sum = (w4 * g2).sum(-1, keepdim=True)              # (B, m, 1)
        d_lambda = - (C + self.a_tilde * lam) / (w_sum + self.a_tilde).clamp_min(_EPS)  # (B, m, 1)

        # dx_i = w_i * grad_i * d_lambda; index_add_ accumulates shared points and
        # *_scale applies the per-point valence under-relaxation (averaged Jacobi).
        upd = w4[..., None] * grad * d_lambda[:, :, None]    # (B, m, 4, 3)
        x.index_add_(1, self.axis0_t, upd[:, :, 0] * self.axis0_scale)
        x.index_add_(1, self.axis1_t, upd[:, :, 1] * self.axis1_scale)
        x.index_add_(1, self.from_t,  upd[:, :, 2] * self.from_scale)
        x.index_add_(1, self.to_t,    upd[:, :, 3] * self.to_scale)

        return lam + d_lambda
    
    @property
    def shape(self):
        return (len(self.from_idx), 1)

    def to_dict(self):
        return {
            "class": "VectorizedJoint",
            "params": {
                "axis_points": self.axis_points,
                "from_idx": self.from_idx,
                "to_idx": self.to_idx,
                "alpha": self.alpha,
            },
        }