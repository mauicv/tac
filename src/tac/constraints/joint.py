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
        return C + self.a_tilde * lam

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

        w = self.state.inv_mass     # (B, N) inverse masses
        x = self.state.x_current    # (B, N, 3)
        current_theta, u, v, n = self._compute_joint_state(self.state)
        C = current_theta - self.theta + self.offset
        # C is an angle: wrap to (-pi, pi] so the constraint always takes the
        # shortest way round. Without this, crossing the atan2 branch cut (e.g.
        # a straight-bar rest pose sits at +/-pi) makes C jump by ~2*pi and the
        # solver violently "corrects" a spurious full turn.
        C = torch.atan2(torch.sin(C), torch.cos(C))
        g_from = -torch.linalg.cross(n, u) / (u * u).sum(-1, keepdim=True).clamp_min(_EPS)  # (B, 3)
        g_to   =  torch.linalg.cross(n, v) / (v * v).sum(-1, keepdim=True).clamp_min(_EPS)  # (B, 3)

        gf2 = (g_from * g_from).sum(-1, keepdim=True)   # |g_from|² (B, 1)
        gt2 = (g_to   * g_to).sum(-1, keepdim=True)
        w_from = w[:, [self.from_point]]   # (B, 1)
        w_to   = w[:, [self.to_point]]     # (B, 1)
        w_sum = w_from * gf2 + w_to * gt2  # (B, 1)

        residual = C + self.a_tilde * lam
        d_lambda = - residual / (w_sum + self.a_tilde).clamp_min(_EPS)  # (B, 1)

        from_update = w_from * g_from * d_lambda   # (B, 3)
        to_update   = w_to   * g_to   * d_lambda   # (B, 3)

        # The reaction is split across the two axis points so the net displacement
        # over the four points is zero (an internal constraint must not translate
        # the system). Without the /2 each axis point takes the full reaction and
        # the joint injects spurious momentum.
        sum_update = (from_update + to_update) / 2   # (B, 3)

        x[:, self.from_point] += from_update
        x[:, self.to_point]   += to_update
        x[:, self.axis_points] += -sum_update[:, None, :]

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
        return C + self.a_tilde * lam

    def _apply(self, lam, dt, **kwargs):
        assert lam.shape == (self.state.batch_size, len(self.from_idx), 1), \
            "Lambda must be a tensor of shape (b, m, 1)"
        offset = kwargs.get("offset", None)
        if offset is not None:
            self.offset = offset   # broadcastable to (B, m, 1)

        w = self.state.inv_mass     # (B, N)
        current_theta, u, v, n = self._compute_joint_state(self.state)   # theta (B,m,1), u/v/n (B,m,3)
        C = current_theta - self.theta + self.offset                     # (B, m, 1)
        # wrap to (-pi, pi] so the constraint always takes the shortest way round
        # (see Joint.apply for why the atan2 branch cut would otherwise spin it).
        C = torch.atan2(torch.sin(C), torch.cos(C))
        g_from = -torch.linalg.cross(n, u) / (u * u).sum(-1, keepdim=True).clamp_min(_EPS)  # (B, m, 3)
        g_to   =  torch.linalg.cross(n, v) / (v * v).sum(-1, keepdim=True).clamp_min(_EPS)  # (B, m, 3)

        gf2 = (g_from * g_from).sum(-1, keepdim=True)   # |g_from|² (B, m, 1)
        gt2 = (g_to   * g_to).sum(-1, keepdim=True)
        w_from = w[:, self.from_idx, None]   # (B, m, 1)
        w_to   = w[:, self.to_idx, None]     # (B, m, 1)
        w_sum = w_from * gf2 + w_to * gt2    # (B, m, 1)

        residual = C + self.a_tilde * lam
        d_lambda = - residual / (w_sum + self.a_tilde).clamp_min(_EPS)  # (B, m, 1)

        from_update = w_from * g_from * d_lambda   # (B, m, 3)
        to_update   = w_to   * g_to   * d_lambda   # (B, m, 3)
        # reaction split across the two axis points -> zero net displacement per
        # joint (see Joint.apply); otherwise each axis point takes the full reaction
        # and the joint injects spurious momentum.
        sum_update  = (from_update + to_update) / 2   # (B, m, 3)

        # index_add_ accumulates duplicate indices (joints may share points); the
        # reaction -sum_update is applied to both axis points, as in Joint.apply.
        # *_scale applies the per-point valence under-relaxation (averaged Jacobi).
        xc = self.state.x_current
        xc.index_add_(1, self.from_t,  from_update * self.from_scale)
        xc.index_add_(1, self.to_t,    to_update * self.to_scale)
        xc.index_add_(1, self.axis0_t, -sum_update * self.axis0_scale)
        xc.index_add_(1, self.axis1_t, -sum_update * self.axis1_scale)

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