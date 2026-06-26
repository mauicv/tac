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
            dt: float,
        ):
        assert len(axis_points) == 2, "Axis points must be a pair"
        super().__init__(state, alpha, dt)
        self.axis_points = axis_points
        self.from_point = from_point
        self.to_point = to_point
        theta, *_ = self._compute_joint_state(state)
        self.theta = theta
        self.offset = 0.0

    def to_dict(self):
        return {
            "class": "Joint",
            "params": {
                "axis_points": self.axis_points,
                "from_point": self.from_point,
                "to_point": self.to_point,
                "alpha": self.alpha,
                "dt": self.dt,
            },
        }

    @property
    def shape(self):
        return (1, )

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

    def apply(self, lam, dt, **kwargs):
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

        d_lambda = (-C - self.a_tilde * lam) / (w_sum + self.a_tilde).clamp_min(_EPS)  # (B, 1)

        from_update = w_from * g_from * d_lambda   # (B, 3)
        to_update   = w_to   * g_to   * d_lambda   # (B, 3)

        # TODO: is this correct? Reaction applied to both axis points.
        sum_update = from_update + to_update       # (B, 3)

        x[:, self.from_point] += from_update
        x[:, self.to_point]   += to_update
        x[:, self.axis_points] += -sum_update[:, None, :]

        return lam + d_lambda