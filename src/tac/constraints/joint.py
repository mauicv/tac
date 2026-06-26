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
            from_points: List[int],
            to_points: List[int],
            alpha: float,
            dt: float,
        ):
        assert len(axis_points) == 2, "Axis points must be a pair"
        super().__init__(state, alpha, dt)
        self.axis_points = axis_points
        self.from_points = from_points
        self.to_points = to_points
        theta, *_ = self._compute_joint_state(state)
        self.theta = theta
        self.offset = 0.0

    def to_dict(self):
        return {
            "class": "Joint",
            "params": {
                "axis_points": self.axis_points,
                "from_points": self.from_points,
                "to_points": self.to_points,
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
        c_from = torch.mean(x[:, self.from_points], dim=1)
        c_to   = torch.mean(x[:, self.to_points], dim=1)
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
        g_from = -torch.linalg.cross(n, u) / (u * u).sum(-1, keepdim=True).clamp_min(_EPS)
        g_to   =  torch.linalg.cross(n, v) / (v * v).sum(-1, keepdim=True).clamp_min(_EPS)

        n_from, n_to = len(self.from_points), len(self.to_points)

        gf2 = (g_from * g_from).sum(-1, keepdim=True)   # |g_from|² (B, 1)
        gt2 = (g_to   * g_to).sum(-1, keepdim=True)
        w_sum = (w[:, self.from_points].sum(-1, keepdim=True) * gf2 / n_from**2
                + w[:, self.to_points].sum(-1, keepdim=True) * gt2 / n_to**2)

        d_lambda = (-C - self.a_tilde * lam) / (w_sum + self.a_tilde).clamp_min(_EPS)  # (B, 1)

        from_update = (w[:, self.from_points, None] * (g_from[:, None, :] / n_from) * d_lambda[:, None])
        to_update = (w[:, self.to_points, None] * (g_to[:, None, :] / n_to) * d_lambda[:, None])

        # TODO: is this correct?
        sum_update = from_update.mean(dim=1, keepdim=True) + to_update.mean(dim=1, keepdim=True)

        x[:, self.from_points] += from_update
        x[:, self.to_points]   += to_update
        x[:, self.axis_points] += -sum_update

        return lam + d_lambda