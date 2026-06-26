from tac.constraints.util import _normalize, _perp, _cross, _EPS
from tac.core.state import State
from typing import List
from tac.constraints.base import Constraint


import torch

def pitch_roll_2_vect(pitch, roll):
    g = torch.stack([
        -torch.sin(pitch),
         torch.cos(pitch) * torch.sin(roll),
         torch.cos(pitch) * torch.cos(roll),
    ], dim=-1)
    return g


class PitchRollIMU(Constraint):
    constraint_types = ["measured"]

    def __init__(
            self,
            state: State,
            imu_points: List[int],
            alpha: float,
            dt: float
        ):
        assert len(imu_points) == 2, "There must be exactly two IMU points"
        super().__init__(state, alpha, dt)
        self.imu_points = imu_points

        # for logging:
        self.f = None
        self.u = None
        self.v = None

    def to_dict(self):
        return {
            "class": "PitchRollIMU",
            "params": {
                "imu_points": self.imu_points,
                "alpha": self.alpha,
            },
        }

    @property
    def shape(self):
        return (1, )

    def get_imu_state(self):
        v = self.state.x_current[:, self.imu_points[1]] - self.state.x_current[:, self.imu_points[0]] # (B, 2, 3)
        return _normalize(v) # (B, 1, 3)

    def apply(self, lam, dt, **kwargs):
        pitch = kwargs.get("pitch", None)
        roll = kwargs.get("roll", None)
        if pitch is None or roll is None: return lam

        # compute down vector from pitch/roll, and the body IMU axis u = head - tail.
        v = -pitch_roll_2_vect(pitch, roll)                                                          # world target (B, 3)
        tail, head = self.imu_points[0], self.imu_points[1]
        u_raw = self.state.x_current[:, head] - self.state.x_current[:, tail]                       # (B, 3)
        L2 = (u_raw * u_raw).sum(-1, keepdim=True).clamp_min(_EPS)                                  # |u|² (B, 1)
        u = _normalize(u_raw)                                                                       # (B, 3)
        n = _cross(v, u)

        # if n is close to zero (u ∥ v) the rotation axis is undefined — skip.
        sin_t = torch.linalg.vector_norm(n, dim=-1)
        if sin_t > _EPS:
            cos_t = (u * v).sum(-1, keepdim=True)              # u·v = cosθ                         # (B, 1)
            C = torch.atan2(sin_t[..., None], cos_t)           # angle θ ∈ [0, π]                   # (B, 1)

            n = _normalize(n)
            # dC = ∂θ/∂(head): tangential direction, scaled by the real lever arm |u|.
            dC = _cross(n, u) / L2                                                                  # (B, 3)

            w_head = self.state.inv_mass[:, [head]]                                                 # (B, 1)
            w_tail = self.state.inv_mass[:, [tail]]                                                 # (B, 1)
            dC2 = (dC * dC).sum(-1, keepdim=True)                                                   # |dC|² (B, 1)
            w_sum = (w_head + w_tail) * dC2                                                         # (B, 1)

            d_lambda = (-C - self.a_tilde * lam) / (w_sum + self.a_tilde).clamp_min(_EPS)             # (B, 1)
            # Apply as an equal-and-opposite couple on the two IMU points: ∂θ/∂head = +dC,
            # ∂θ/∂tail = -dC. This rotates the body axis; weld rigidity carries the rest.
            self.state.x_current[:, head] +=  w_head * dC * d_lambda                                # (B, 3)
            self.state.x_current[:, tail] += -w_tail * dC * d_lambda                                # (B, 3)
        else:
            dC = torch.zeros_like(u)
            d_lambda = torch.zeros_like(lam)


        # for logging:
        self.u = u
        self.v = v
        self.f = dC

        return lam + d_lambda