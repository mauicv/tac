from typing import List

import torch

from tac.core.state import State
from tac.constraints.util import _normalize, _perp
from tac.constraints.autograd import AutogradConstraint, VectorizedAutogradConstraint


def _joint_theta(pts: torch.Tensor) -> torch.Tensor:
    """Hinge angle of from/to about the axis, differentiable in all four points.

    pts (..., 4, 3) in order [axis0, axis1, from, to] -> (..., 1).
    """
    p, q, cf, ct = pts[..., 0, :], pts[..., 1, :], pts[..., 2, :], pts[..., 3, :]
    n = _normalize(q - p)
    u = _perp(cf - p, n)
    v = _perp(ct - p, n)
    cross = (n * torch.linalg.cross(u, v, dim=-1)).sum(-1, keepdim=True)
    dot = (u * v).sum(-1, keepdim=True)
    return torch.atan2(cross, dot)


class Joint(AutogradConstraint):
    """Revolute joint: holds the angle of `from`/`to` about the `axis` edge.

    Defined entirely by C = wrap(theta(pts) - rest + offset); the autograd base
    supplies dC/dx. Because theta is translation-invariant, sum_i dC/dx_i == 0, so
    the projection conserves momentum and is conservative (no energy injection).
    """

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
        self.axis_points = axis_points
        self.from_point = from_point
        self.to_point = to_point
        # point order is the contract for _joint_theta: [axis0, axis1, from, to].
        super().__init__(state, [axis_points[0], axis_points[1], from_point, to_point], alpha)

    def C(self, pts: torch.Tensor) -> torch.Tensor:
        return _joint_theta(pts)   # raw hinge angle (..., 1); base captures rest = angle at init

    def residual(self, pts: torch.Tensor, rest: torch.Tensor) -> torch.Tensor:
        c = self.C(pts) - rest + self.offset
        return torch.atan2(torch.sin(c), torch.cos(c))   # wrap to (-pi, pi]

    def get_theta(self):
        return _joint_theta(self.state.x_current[:, self.point_idx])   # (B, 1)

    def _apply(self, lam, dt, **kwargs):
        offset = kwargs.get("offset", None)
        if offset is not None:
            self.offset = offset
        return super()._apply(lam, dt, **kwargs)

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


class VectorizedJoint(VectorizedAutogradConstraint):
    """Batched Joint: m revolute joints solved together (k=4 points each)."""

    constraint_types = ["measured", "simulated"]

    def __init__(
        self,
        state: State,
        axis_points: List[List[int]],   # m hinge edges, each [a0, a1]
        from_idx: List[int],            # m from-points
        to_idx: List[int],              # m to-points
        alpha: float,
    ):
        assert len(from_idx) == len(to_idx) == len(axis_points), \
            "axis_points, from_idx and to_idx must have the same length"
        assert all(len(ap) == 2 for ap in axis_points), "Each axis must be a pair of points"
        self.axis_points = axis_points
        self.from_idx = list(from_idx)
        self.to_idx = list(to_idx)
        # (m, 4) in order [axis0, axis1, from, to] per joint
        idx = [[ap[0], ap[1], f, t] for ap, f, t in zip(axis_points, from_idx, to_idx)]
        super().__init__(state, idx, alpha)

    def C(self, pts: torch.Tensor) -> torch.Tensor:
        return _joint_theta(pts)   # raw hinge angle (B, m, 1); base captures rest at init

    def residual(self, pts: torch.Tensor, rest: torch.Tensor) -> torch.Tensor:
        c = self.C(pts) - rest + self.offset
        return torch.atan2(torch.sin(c), torch.cos(c))

    def get_theta(self):
        return _joint_theta(self.state.x_current[:, self.idx])   # (B, m, 1)

    def _apply(self, lam, dt, **kwargs):
        offset = kwargs.get("offset", None)
        if offset is not None:
            self.offset = offset   # broadcastable to (B, m, 1)
        return super()._apply(lam, dt, **kwargs)

    @classmethod
    def from_constraints(cls, state: State, joints: List[Joint], alpha: float):
        return cls(
            state,
            [joint.axis_points for joint in joints],
            [joint.from_point for joint in joints],
            [joint.to_point for joint in joints],
            alpha,
        )

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
