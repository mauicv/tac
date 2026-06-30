from typing import List

import torch

from tac.core.state import State
from tac.constraints.autograd import AutogradConstraint, VectorizedAutogradConstraint


def _planar_measure(pts: torch.Tensor) -> torch.Tensor:
    if pts.shape[-2] == 4:
        a, b, c, d = pts[..., 0, :], pts[..., 1, :], pts[..., 2, :], pts[..., 3, :]
        return ((b - a) * torch.linalg.cross(c - a, d - a, dim=-1)).sum(-1, keepdim=True)
    centred = pts - pts.mean(-2, keepdim=True)                      # (..., k, 3)
    cov = torch.einsum("...ki,...kj->...ij", centred, centred)      # (..., 3, 3)
    return torch.linalg.eigvalsh(cov)[..., :1]                      # smallest eigenvalue (..., 1)


class Planar(AutogradConstraint):
    def __init__(self, state: State, point_idx: List[int], alpha: float):
        assert len(point_idx) >= 4, "Planar needs >=4 points (3 are always coplanar)"
        super().__init__(state, point_idx, alpha)   # base captures rest = measure at init

    def C(self, pts: torch.Tensor) -> torch.Tensor:
        return _planar_measure(pts)   # raw measure; default residual = C - rest

    def to_dict(self):
        return {"class": "Planar", "params": {"point_idx": self.point_idx, "alpha": self.alpha}}


class VectorizedPlanar(VectorizedAutogradConstraint):
    def __init__(self, state: State, point_idx, alpha: float):
        super().__init__(state, point_idx, alpha)   # base captures rest = measure at init
        assert self.k >= 4, "Planar needs >=4 points (3 are always coplanar)"

    def C(self, pts: torch.Tensor) -> torch.Tensor:
        return _planar_measure(pts)   # raw measure; default residual = C - rest

    @classmethod
    def from_constraints(cls, state: State, planars: List[Planar], alpha: float):
        return cls(state, [p.point_idx for p in planars], alpha)

    def to_dict(self):
        return {
            "class": "VectorizedPlanar",
            "params": {"point_idx": self.idx.tolist(), "alpha": self.alpha},
        }
