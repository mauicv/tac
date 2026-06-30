from typing import List

import torch

from tac.core.state import State
from tac.constraints.autograd import AutogradConstraint, VectorizedAutogradConstraint


def _dist(pts: torch.Tensor) -> torch.Tensor:
    """Distance between the two endpoints. pts (..., 2, 3) -> (..., 1)."""
    return torch.linalg.vector_norm(pts[..., 0, :] - pts[..., 1, :], dim=-1, keepdim=True)


class Length(AutogradConstraint):
    """Distance constraint between two points (holds |x_i - x_j| at its rest value)."""

    constraint_types = ["simulated"]

    def __init__(self, state: State, i: int, j: int, alpha: float):
        self.i = i
        self.j = j
        super().__init__(state, [i, j], alpha)   # base captures rest = |x_i - x_j| at init
        self.L = self.rest[[0]]                   # (1, 1) rest length (back-compat alias)

    def C(self, pts: torch.Tensor) -> torch.Tensor:
        return _dist(pts)   # raw measure; default residual = C - rest

    def get_length(self):
        return self.C(self.state.x_current[:, self.point_idx])   # (B, 1) current length

    def to_dict(self):
        return {"class": "Length", "params": {"i": self.i, "j": self.j, "alpha": self.alpha}}


class VectorizedLength(VectorizedAutogradConstraint):
    """Batched distance constraint: m length constraints solved together (k=2)."""

    constraint_types = ["simulated"]

    def __init__(self, state: State, i_idx: List[int], j_idx: List[int], alpha: float):
        assert len(i_idx) == len(j_idx), "i_idx and j_idx must have the same length"
        self.i_idx = list(i_idx)
        self.j_idx = list(j_idx)
        idx = [[i, j] for i, j in zip(i_idx, j_idx)]   # (m, 2)
        super().__init__(state, idx, alpha)
        self.L = self.rest[[0]]                         # (1, m, 1) rest lengths (back-compat alias)

    @classmethod
    def from_constraints(cls, state: State, lengths: List[Length], alpha: float):
        return cls(state, [length.i for length in lengths], [length.j for length in lengths], alpha)

    def C(self, pts: torch.Tensor) -> torch.Tensor:
        return _dist(pts)   # pts (B, m, 2, 3) -> (B, m, 1)

    def get_length(self):
        return self.C(self.state.x_current[:, self.idx])   # (B, m, 1) current lengths

    def to_dict(self):
        return {
            "class": "VectorizedLength",
            "params": {"i_idx": self.i_idx, "j_idx": self.j_idx, "alpha": self.alpha},
        }
