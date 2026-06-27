from tac.core.state import State
from typing import List
import torch
from tac.constraints.base import Constraint


class Length(Constraint):
    constraint_types = ["simulated"]

    def __init__(self, state: State, i: int, j: int, alpha: float):
        super().__init__(state, alpha)
        self.i = i
        self.j = j
        dist = self.get_length()    # shape, (b, 1)
        self.L = dist[[0], :]     # shape, (1, 1)

    def get_length(self):
        x = self.state.x_current[:,self.i] 
        y = self.state.x_current[:,self.j]
        d = x - y
        dist = torch.linalg.vector_norm(d, dim=-1, keepdim=True)
        return dist # shape, (b, 1)

    @property
    def shape(self):
        return (1, )

    def _apply(self, lam, dt, **kwargs):
        assert lam.shape == (self.state.batch_size, 1), "Lambda must be a tensor of shape (b, 1)"
        x,y = self.state.x_current[:, self.i], self.state.x_current[:, self.j]
        delta = x - y                                                                   # shape, (b, 3)
        dist = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)                    # shape, (b, 1)
        n = delta / dist.clamp(min=1e-6)                                                # shape, (b, 3)
        C = dist - self.L                                                               # shape, (b, 1)
        wa, wb = self.state.inv_mass[:, [self.i]], self.state.inv_mass[:, [self.j]]     # shape, (b, 1)
        denum = wa + wb + self.a_tilde                                                       # shape, (b, 1)
        dlam = - (C + self.a_tilde * lam) / denum                                            # shape, (b, 1)
        lam = lam + dlam                                                                # shape, (b, 1)
        x_update = wa * dlam * n
        y_update = - wb * dlam * n
        self.state.x_current[:, self.i] += x_update                                     # shape, (b, 3)
        self.state.x_current[:, self.j] += y_update                                     # shape, (b, 3)
        return lam

    def to_dict(self):
        return {
            "class": "Length",
            "params": {
                "i": self.i,
                "j": self.j,
                "alpha": self.alpha,
            },
        }

    def _get_points(self):
        return [self.i, self.j]


class VectorizedLength(Constraint):
    def __init__(self, state: State, i_idx: List[int], j_idx: List[int], alpha: float):
        assert len(i_idx) == len(j_idx), "i_idx and j_idx must have the same length"
        super().__init__(state, alpha)
        self.i_idx = i_idx
        self.j_idx = j_idx
        # long index tensors for index_add_ scatter (cached to avoid per-step rebuilds)
        self.i_t = torch.as_tensor(i_idx, dtype=torch.long, device=state.x_current.device)
        self.j_t = torch.as_tensor(j_idx, dtype=torch.long, device=state.x_current.device)
        dist = self.get_length()    # shape, (b, m)
        self.L = dist[[0], :]     # shape, (1, m)

    @classmethod
    def from_constraints(cls, state: State, lengths: List[Length], alpha: float):
        return cls(state, [length.i for length in lengths], [length.j for length in lengths], alpha)

    @property
    def shape(self):
        return (len(self.i_idx), 1)

    def get_length(self):
        x = self.state.x_current[:,self.i_idx] 
        y = self.state.x_current[:,self.j_idx]
        dist = torch.linalg.vector_norm(x - y, dim=-1, keepdim=True)
        return dist # shape, (b, m, 1)

    def _apply(self, lam, dt, **kwargs):
        assert lam.shape == (self.state.batch_size, len(self.i_idx), 1), "Lambda must be a tensor of shape (b, m, 1)"
        x = self.state.x_current[:,self.i_idx]                                                      # shape, (b, m, 3)
        y = self.state.x_current[:,self.j_idx]                                                      # shape, (b, m, 3)
        dist = torch.linalg.vector_norm(x - y, dim=-1, keepdim=True)                                # shape, (b, m, 1)
        n = (x - y) / dist.clamp(min=1e-6)
        C = dist - self.L                                                                           # shape, (b, m, 1)
        wa, wb = self.state.inv_mass[:,self.i_idx, None], self.state.inv_mass[:,self.j_idx, None]   # shape, (b, m, 1)
        denum = wa + wb + self.a_tilde                                                                   # shape, (b, m, 1)
        dlam = - (C + self.a_tilde * lam) / denum                                                        # shape, (b, m, 1)
        lam = lam + dlam                                                                            # shape, (b, m, 1)
        x_update = wa * dlam * n                                                                    # shape, (b, m, 3)
        y_update = - wb * dlam * n                                                                  # shape, (b, m, 3)
        # index_add_ accumulates duplicate indices; plain `x[:, idx] += ...` does NOT
        # (last-write-wins), which silently drops most updates when points are shared.
        self.state.x_current.index_add_(1, self.i_t, x_update)                                      # shape, (b, m, 3)
        self.state.x_current.index_add_(1, self.j_t, y_update)                                      # shape, (b, m, 3)
        return lam                                                                                  # shape, (b, m, 1)

    def to_dict(self):
        return {
            "class": "VectorizedLength",
            "params": {
                "i_idx": self.i_idx,
                "j_idx": self.j_idx,
                "alpha": self.alpha,
            },
        }