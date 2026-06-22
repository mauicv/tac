"""Anchor — pin a point to a fixed world position (a fixed root).

It is just a `length` constraint to a *fixed world point* with rest length 0:

    C(x) = |x_p - target|            grad_p C = n = (x_p - target)/|x_p - target|

so |grad|^2 = 1 and the XPBD denominator is w_p + alpha/dt^2 (the target is fixed, i.e.
infinite mass, so it contributes nothing). With alpha = 0 a free point lands exactly on
the target in one iteration: correction = w_p * (-C/w_p) * n = -(x_p - target). The arm
anchors its base this way; gnoci omits it (anchored by foot contacts).
"""

from __future__ import annotations

import torch

from ..core.state import State
from .base import Constraint

Tensor = torch.Tensor

_EPS = 1e-9


class Anchor(Constraint):
    def __init__(self, point: int, target, alpha: float = 0.0):
        self.point = int(point)
        self.target = target  # (3,) or (B, 3) world position; may be a learnable tensor
        self.alpha = float(alpha)

    def project(self, state: State, lam: Tensor, dt: float) -> tuple[State, Tensor]:
        x = state.x_current
        target = torch.as_tensor(self.target, dtype=x.dtype, device=x.device)
        target = target.broadcast_to(x[:, self.point].shape)  # (B, 3)

        d = x[:, self.point] - target                          # (B, 3)
        dist = torch.linalg.vector_norm(d, dim=-1, keepdim=True)  # (B, 1)
        n = d / dist.clamp_min(_EPS)                           # (B, 3)
        C = dist.squeeze(-1)                                   # (B,) rest length is 0

        wp = self._w(state, self.point)                        # (B,)
        a_tilde = self.alpha / (dt * dt)
        denom = wp + a_tilde
        # Pinned point, hard anchor -> denom 0; the point is already fixed, no correction.
        dlam = torch.where(
            denom > _EPS, (-C - a_tilde * lam) / denom.clamp_min(_EPS), torch.zeros_like(C)
        )
        lam = lam + dlam

        delta = torch.zeros_like(x)
        delta[:, self.point] = (wp.unsqueeze(-1) * dlam.unsqueeze(-1)) * n

        new_state = State(
            x_current=x + delta,
            x_previous=state.x_previous,
            inv_mass=state.inv_mass,
        )
        return new_state, lam

    @staticmethod
    def _w(state: State, idx: int) -> Tensor:
        im = state.inv_mass
        if im.ndim == 1:
            return im[idx].expand(state.batch)
        return im[:, idx]
