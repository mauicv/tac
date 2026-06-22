"""Length (distance) constraint — the atom welds are built from.

    C(x) = |x_a - x_b| - L
    grad_a C = n = (x_a - x_b)/|x_a - x_b|      grad_b C = -n

Both gradients are unit vectors, so |grad|^2 = 1 at each endpoint and the XPBD
denominator is simply w_a + w_b + alpha/dt^2.
"""

from __future__ import annotations

import torch

from ..core.state import State
from .base import Constraint

Tensor = torch.Tensor

_EPS = 1e-9


class Length(Constraint):
    def __init__(self, a: int, b: int, rest_length, alpha: float = 0.0):
        self.a = int(a)
        self.b = int(b)
        self.rest_length = rest_length  # float or tensor (learnable L)
        self.alpha = float(alpha)

    def project(self, state: State, lam: Tensor, dt: float) -> tuple[State, Tensor]:
        x = state.x_current
        d = x[:, self.a] - x[:, self.b]                       # (B, 3)
        dist = torch.linalg.vector_norm(d, dim=-1, keepdim=True)  # (B, 1)
        n = d / dist.clamp_min(_EPS)                          # (B, 3) unit direction

        L = torch.as_tensor(self.rest_length, dtype=x.dtype, device=x.device)
        C = dist.squeeze(-1) - L                              # (B,)

        wa = self._w(state, self.a)                           # (B,)
        wb = self._w(state, self.b)
        a_tilde = self.alpha / (dt * dt)
        denom = wa + wb + a_tilde
        # Both endpoints pinned (and hard) -> denom 0; nothing can move, so no correction.
        dlam = torch.where(
            denom > _EPS, (-C - a_tilde * lam) / denom.clamp_min(_EPS), torch.zeros_like(C)
        )
        lam = lam + dlam

        corr = dlam.unsqueeze(-1) * n                         # (B, 3) = dlam * grad dir
        delta = torch.zeros_like(x)
        delta[:, self.a] = wa.unsqueeze(-1) * corr            # x_a += w_a * dlam * n
        delta[:, self.b] = -wb.unsqueeze(-1) * corr           # x_b += w_b * dlam * (-n)

        new_state = State(
            x_current=x + delta,
            x_previous=state.x_previous,
            inv_mass=state.inv_mass,
        )
        return new_state, lam

    @staticmethod
    def _w(state: State, idx: int) -> Tensor:
        """Inverse mass of point `idx` as a (B,) tensor (handles (N,) or (B, N))."""
        im = state.inv_mass
        if im.ndim == 1:
            return im[idx].expand(state.batch)
        return im[:, idx]
