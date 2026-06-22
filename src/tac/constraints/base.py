"""Constraint interface for the XPBD solver.

A constraint defines a scalar error C(x) it tries to drive to zero, with a compliance
`alpha` (0.0 = hard). `project` performs one XPBD solve pass: it computes the constraint
error, nudges the relevant positions, and returns the (accumulated) Lagrange multiplier
`lam`. The solver threads `lam` through the inner iterations and resets it each timestep.

Scope and order are carried for later phases (modeled vs measured passes, solve ordering);
they are not used yet.
"""

from __future__ import annotations

import torch

from ..core.state import State

Tensor = torch.Tensor


class Constraint:
    alpha: float = 0.0          # compliance; 0.0 = hard
    scope: str = "modeled"      # "modeled" | "measured" (used in later phases)
    order: int = 0              # solve-order key (used in later phases)

    def init_lambda(self, state: State) -> Tensor:
        """Zeroed Lagrange multiplier for this constraint, one per batch element."""
        return torch.zeros(
            state.batch, dtype=state.x_current.dtype, device=state.x_current.device
        )

    def project(self, state: State, lam: Tensor, dt: float) -> tuple[State, Tensor]:
        """One XPBD pass. Returns (updated state, updated lam)."""
        raise NotImplementedError
