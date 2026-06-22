"""Particle state for the position-based engine.

The engine is *positions only*: a point is just an (x, y, z). Orientation is **not**
stored here — it is emergent from welds (see PLAN.md, "Orientation model"). Velocity is
not stored either; in the Verlet scheme it is implicit, the gap between the current and
previous positions:  v ~= (x_current - x_previous) / dt.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass
class State:
    """Batched particle state.

    Shapes (B = batch, N = number of points, D = 3):
        x_current   (B, N, 3)   current positions
        x_previous  (B, N, 3)   positions one step ago (carries velocity implicitly)
        inv_mass    (B, N)      inverse mass per point; 0.0 == pinned (infinite mass)

    `inv_mass` may also be given as (N,) and is broadcast over the batch.
    """

    x_current: Tensor
    x_previous: Tensor
    inv_mass: Tensor

    def __post_init__(self) -> None:
        if self.x_current.shape != self.x_previous.shape:
            raise ValueError(
                f"x_current {tuple(self.x_current.shape)} and x_previous "
                f"{tuple(self.x_previous.shape)} must have the same shape"
            )
        if self.x_current.ndim != 3 or self.x_current.shape[-1] != 3:
            raise ValueError(
                f"positions must be (B, N, 3); got {tuple(self.x_current.shape)}"
            )

    # --- shape helpers ---------------------------------------------------
    @property
    def batch(self) -> int:
        return self.x_current.shape[0]

    @property
    def num_points(self) -> int:
        return self.x_current.shape[1]

    # --- derived quantities ----------------------------------------------
    def velocity(self, dt: float) -> Tensor:
        """Implicit Verlet velocity, (B, N, 3)."""
        return (self.x_current - self.x_previous) / dt

    # --- constructors ----------------------------------------------------
    @classmethod
    def at_rest(cls, positions: Tensor, inv_mass: Tensor | None = None) -> "State":
        """Build a state at rest (zero velocity): x_previous == x_current.

        `positions` is (B, N, 3). If `inv_mass` is None every point is free (1.0).
        """
        if inv_mass is None:
            inv_mass = torch.ones(
                positions.shape[:2], dtype=positions.dtype, device=positions.device
            )
        return cls(
            x_current=positions.clone(),
            x_previous=positions.clone(),
            inv_mass=inv_mass,
        )
