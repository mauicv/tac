"""Contact — the asymmetric floor (PLAN.md, Decision 10 + Note 5).

The only constraint that edits **`x_previous`** as well as `x_current`. In the Verlet scheme
velocity is implicit (`v ~ (x_current - x_previous)/dt`), and contact is a *velocity*-level
condition, so it has to move `x_previous` to change velocity:

  - **z is one-sided (floor pushes up only).** While a point is below the floor, snap **both**
    `z_current` *and* `z_previous` to the floor height. Lifting `z_current` alone leaves a
    velocity across the gap to `z_previous` — the point either keeps driving through or, pushed
    past it, **bounces**. Snapping both sets `v_z = 0`: it rests, no bounce. Above the floor the
    point is free, so a foot can lift.
  - **xy is no-slip while in contact.** Pin `x_current.xy = x_previous.xy`, zeroing the in-plane
    velocity so the foot sticks (infinite friction; a Coulomb cone is a later refinement).

Per contact point (2 per foot, both in the triangular foot weld). It is a hard kinematic
projection, not a compliant XPBD spring, so it ignores `lam`/`alpha`. Runs **last** in the
solver list and iterates against the foot weld's rigidity; two pinned points fix the stance
foot's position and heading, which is what anchors the floating base (Note 8 double support).

Activation has two sources (same constraint):
  - **modeled** (forward sim): geometric, `z < plane_z`.
  - **measured** (estimation): the foot's **touch sensor** — pass it as `active`, and the
    floor is enforced because the switch says the foot is grounded, regardless of modeled z.

The snap is non-smooth at the contact boundary (Note 5); gradients through z are cut while in
contact (straight-through-like). A soft/compliant floor is the deferred smooth variant.
"""

from __future__ import annotations

import torch

from ..core.state import State
from .base import Constraint

Tensor = torch.Tensor


class Contact(Constraint):
    def __init__(self, point: int, plane_z: float, active: Tensor | None = None,
                 scope: str = "modeled"):
        """Asymmetric floor contact for one point.

        point   : the point index this floor acts on.
        plane_z : floor height (gnoci scene: -0.23).
        active  : optional (B,) bool override (a touch sensor, measured scope). When None,
                  contact engages geometrically where the point is below the floor.
        scope   : "modeled" (geometric) or "measured" (sensor-driven); informational for now.
        """
        self.point = int(point)
        self.plane_z = plane_z
        self.active = active
        self.scope = scope

    def project(self, state: State, lam: Tensor, dt: float) -> tuple[State, Tensor]:
        x, xp = state.x_current, state.x_previous
        p = self.point
        z = x[:, p, 2]
        plane = torch.full_like(z, float(self.plane_z))

        if self.active is None:
            below = z < plane                                  # (B,) geometric
        else:
            below = self.active.to(device=z.device, dtype=torch.bool).broadcast_to(z.shape)

        new_x, new_xp = x.clone(), xp.clone()
        # z: push up to the floor and kill z-velocity (snap current AND previous), no bounce.
        new_x[:, p, 2] = torch.where(below, plane, z)
        new_xp[:, p, 2] = torch.where(below, plane, xp[:, p, 2])
        # xy: no-slip -> pin current in-plane position to previous (zero in-plane velocity).
        new_x[:, p, :2] = torch.where(below.unsqueeze(-1), xp[:, p, :2], x[:, p, :2])

        return State(new_x, new_xp, state.inv_mass), lam      # hard: lam untouched
