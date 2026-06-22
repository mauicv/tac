"""The free rigid box — the simplest model in the engine.

A box is a single `Weld` of its 8 cube corners (held rigid by pairwise lengths) and
*nothing else*: no hinges, no actuators, no anchor. It is the minimal exercise of the
core stack — points + Verlet gravity + weld rigidity (+ optional floor `Contact`) — and a
clean baseline to sanity-check dynamics in isolation before the articulated arm / gnoci.

    corners (relative to the centre, half-size s):
        0 (-s,-s,-s)   1 (+s,-s,-s)   2 (+s,+s,-s)   3 (-s,+s,-s)   <- bottom face (z low)
        4 (-s,-s,+s)   5 (+s,-s,+s)   6 (+s,+s,+s)   7 (-s,+s,+s)   <- top face   (z high)

The four bottom corners {0,1,2,3} double as the floor-contact points. Dropped from rest a
box is in **pure translation**, so the weld never sees a constraint violation and the fall
is exactly Verlet's — the engine's free-fall reference. Spin it (give the corners differing
x_previous) and the weld's pairwise lengths keep it rigid as it tumbles.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..constraints import Contact, Weld, solve
from ..core import State, verlet_step

Tensor = torch.Tensor

# corner offsets (in units of the half-size), bottom face first so {0,1,2,3} are the
# contacts and corners 0,1,2 are non-colinear (the weld's frame triad).
_CORNERS = [
    [-1.0, -1.0, -1.0],
    [+1.0, -1.0, -1.0],
    [+1.0, +1.0, -1.0],
    [-1.0, +1.0, -1.0],
    [-1.0, -1.0, +1.0],
    [+1.0, -1.0, +1.0],
    [+1.0, +1.0, +1.0],
    [-1.0, +1.0, +1.0],
]
BOTTOM = (0, 1, 2, 3)


@dataclass
class Box:
    state: State
    weld: Weld
    contacts: list[int]            # bottom-corner indices, used as floor contacts
    plane_z: float | None = None   # floor height; None = no floor in the scene

    def constraints(self, contact: bool = False):
        """Weld rigidity, optionally followed by the floor (one `Contact` per bottom corner)."""
        cons = list(self.weld.constraints())
        if contact and self.plane_z is not None:
            cons += [Contact(p, self.plane_z) for p in self.contacts]
        return cons

    def step(self, dt: float, gravity=(0.0, 0.0, -9.81), contact: bool = False,
             iters: int = 10) -> State:
        """Advance one step: Verlet under gravity, then solve the weld (+ floor)."""
        s = verlet_step(self.state, dt, gravity)
        s = solve(s, self.constraints(contact), dt, iters)
        self.state = s
        return s

    def center(self) -> Tensor:
        """The box centroid, (B, 3) — the corners' mean (== the rigid-body centre)."""
        return self.state.x_current.mean(dim=1)


def build_box(
    size: float = 0.2,
    center=(0.0, 0.0, 0.5),
    B: int = 1,
    dtype: torch.dtype = torch.float64,
    floor_z: float | None = None,
) -> Box:
    """Build a free rigid box.

    size    : full edge length (meters); corners sit at +/- size/2 from the centre.
    center  : (3,) build-pose centre of the box.
    floor_z : if given, the box's bottom corners become floor contacts at this height
              (pass it to `step(..., contact=True)` to let the box land).
    """
    half = size / 2.0
    c = torch.tensor(center, dtype=dtype)
    corners = c + half * torch.tensor(_CORNERS, dtype=dtype)   # (8, 3)
    N = corners.shape[0]

    state = State.at_rest(
        corners.expand(B, N, 3).clone(), torch.ones(B, N, dtype=dtype)
    )
    weld = Weld(list(range(N)), corners)
    return Box(state, weld, list(BOTTOM), floor_z)
