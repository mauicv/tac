"""Two rigid boxes joined at a single shared edge by a revolute hinge + servo.

Builds on `box.py`: two cubes (each a `Weld` of 8 corners) that **share exactly one edge**
(2 points). That shared edge *is* the hinge axis (PLAN "Revolute joint = two welds sharing
an edge") — box B swings about it, driven by an `Actuator` servo, while box A is the fixed
(anchored) base.

The edge is laid **horizontal** (along y), with the two cubes diagonal across it in the x-z
plane (A in the lower-left quadrant, B in the upper-right):

        z
        ^         +-------+
        |        /   B   /|         shared edge: the y-line at (x=0, z=0),
        |       +-------+ +         i.e. A's top-front edge == B's bottom-back edge
    ----+------#========#-----> x   (# = the 2 shared points)
        |     /|  edge ‖ y
        |    +-------+ /
        |    |   A   |/
        |    +-------+

Because the axis is horizontal, gravity (-z) makes a real **torque** about it (box B's mass
sits offset in +x), so the servo genuinely works against gravity — a vertical hinge would
feel none. Anchor the base (default) and the whole thing is a driven flap; unanchor it and
both boxes fall together while B still articulates relative to A.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..constraints import Actuator, Anchor, Contact, Hinge, Weld, solve
from ..core import State, verlet_step

Tensor = torch.Tensor


@dataclass
class HingedBoxes:
    state: State
    welds: list[Weld]            # [box A, box B]
    hinge: Hinge
    actuator: Actuator
    anchors: list[Anchor]        # box A's points (empty if unanchored)
    contacts: list[int]          # bottom corners, used as floor contacts
    plane_z: float | None
    gain: float

    def constraints(self, contact: bool = False):
        # order (as in Arm): weld rigidity, then the servo, then anchors, then the floor.
        cons = []
        for w in self.welds:
            cons += w.constraints()
        cons.append(self.actuator)
        cons += self.anchors
        if contact and self.plane_z is not None:
            cons += [Contact(p, self.plane_z) for p in self.contacts]
        return cons

    @property
    def target(self) -> Tensor:
        return self.actuator.theta_target

    def command(self, dtheta: Tensor) -> None:
        """Apply a hinge angle delta (action): theta_target += gain * dtheta. (B,)."""
        self.actuator.theta_target = self.actuator.theta_target + self.gain * dtheta

    def step(self, dtheta: Tensor, dt: float, gravity=(0.0, 0.0, -9.81),
             contact: bool = False, iters: int = 20) -> State:
        self.command(dtheta)
        s = verlet_step(self.state, dt, gravity)
        s = solve(s, self.constraints(contact), dt, iters)
        self.state = s
        return s

    def angle(self) -> Tensor:
        """Current hinge angle (B,), zero at the build pose."""
        return self.hinge.theta(self.state)


def build_hinged_boxes(
    size: float = 0.2,
    center=(0.0, 0.0, 0.0),
    B: int = 1,
    gain: float = 1.0,
    anchor: bool = True,
    dtype: torch.dtype = torch.float64,
    floor_z: float | None = None,
) -> HingedBoxes:
    """Build two cubes joined at one edge by a hinge + servo.

    size    : cube edge length (meters).
    center  : (3,) world translation added to the whole assembly.
    anchor  : pin box A so it is a fixed base (the default — box B then swings as a flap);
              False leaves both boxes free to fall under gravity while still hinged.
    floor_z : optional floor height; **every** point becomes a floor contact (so the
              assembly can land on any face / corner / edge as it falls and tumbles).
    """
    w = size
    rows: list[list[float]] = []

    def P(x: float, y: float, z: float) -> int:
        rows.append([x, y, z])
        return len(rows) - 1

    # the shared edge: a horizontal y-line at (x=0, z=0) -> the hinge axis (P->Q = +y).
    s0 = P(0.0, 0.0, 0.0)
    s1 = P(0.0, w, 0.0)
    # box A's other 6 corners: the lower-left x-z quadrant (x in [-w,0], z in [-w,0]).
    a = [P(-w, 0.0, -w), P(0.0, 0.0, -w), P(-w, 0.0, 0.0),
         P(-w, w, -w), P(0.0, w, -w), P(-w, w, 0.0)]
    # box B's other 6 corners: the upper-right x-z quadrant (x in [0,w], z in [0,w]).
    b = [P(w, 0.0, 0.0), P(w, 0.0, w), P(0.0, 0.0, w),
         P(w, w, 0.0), P(w, w, w), P(0.0, w, w)]

    box_a_pts = [s0, s1] + a    # 8 corners
    box_b_pts = [s0, s1] + b    # 8 corners, sharing the s0/s1 edge with A

    pts = torch.tensor(rows, dtype=dtype) + torch.tensor(center, dtype=dtype)
    N = pts.shape[0]
    state = State.at_rest(pts.expand(B, N, 3).clone(), torch.ones(B, N, dtype=dtype))

    box_a = Weld(box_a_pts, pts[box_a_pts])
    box_b = Weld(box_b_pts, pts[box_b_pts])

    # hinge: A -> B about the shared edge; the actuator swings B's 6 off-edge corners.
    hinge = Hinge(box_a, box_b).calibrate(state)
    actuator = Actuator(hinge, theta_target=torch.zeros(B, dtype=dtype))

    anchors = [Anchor(i, pts[i]) for i in box_a_pts] if anchor else []

    # every point is a floor contact, so the assembly can land on any corner/edge/face.
    contacts = list(range(N)) if floor_z is not None else []

    return HingedBoxes(state, [box_a, box_b], hinge, actuator, anchors,
                       contacts, floor_z, gain)
