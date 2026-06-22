"""The 2-DOF planar arm — the Phase 1 milestone (PLAN.md "Build target").

A revolute-revolute arm fixed to the ground, built in 3D about two stacked **world-z shared
edges**. It exercises the whole Phase-1 stack at once: points + Verlet + length + weld +
shared-edge hinge + servo actuator + anchor + the XPBD solver + velocity readout, with no
contact, no IMU and no floating base.

    point 0,1   shoulder edge  (vertical, axis = z)   -- shared base/link1, anchored
    point 2,3   elbow edge     (vertical, axis = z)   -- shared link1/link2
    point 4     tip                                   -- link2 only
    point 5     base frame point                      -- gives the base a frame, anchored

    base weld  = {0,1,5}        anchored to the world
    link1 weld = {0,1,2,3}      vertical "ladder" shoulder->elbow
    link2 weld = {2,3,4}        elbow->tip

    shoulder hinge: base -> link1, axis {0,1}, downstream = {2,3,4} (interior joint: the
                    whole distal arm swings as one)
    elbow hinge:    link1 -> link2, axis {2,3}, downstream = {4} (leaf)

At the build pose the arm lies straight along +x; both joint angles are 0 (calibrated).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..constraints import Actuator, Anchor, Hinge, Weld, solve
from ..core import State, verlet_step

Tensor = torch.Tensor


def fk(L1: float, L2: float, theta1: Tensor, theta2: Tensor) -> tuple[Tensor, Tensor]:
    """Analytic 2-link planar forward kinematics: (elbow, tip), each (..., 3) in the z=0 plane."""
    theta1 = torch.as_tensor(theta1)
    theta2 = torch.as_tensor(theta2)
    c1, s1 = torch.cos(theta1), torch.sin(theta1)
    c12, s12 = torch.cos(theta1 + theta2), torch.sin(theta1 + theta2)
    z = torch.zeros_like(c1)
    elbow = torch.stack([L1 * c1, L1 * s1, z], dim=-1)
    tip = torch.stack([L1 * c1 + L2 * c12, L1 * s1 + L2 * s12, z], dim=-1)
    return elbow, tip


def fk_tip_velocity(L1, L2, theta1, theta2, dtheta1, dtheta2) -> Tensor:
    """Analytic tip velocity J(theta) @ theta_dot, (..., 3)."""
    theta1 = torch.as_tensor(theta1)
    theta2 = torch.as_tensor(theta2)
    s1, c1 = torch.sin(theta1), torch.cos(theta1)
    s12, c12 = torch.sin(theta1 + theta2), torch.cos(theta1 + theta2)
    # d(tip)/dtheta1, d(tip)/dtheta2
    vx = (-L1 * s1 - L2 * s12) * dtheta1 + (-L2 * s12) * dtheta2
    vy = (L1 * c1 + L2 * c12) * dtheta1 + (L2 * c12) * dtheta2
    return torch.stack([vx, vy, torch.zeros_like(vx)], dim=-1)


TIP = 4
ELBOW = 2
BASE_POINTS = (0, 1, 5)


@dataclass
class Arm:
    state: State
    welds: list[Weld]
    shoulder: Hinge
    elbow: Hinge
    shoulder_act: Actuator
    elbow_act: Actuator
    anchors: list[Anchor]
    L1: float
    L2: float
    gain: float

    def constraints(self):
        # order: weld rigidity, then servos (toward theta_target), then anchors (pin base)
        cons = []
        for w in self.welds:
            cons += w.constraints()
        cons += [self.shoulder_act, self.elbow_act]
        cons += self.anchors
        return cons

    @property
    def targets(self) -> Tensor:
        return torch.stack([self.shoulder_act.theta_target, self.elbow_act.theta_target], -1)

    def command(self, dtheta: Tensor) -> None:
        """Apply a per-joint angle delta (action): theta_target += gain * dtheta. (B, 2)."""
        self.shoulder_act.theta_target = self.shoulder_act.theta_target + self.gain * dtheta[:, 0]
        self.elbow_act.theta_target = self.elbow_act.theta_target + self.gain * dtheta[:, 1]

    def step(self, dtheta: Tensor, dt: float, gravity=(0.0, 0.0, 0.0), iters: int = 20) -> State:
        self.command(dtheta)
        s = verlet_step(self.state, dt, gravity)
        s = solve(s, self.constraints(), dt, iters)
        self.state = s
        return s

    def tip(self) -> Tensor:
        return self.state.x_current[:, TIP]


def build_arm(
    L1: float = 1.0, L2: float = 0.8, h: float = 0.3, B: int = 1, gain: float = 1.0,
    dtype: torch.dtype = torch.float32, oop: float = 0.0,
) -> Arm:
    """Build the 2-DOF arm.

    oop : out-of-plane offset (meters). The whole structure otherwise lies in the y=0
          plane (flat plates in xz), so every weld is degenerate in y. When ``oop > 0`` an
          extra point displaced by ``+oop`` in y is welded into *each* link, giving each
          weld real 3D volume instead of a flat sheet. Wired into the hinge downstream sets
          (so they rotate rigidly with their link) and anchored on the base. Experiment:
          does bracing the welds out of plane kill the z wobble (PLAN Note 12)?
    """
    rows = [
        [0.0, 0.0, 0.0],      # 0 shoulder bottom
        [0.0, 0.0, h],        # 1 shoulder top
        [L1, 0.0, 0.0],       # 2 elbow bottom
        [L1, 0.0, h],         # 3 elbow top
        [L1 + L2, 0.0, 0.0],  # 4 tip
        [-0.3, 0.0, 0.0],     # 5 base frame point
    ]
    base_pts = [0, 1, 5]
    link1_pts = [0, 1, 2, 3]
    link2_pts = [2, 3, 4]
    sh_down = [2, 3, 4]   # shoulder rotates the whole distal arm
    el_down = None        # elbow leaf: defaults to child off-axis {4}
    anchor_pts = list(BASE_POINTS)

    if oop > 0.0:
        rows += [
            [0.0, oop, h / 2],            # 6 base out-of-plane    -> base weld (anchored)
            [L1 / 2, oop, h / 2],         # 7 link1 out-of-plane   -> link1 weld
            [L1 + L2 / 2, oop, h / 2],    # 8 link2 out-of-plane   -> link2 weld
        ]
        base_pts.append(6)
        link1_pts.append(7)
        link2_pts.append(8)
        sh_down = [2, 3, 4, 7, 8]         # 7 (link1) + 8 (link2) swing with the distal arm
        el_down = [4, 8]                  # link2's tip + its out-of-plane point
        anchor_pts.append(6)

    pts = torch.tensor(rows, dtype=dtype)
    N = pts.shape[0]
    state = State.at_rest(pts.expand(B, N, 3).clone(), torch.ones(B, N, dtype=dtype))

    base = Weld(base_pts, pts[base_pts])
    link1 = Weld(link1_pts, pts[link1_pts])
    link2 = Weld(link2_pts, pts[link2_pts])

    shoulder = Hinge(base, link1, downstream=sh_down).calibrate(state)
    elbow = Hinge(link1, link2, downstream=el_down).calibrate(state)

    zeros = torch.zeros(B, dtype=dtype)
    shoulder_act = Actuator(shoulder, theta_target=zeros.clone())
    elbow_act = Actuator(elbow, theta_target=zeros.clone())
    anchors = [Anchor(i, pts[i]) for i in anchor_pts]

    return Arm(state, [base, link1, link2], shoulder, elbow,
               shoulder_act, elbow_act, anchors, L1, L2, gain)
