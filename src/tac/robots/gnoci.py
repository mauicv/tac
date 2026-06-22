"""gnoci — the bipedal robot, built by hand (PLAN.md "The gnoci robot", build step 9).

Floating-base biped: torso (`head_base`) → per leg a 5-hinge serial chain
**yaw → roll → pitch → knee → ankle**, mirrored left/right. Each body is a `Weld`;
consecutive bodies **share the 2-point edge that is their hinge axis** (PLAN "Revolute joint
= two welds sharing an edge"). The stored representation is purely points + welds + shared
edges; there is no builder DSL (Note 13) — the two legs are laid out by a private `_leg`
helper, called once per side.

**Idealized geometry.** Built from a clean, topologically-faithful layout (round coordinates,
legs hanging straight down, every joint angle = 0 at the build pose), *not* the exact onshape
mesh in `gnoci_xml/`. Metric calibration to MuJoCo is deferred to the learning / end-to-end
steps (PLAN: "learn axis-point rest positions"); gnoci_desc.py is the loose API target.

**Stacked orthogonal hip** (the geometry the 2-DOF arm could not exercise): the first three
hinges share edges along three mutually orthogonal axes through ~the hip point — **yaw about
z, roll about x, pitch about y** — then knee and ankle are pitch axes (y), parallel below
(PLAN: "stacked hip hinges ... orthogonal"; "pitch at the hip, knee at the knee ... parallel").

Coordinates: forward +x, left +y, up +z.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..constraints import Actuator, Contact, Hinge, Weld, solve
from ..core import State, verlet_step

Tensor = torch.Tensor

# --- idealized dimensions (meters) ------------------------------------------------
HIP_Y = 0.05      # half hip width (left leg at +HIP_Y, right at -HIP_Y)
EDGE = 0.03       # half-length of each hinge-axis edge
UPPER = 0.14      # upper-leg length (hip → knee, down -z)
LOWER = 0.14      # lower-leg length (knee → ankle)
SOLE = 0.05       # ankle → sole drop
FOOT = 0.05       # foot half-length (forward/back contact spread)
FLOOR_Z = -(UPPER + LOWER + SOLE + 0.1)   # sole height in the build pose (idealized floor)


@dataclass
class Leg:
    """One leg's welds + hinges + actuators + the 2 foot-contact point indices."""
    welds: list[Weld]
    hinges: list[Hinge]
    actuators: list[Actuator]
    contacts: tuple[int, int]


@dataclass
class Gnoci:
    state: State
    welds: list[Weld]
    hinges: list[Hinge]
    actuators: list[Actuator]
    contacts: list[int]          # all foot-contact point indices (4: 2 per foot)
    imu_point: int
    plane_z: float = FLOOR_Z
    gain: float = 1.0
    legs: tuple[Leg, Leg] = field(default=None)

    # --- constraint assembly --------------------------------------------------
    def modeled_constraints(self, contact: bool = False):
        """The dynamics prior: weld rigidity, then the servos. Optionally the simulated
        floor (geometric `Contact`) — off by default so the gravity-free static / hip tests
        see only the modeled skeleton (PLAN: one GS loop, measured layer added in step 11)."""
        cons = []
        for w in self.welds:
            cons += w.constraints()
        cons += self.actuators
        if contact:
            cons += [Contact(p, self.plane_z) for p in self.contacts]
        return cons

    @property
    def targets(self) -> Tensor:
        return torch.stack([a.theta_target for a in self.actuators], dim=-1)  # (B, 10)

    def command(self, dtheta: Tensor) -> None:
        """Per-joint angle delta (action): theta_target += gain * dtheta. (B, 10)."""
        for j, a in enumerate(self.actuators):
            a.theta_target = a.theta_target + self.gain * dtheta[:, j]

    def step_sim(self, dtheta: Tensor, dt: float, gravity=(0.0, 0.0, -9.81),
                 iters: int = 8, contact: bool = True) -> State:
        """Forward dynamics step (like `Arm.step`): command → Verlet → one GS solve."""
        self.command(dtheta)
        s = verlet_step(self.state, dt, gravity)
        s = solve(s, self.modeled_constraints(contact=contact), dt, iters)
        self.state = s
        return s

    def joint_angles(self) -> Tensor:
        return torch.stack([h.theta(self.state) for h in self.hinges], dim=-1)  # (B, 10)


def build_gnoci(B: int = 1, dtype: torch.dtype = torch.float64) -> Gnoci:
    rows: list[list[float]] = []

    def P(x: float, y: float, z: float) -> int:
        rows.append([x, y, z])
        return len(rows) - 1

    # --- torso (head_base): two vertical yaw edges + an IMU/frame point ----------
    # left & right yaw axes are vertical (z) edges at (0, ±HIP_Y, 0).
    lA1, lA2 = P(0.0, HIP_Y, EDGE), P(0.0, HIP_Y, -EDGE)
    rA1, rA2 = P(0.0, -HIP_Y, EDGE), P(0.0, -HIP_Y, -EDGE)
    imu = P(0.04, 0.0, 0.03)   # forward-up: gives the torso area + carries the IMU
    torso_pts = [lA1, lA2, rA1, rA2, imu]

    legs: list[Leg] = []

    def leg(side: float, A1: int, A2: int) -> Leg:
        """Build one leg below the torso yaw edge (A1, A2). side = +1 left, -1 right."""
        y0 = side * HIP_Y

        # hinge-axis edges, each a 2-point shared edge through ~the hip / joint centre:
        #   roll  (x): B   |  pitch (y): D   |  knee (y): E   |  ankle (y): F
        B1, B2 = P(EDGE, y0, 0.0), P(-EDGE, y0, 0.0)                 # roll axis ‖ x
        D1, D2 = P(0.0, y0 + EDGE, 0.0), P(0.0, y0 - EDGE, 0.0)      # pitch axis ‖ y
        zk = -UPPER
        E1, E2 = P(0.0, y0 + EDGE, zk), P(0.0, y0 - EDGE, zk)        # knee axis ‖ y
        za = -(UPPER + LOWER)
        F1, F2 = P(0.0, y0 + EDGE, za), P(0.0, y0 - EDGE, za)        # ankle axis ‖ y
        zs = -(UPPER + LOWER + SOLE)
        Cf, Cb = P(FOOT, y0, zs), P(-FOOT, y0, zs)                   # forward / back contacts

        yoke = Weld([A1, A2, B1, B2], _rest([A1, A2, B1, B2]))       # yaw edge + roll edge
        hip = Weld([B1, B2, D1, D2], _rest([B1, B2, D1, D2]))        # roll edge + pitch edge
        upper = Weld([D1, D2, E1, E2], _rest([D1, D2, E1, E2]))      # pitch edge + knee edge
        lower = Weld([E1, E2, F1, F2], _rest([E1, E2, F1, F2]))      # knee edge + ankle edge
        foot = Weld([F1, F2, Cf, Cb], _rest([F1, F2, Cf, Cb]))       # ankle edge + 2 contacts

        # downstream sets: every point strictly below each hinge (nested, deepest-up).
        ankle_ds = [Cf, Cb]
        knee_ds = [F1, F2] + ankle_ds
        pitch_ds = [E1, E2] + knee_ds
        roll_ds = [D1, D2] + pitch_ds
        yaw_ds = [B1, B2] + roll_ds

        rev = side > 0   # mirror the left leg's angle sign (gnoci_desc convention)
        yaw = Hinge(torso, yoke, reverse=rev, downstream=yaw_ds)
        roll = Hinge(yoke, hip, reverse=rev, downstream=roll_ds)
        pitch = Hinge(hip, upper, reverse=rev, downstream=pitch_ds)
        knee = Hinge(upper, lower, reverse=rev, downstream=knee_ds)
        ankle = Hinge(lower, foot, reverse=rev, downstream=ankle_ds)

        hinges = [yaw, roll, pitch, knee, ankle]
        return Leg(
            welds=[yoke, hip, upper, lower, foot],
            hinges=hinges,
            actuators=[],   # filled after calibration, once the state exists
            contacts=(Cf, Cb),
        )

    def _rest(idx):
        return torch.tensor([rows[i] for i in idx], dtype=dtype)

    torso = Weld(torso_pts, _rest(torso_pts))
    legs.append(leg(+1.0, lA1, lA2))   # left
    legs.append(leg(-1.0, rA1, rA2))   # right

    pts = torch.tensor(rows, dtype=dtype)
    N = pts.shape[0]
    state = State.at_rest(pts.expand(B, N, 3).clone(), torch.ones(B, N, dtype=dtype))

    welds = [torso]
    hinges: list[Hinge] = []
    contacts: list[int] = []
    zeros = torch.zeros(B, dtype=dtype)
    for lg in legs:
        welds += lg.welds
        for h in lg.hinges:
            h.calibrate(state)                                   # theta = 0 at build pose
            lg.actuators.append(Actuator(h, theta_target=zeros.clone()))
        hinges += lg.hinges
        contacts += list(lg.contacts)

    actuators = [a for lg in legs for a in lg.actuators]
    return Gnoci(state, welds, hinges, actuators, contacts, imu, FLOOR_Z, 1.0,
                 (legs[0], legs[1]))
