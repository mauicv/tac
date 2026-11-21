"""ModelBuilder: assemble points + constraints for a robot, then build a sim-ready Model.

Points are append-only: every `add_*` method appends new points and returns their ids,
and nothing is ever removed or reordered, so an id stays valid for the builder's life.
Constraints are recorded against those ids immediately; the `State` (and therefore the
constraints' rest values, captured from it) is only created in `build()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import warnings

import jax.numpy as jnp
import numpy as np

from tac.constraints.autograd import AutogradConstraint
from tac.constraints.hard_floor import HardFloorConstraint
from tac.constraints.builders import ConstraintBuilder, JointBuilder, LengthBuilder
from tac.core.state import State
from tac.effects.inertial import Gravity, Inertia
from tac.io.mjcf import RobotSpec
from tac.model.geometry import OrientedBox, oriented_box, patch_corners


@dataclass
class Model:
    state: State
    constraints: list            # ordered: lengths -> joints -> contact
    effects: list                # ordered: Inertia -> Gravity
    bodies: dict[str, list[int]] # body name -> 8 corner point ids
    joints: dict[str, tuple[int, int]]  # joint name -> (axis0, axis1), axis1 - axis0 ∥ +axis
    actuators: dict[str, int]    # actuator name -> control column
    contacts: dict[str, list[int]]  # body name -> floor-contact point ids
    sites: dict[str, int]        # site name -> point id
    boxes: dict[str, OrientedBox]
    lines: list[tuple[int, int]] # for the Renderer
    faces: list[list[int]]       # box faces (quads) for the Renderer


@dataclass
class ModelBuilder:
    alpha: float = 0.0           # compliance of rigid-body lengths
    actuator_alpha: float = 0.0  # compliance of actuated joint angles
    min_half: float = 0.003      # minimum box half-extent (m)

    positions: list[np.ndarray] = field(default_factory=list)
    inv_mass: list[float] = field(default_factory=list)
    bodies: dict[str, list[int]] = field(default_factory=dict)
    joints: dict[str, tuple[int, int]] = field(default_factory=dict)
    joint_bodies: dict[str, tuple[str, str]] = field(default_factory=dict)
    actuators: dict[str, int] = field(default_factory=dict)
    contacts: dict[str, list[int]] = field(default_factory=dict)
    sites: dict[str, int] = field(default_factory=dict)
    boxes: dict[str, OrientedBox] = field(default_factory=dict)
    lines: list[tuple[int, int]] = field(default_factory=list)
    faces: list[list[int]] = field(default_factory=list)
    lengths: LengthBuilder = field(default_factory=LengthBuilder)
    angles: JointBuilder = field(default_factory=JointBuilder)   # rest=None: home angle

    # --- points -------------------------------------------------------------------------

    def _add_point(self, pos: np.ndarray, inv_mass: float = 1.0) -> int:
        self.positions.append(np.asarray(pos, dtype=np.float64))
        self.inv_mass.append(inv_mass)
        return len(self.positions) - 1

    # --- components ---------------------------------------------------------------------

    def add_body(self, name: str, verts: np.ndarray) -> list[int]:
        """Rigid body from a (V, 3) world-frame vertex cloud: 8 PCA-box corners + 28 lengths."""
        if name in self.bodies:
            raise ValueError(f"body {name} already added")
        box = oriented_box(verts, min_half=self.min_half)
        ids = [self._add_point(c) for c in box.corners]
        for i, j in itertools.combinations(ids, 2):
            self.lengths.add_constraint([i, j], self.alpha)
        self.bodies[name] = ids
        self.boxes[name] = box
        self.lines += [(ids[i], ids[j]) for i, j in box.edges]
        self.faces += [[ids[i] for i in face] for face in box.faces]
        return ids

    def add_hinge(
            self,
            name: str,
            parent: str,
            child: str,
            anchor: np.ndarray,
            axis: np.ndarray,
            half_length: float | None = None,
        ) -> tuple[int, int]:
        """Revolute joint: two axis points shared by parent and child.

        The points sit at `anchor ∓ h·axis` (so axis1 − axis0 points along +axis) and are
        tied by lengths to the 8 corners of both bodies; the only motion left between the
        bodies is rotation about the line through them. `h` defaults to the smaller of the
        two boxes' half-extents along the axis.
        """
        if name in self.joints:
            raise ValueError(f"joint {name} already added")
        for body in (parent, child):
            if body not in self.bodies:
                raise ValueError(f"joint {name}: body {body} not added yet")
        anchor = np.asarray(anchor, dtype=np.float64)
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        if half_length is None:
            half_length = min(self.boxes[parent].extent_along(axis),
                              self.boxes[child].extent_along(axis))
        ids = (self._add_point(anchor - half_length * axis),
               self._add_point(anchor + half_length * axis))
        for p in ids:
            for c in self.bodies[parent] + self.bodies[child]:
                self.lengths.add_constraint([p, c], self.alpha)
        self.joints[name] = ids
        self.joint_bodies[name] = (parent, child)
        self.lines.append(ids)
        return ids

    def _farthest_corner(self, body: str, axis_ids: tuple[int, int]) -> int:
        """Corner of `body` farthest from the line through the two axis points."""
        p0, p1 = (self.positions[i] for i in axis_ids)
        n = (p1 - p0) / np.linalg.norm(p1 - p0)
        def dist(i):
            r = self.positions[i] - p0
            return np.linalg.norm(r - (r @ n) * n)
        return max(self.bodies[body], key=dist)

    def add_actuator(self, name: str, joint: str) -> int:
        """Position actuator on a hinge; returns its control column.

        One angle constraint over [axis0, axis1, child arm, parent arm], arms being each
        body's corner farthest from the axis. Its rest is the home angle, so control 0 holds
        home. Child arm first makes the measured angle −q (q = MuJoCo joint angle, right-hand
        about +axis); with residual `C − rest + u` the solver drives C → rest − u, i.e.
        q → u: control is the target joint angle in radians, same sign as MuJoCo ctrl.
        """
        if name in self.actuators:
            raise ValueError(f"actuator {name} already added")
        if joint not in self.joints:
            raise ValueError(f"actuator {name}: joint {joint} not added yet")
        axis_ids = self.joints[joint]
        parent, child = self.joint_bodies[joint]
        arms = [self._farthest_corner(child, axis_ids), self._farthest_corner(parent, axis_ids)]
        column = len(self.actuators)
        self.angles.add_constraint([*axis_ids, *arms], self.actuator_alpha, control=column)
        self.actuators[name] = column
        return column

    def add_contacts(self, body: str, points: np.ndarray) -> list[int]:
        """Floor-contact points on `body`: each tied by 8 lengths to the body's corners.

        With 3+ points, they are rendered as one polygon (the sole), ordered around
        their centroid in xy.
        """
        if body not in self.bodies:
            raise ValueError(f"contacts: body {body} not added yet")
        points = np.atleast_2d(points)
        ids = [self._add_point(p) for p in points]
        for p in ids:
            for c in self.bodies[body]:
                self.lengths.add_constraint([p, c], self.alpha)
        self.contacts.setdefault(body, []).extend(ids)
        if len(ids) >= 3:
            d = points[:, :2] - points[:, :2].mean(axis=0)
            loop = [ids[k] for k in np.argsort(np.arctan2(d[:, 1], d[:, 0]))]
            self.faces.append(loop)
            self.lines += list(zip(loop, loop[1:] + loop[:1]))
        return ids

    def add_site(self, name: str, body: str, pos: np.ndarray) -> int:
        """Named point (e.g. IMU) rigidly attached to `body` by 8 lengths to its corners."""
        if name in self.sites:
            raise ValueError(f"site {name} already added")
        if body not in self.bodies:
            raise ValueError(f"site {name}: body {body} not added yet")
        i = self._add_point(pos)
        for c in self.bodies[body]:
            self.lengths.add_constraint([i, c], self.alpha)
        self.sites[name] = i
        return i

    @classmethod
    def from_spec(cls, spec: RobotSpec, contact_eps: float = 0.0015, **kwargs) -> ModelBuilder:
        """Build from a loaded robot. Contacts: per body, the mesh vertices within
        `contact_eps` of the robot's lowest point (reduced to ≤4 corners)."""
        builder = cls(**kwargs)
        for body in spec.bodies.values():
            builder.add_body(body.name, body.verts)
        for j in spec.joints.values():
            builder.add_hinge(j.name, j.parent, j.child, j.anchor, j.axis)
        for a in spec.actuators:
            builder.add_actuator(a.name, a.joint)
        for site in spec.sites.values():
            builder.add_site(site.name, site.body, site.pos)
        z_min = min(b.verts[:, 2].min() for b in spec.bodies.values())
        for b in spec.bodies.values():
            patch = b.verts[b.verts[:, 2] < z_min + contact_eps]
            if len(patch):
                builder.add_contacts(b.name, patch_corners(patch))
        if len(builder.contacts) < 2:
            warnings.warn(f"only {len(builder.contacts)} body touches the floor: {list(builder.contacts)}")
        return builder

    # --- build --------------------------------------------------------------------------

    @staticmethod
    def _build(cb: ConstraintBuilder, state: State) -> list[AutogradConstraint]:
        # a fresh ConstraintBuilder holds one empty set; don't build it
        if not cb.constraint_sets[0]['points']:
            return []
        return cb.build(state)

    def build(
            self,
            batch: int = 1,
            floor_z: float | None = None,
            gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
            friction: float = 0.0,
        ) -> Model:
        x = jnp.asarray(np.stack(self.positions), dtype=jnp.float32)          # (N, 3)
        w = jnp.asarray(self.inv_mass, dtype=jnp.float32)                    # (N,)
        state = State.create(
            x_current=jnp.broadcast_to(x, (batch, *x.shape)),                # (B, N, 3)
            inv_mass=jnp.broadcast_to(w, (batch, *w.shape)),                 # (B, N)
            control_size=len(self.actuators),                                 # (B, n_act)
        )
        constraints = [
            *self._build(self.lengths, state),
            *self._build(self.angles, state),
        ]
        contact_ids = [i for ids in self.contacts.values() for i in ids]
        if contact_ids:
            if floor_z is None:   # default: the floor is where the contacts are at build time
                floor_z = float(min(self.positions[i][2] for i in contact_ids))
            constraints.append(HardFloorConstraint.create(contact_ids, floor_z))   # last
        dynamic = jnp.asarray(np.flatnonzero(np.asarray(self.inv_mass) > 0), dtype=jnp.int32)
        effects = [
            Inertia(dynamic, jnp.array([friction], dtype=jnp.float32)),
            Gravity(dynamic, jnp.asarray(gravity, dtype=jnp.float32)),
        ]
        return Model(
            state=state,
            constraints=constraints,
            effects=effects,
            bodies=dict(self.bodies),
            joints=dict(self.joints),
            actuators=dict(self.actuators),
            contacts={k: list(v) for k, v in self.contacts.items()},
            sites=dict(self.sites),
            boxes=dict(self.boxes),
            lines=list(self.lines),
            faces=[list(f) for f in self.faces],
        )
