"""Scene — the renderable topology of a model, as **plain data**.

Per PLAN.md Note 11 the renderer reads plain arrays, never the live `core/` types, so the
viewer can't prejudge the engine's representation. A `Scene` is just index bookkeeping
(which points form each weld, which two points are each hinge axis, etc.); the renderer
pairs it with a `State`'s positions at draw time.

Building a Scene also **validates structure** (the plan's free correctness net for hand-typed
geometry): every weld must have >=3 points whose first three are non-colinear (so it carries
a frame), and every hinge axis must be two distinct points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-9


@dataclass
class Scene:
    weld_groups: list[list[int]]                  # point indices per rigid cluster
    hinge_axes: list[tuple[int, int]]             # (P, Q) oriented shared edge per hinge
    anchor_points: list[int] = field(default_factory=list)
    contact_points: list[int] = field(default_factory=list)
    imu_site: int | None = None
    floor_z: float | None = None

    def validate(self, positions: np.ndarray) -> "Scene":
        """Check the structural invariants against a build pose; raise on violation."""
        for g in self.weld_groups:
            if len(g) < 3:
                raise ValueError(f"weld {g} has <3 points; cannot carry orientation")
            p0, p1, p2 = (positions[g[i]] for i in range(3))
            area = np.linalg.norm(np.cross(p1 - p0, p2 - p0))
            if area < _EPS:
                raise ValueError(
                    f"weld {g}: first three points are colinear (area={area:.2e}); "
                    "its frame would be degenerate"
                )
        for (P, Q) in self.hinge_axes:
            if P == Q:
                raise ValueError(f"hinge axis ({P}, {Q}) needs two distinct points")
        return self


def scene_from_arm(arm) -> Scene:
    """Extract a validated Scene from a built `Arm`."""
    scene = Scene(
        weld_groups=[list(w.points) for w in arm.welds],
        hinge_axes=[(arm.shoulder.P, arm.shoulder.Q), (arm.elbow.P, arm.elbow.Q)],
        anchor_points=[a.point for a in arm.anchors],
    )
    return scene.validate(arm.state.x_current[0].detach().cpu().numpy())


def scene_from_box(box) -> Scene:
    """Extract a validated Scene from a built `Box` (one weld, optional floor + contacts)."""
    scene = Scene(
        weld_groups=[list(box.weld.points)],
        hinge_axes=[],
        contact_points=list(box.contacts) if box.plane_z is not None else [],
        floor_z=box.plane_z,
    )
    return scene.validate(box.state.x_current[0].detach().cpu().numpy())


def scene_from_hinged_boxes(hb) -> Scene:
    """Extract a validated Scene from a built `HingedBoxes` (two welds + one hinge axis)."""
    scene = Scene(
        weld_groups=[list(w.points) for w in hb.welds],
        hinge_axes=[(hb.hinge.P, hb.hinge.Q)],
        anchor_points=[a.point for a in hb.anchors],
        contact_points=list(hb.contacts) if hb.plane_z is not None else [],
        floor_z=hb.plane_z,
    )
    return scene.validate(hb.state.x_current[0].detach().cpu().numpy())


def scene_from_gnoci(gnoci) -> Scene:
    """Extract a validated Scene from a built `Gnoci` biped.

    Draws all 11 welds, each of the 10 shared-edge hinge axes (the stacked orthogonal hip
    axes should render mutually perpendicular), the 4 foot-contact points, the IMU site, and
    the floor.
    """
    scene = Scene(
        weld_groups=[list(w.points) for w in gnoci.welds],
        hinge_axes=[(h.P, h.Q) for h in gnoci.hinges],
        contact_points=list(gnoci.contacts),
        imu_site=gnoci.imu_point,
        floor_z=gnoci.plane_z,
    )
    return scene.validate(gnoci.state.x_current[0].detach().cpu().numpy())
