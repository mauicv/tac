"""Renderer — log a Scene + State(s) to rerun (rerun.io).

`rerun` is an optional dependency (the `viz` extra) and is imported lazily, so `import tac`
never requires it. Use `spawn=True` to pop the native viewer, or `save="run.rrd"` to log
headlessly to a file you can open later (`rerun run.rrd`) — the latter is what the tests use.

Drawn (PLAN.md Note 11): points, weld edges (all pairwise, = the rigidity constraints),
each hinge axis as a ± arrow along the shared edge, plus optional anchor / contact / IMU
highlights and a floor grid. Batched state: render one batch element (default 0).
"""

from __future__ import annotations

import itertools

import numpy as np

from .scene import Scene

_POINT = [200, 200, 200]
_WELD = [80, 140, 255]
_AXIS = [255, 80, 80]
_ANCHOR = [80, 220, 120]
_CONTACT = [255, 160, 40]
_IMU = [230, 80, 230]
_FLOOR = [110, 110, 110]


def _positions(state, b: int) -> np.ndarray:
    return state.x_current[b].detach().cpu().numpy()


class Renderer:
    def __init__(self, app: str = "tac", spawn: bool = False, save: str | None = None):
        import rerun as rr  # lazy: keeps rerun out of the core import path

        self._rr = rr
        rr.init(app)
        if save is not None:
            rr.save(str(save))
        elif spawn:
            rr.spawn()
        self._floor_logged = False

    def show(self, scene: Scene, state, b: int = 0) -> None:
        """Log a single static pose (batch element `b`)."""
        self._log(scene, _positions(state, b))

    def play(self, scene: Scene, states, dt: float | None = None, b: int = 0) -> None:
        """Log a rollout across a timeline; scrub the `step` axis in the viewer."""
        rr = self._rr
        for t, s in enumerate(states):
            rr.set_time("step", sequence=t)
            self._log(scene, _positions(s, b))

    # --- internals -------------------------------------------------------
    def _log(self, scene: Scene, x: np.ndarray) -> None:
        rr = self._rr
        rr.log("world/points", rr.Points3D(x, radii=0.02, colors=_POINT))

        strips = [
            np.stack([x[a], x[b]])
            for g in scene.weld_groups
            for a, b in itertools.combinations(g, 2)
        ]
        if strips:
            rr.log("world/welds", rr.LineStrips3D(strips, radii=0.005, colors=_WELD))

        if scene.hinge_axes:
            origins = np.stack([x[P] for P, _ in scene.hinge_axes])
            vectors = np.stack([x[Q] - x[P] for P, Q in scene.hinge_axes])
            rr.log("world/axes", rr.Arrows3D(origins=origins, vectors=vectors, colors=_AXIS))

        self._highlight("world/anchors", x, scene.anchor_points, _ANCHOR)
        self._highlight("world/contacts", x, scene.contact_points, _CONTACT)
        if scene.imu_site is not None:
            self._highlight("world/imu", x, [scene.imu_site], _IMU)

        if scene.floor_z is not None and not self._floor_logged:
            self._log_floor(scene.floor_z)
            self._floor_logged = True

    def _highlight(self, path: str, x: np.ndarray, idxs, color) -> None:
        if idxs:
            self._rr.log(path, self._rr.Points3D(x[list(idxs)], radii=0.03, colors=color))

    def _log_floor(self, z: float, half: float = 1.0, n: int = 10) -> None:
        rr = self._rr
        lines = []
        for i in range(n + 1):
            a = -half + 2 * half * i / n
            lines.append(np.array([[-half, a, z], [half, a, z]]))
            lines.append(np.array([[a, -half, z], [a, half, z]]))
        rr.log("world/floor", rr.LineStrips3D(lines, radii=0.002, colors=_FLOOR), static=True)
