from typing import List, Union
from tac.constraints.length import Length
from tac.constraints.weld import FacetedWeld
from tac.constraints.joint import Joint
import pyvista as pv
from tac.core.state import State
import numpy as np


def weld_2_cells(weld: FacetedWeld):
    faces = weld.faces
    cells = []

    for face in faces:
        cells.append(len(face))
        cells.extend(face)

    return cells


def length_2_lines(length: Length):
    cells = []
    cells.append(2)
    cells.extend([length.i, length.j])
    return cells

# distinct colours cycled per joint
JOINT_COLORS = ["cyan", "magenta", "yellow", "lime", "orange", "deepskyblue", "hotpink", "white"]


def joint_2_arrows(joint: Joint, points: np.ndarray):
    """Three unit-direction arrows visualising a joint, all based at the first axis
    point p: p -> q along the hinge axis, and p -> from / p -> to (the lever arms).
    Directions are normalised so every arrow renders the same size (length is set by
    the glyph factor, not the lever-arm magnitude)."""
    p = points[joint.axis_points[0]]
    q = points[joint.axis_points[1]]
    f = points[joint.from_point]
    t = points[joint.to_point]
    bases = np.array([p, p, p])
    vectors = np.array([q - p, f - p, t - p])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.clip(norms, 1e-6, None)   # unit directions
    return bases, vectors


def state_2_points(state: State):
    return state.x_current[0].to('cpu').numpy()


class Renderer:
    def __init__(
            self,
            state: State,
            visable_constraints: List[Union[FacetedWeld]],
            camera_distance: float = 5.0
        ):
        self.plotter = pv.Plotter()
        points = state_2_points(state)

        # pyvista distinguishes faces (polygons, >=3 verts) from lines (2 verts).
        # A Length is a line segment, so its cells must be passed via `lines=`, not
        # as a face — a 2-vertex face is degenerate and renders as nothing.
        face_cells = []
        line_cells = []
        self.joints = []
        for visable_constraint in visable_constraints:
            if isinstance(visable_constraint, FacetedWeld):
                face_cells.extend(weld_2_cells(visable_constraint))
            elif isinstance(visable_constraint, Length):
                line_cells.extend(length_2_lines(visable_constraint))
            elif isinstance(visable_constraint, Joint):
                # joints aren't faces/lines on the point mesh — drawn as arrow glyphs
                self.joints.append(visable_constraint)

        self.mesh = pv.PolyData(
            points,
            faces=face_cells or None,
            lines=line_cells or None,
        )
        self.plotter.add_mesh(
            self.mesh,
            line_width=5,
            show_edges=True,
            color='red',
            style='wireframe'
        )

        # joints are rendered as arrow glyphs (axis + two lever arms), one actor per
        # joint so each gets its own colour; refreshed each frame in update() since
        # the geometry moves with the points.
        self.arrow_length = 0.2
        self.joint_actors = []
        for i, joint in enumerate(self.joints):
            color = JOINT_COLORS[i % len(JOINT_COLORS)]
            self.joint_actors.append(
                self.plotter.add_mesh(self._joint_glyphs(joint, points), color=color)
            )

        center = self.mesh.points.mean(axis=0)
        self.plotter.camera.position = center + np.array([camera_distance, camera_distance, camera_distance])
        self.plotter.camera.focal_point = center
        self.plotter.camera.up = [0, 0, 1]

        self.plotter.show(interactive_update=True)

        self.arrows = None

    def add_arrows(self, points: List[float], vectors: List[float]):
        points = np.array(points).reshape(-1, 3)
        vectors = np.array(vectors).reshape(-1, 3)

        arrow_mesh = pv.PolyData(points)
        arrow_mesh['vectors'] = vectors
        arrow_mesh.set_active_vectors('vectors')
        glyphs = arrow_mesh.glyph(orient='vectors', scale=False, factor=1.0)

        if self.arrows is None:
            self.arrows = self.plotter.add_mesh(glyphs, color='red')
        else:
            self.arrows.mapper.SetInputData(glyphs)

    def _joint_glyphs(self, joint: Joint, points: np.ndarray):
        """Build fixed-size arrow glyphs for a single joint from the current points.
        scale=False makes every arrow `arrow_length` long regardless of direction
        magnitude (directions are already unit vectors)."""
        bases, vectors = joint_2_arrows(joint, points)
        arrow_mesh = pv.PolyData(bases)
        arrow_mesh['vectors'] = vectors
        arrow_mesh.set_active_vectors('vectors')
        return arrow_mesh.glyph(orient='vectors', scale=False, factor=self.arrow_length)

    def update(self, state: State):
        new_points = state_2_points(state)
        self.mesh.points = new_points
        for joint, actor in zip(self.joints, self.joint_actors):
            actor.mapper.SetInputData(self._joint_glyphs(joint, new_points))
        self.plotter.update()

    def close(self):
        self.plotter.close()