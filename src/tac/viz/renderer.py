from typing import List, Union
from tac.constraints.length import Length
from tac.constraints.weld import FacetedWeld
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


def length_2_cells(length: Length):
    cells = []
    cells.append(2)
    cells.extend([length.i, length.j])
    return cells

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
        for visable_constraint in visable_constraints:
            if isinstance(visable_constraint, FacetedWeld):
                face_cells.extend(weld_2_cells(visable_constraint))
            elif isinstance(visable_constraint, Length):
                line_cells.extend(length_2_cells(visable_constraint))

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

    def update(self, state: State):
        new_points = state_2_points(state)
        self.mesh.points = new_points
        self.plotter.update()

    def close(self):
        self.plotter.close()