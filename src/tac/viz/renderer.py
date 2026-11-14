from typing import List, Union, Tuple
from tac.constraints.autograd import AutogradConstraint
import pyvista as pv
from tac.core.state import State
import numpy as np


def state_2_points(state: State):
    return np.asarray(state.x_current[0])


def lines_2_cells(lines: List[Tuple[int, int]]):
    cells = []
    for line in lines:
        cells.append(2)
        cells.extend(line)
    return cells

def faces_2_cells(faces: List[List[int]]):
    cells = []
    for face in faces:
        cells.append(len(face))
        cells.extend(face)
    return cells


class Renderer:
    def __init__(
            self,
            state: State,
            lines: List[Tuple[int, int]],
            faces: List[List[int]],
            camera_distance: float = 5.0,
            opacity: float = 1.0,
        ):
        self.plotter = pv.Plotter()
        points = state_2_points(state)
        lines_cells = lines_2_cells(lines)
        faces_cells = faces_2_cells(faces)

        self.mesh = pv.PolyData(
            points,
            faces=faces_cells or None,
            lines=lines_cells or None,
        )
        self.plotter.add_mesh(
            self.mesh,
            line_width=5,
            show_edges=True,
            color='gray',
            opacity=opacity,
            # style='wireframe'
        )

        center = self.mesh.points.mean(axis=0)
        self.plotter.camera.position = center + np.array([camera_distance, camera_distance, camera_distance])
        self.plotter.camera.focal_point = center
        self.plotter.camera.up = [0, 0, 1]
        self.plotter.show(interactive_update=True)
        self.arrows = None

    def update(self, state: State):
        new_points = state_2_points(state)
        self.mesh.points = new_points
        self.plotter.update()

    def close(self):
        self.plotter.close()