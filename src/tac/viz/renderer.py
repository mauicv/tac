from typing import List, Union
from tac.constraints.length import Length
from tac.constraints.weld import FacetedWeld
import pyvista as pv
from tac.core.state import State
from tac.viz.convert import state_2_points, weld_2_cells
import numpy as np


class Renderer:
    def __init__(self, state: State, visable_constraints: List[Union[FacetedWeld]], camera_distance: float = 5.0):
        self.plotter = pv.Plotter()
        points = state_2_points(state)
        cells = []

        for visable_constraint in visable_constraints:
            weld_cells = weld_2_cells(visable_constraint)
            cells.extend(weld_cells)

        self.mesh = pv.PolyData(points, cells)
        self.plotter.add_mesh(self.mesh, show_edges=True, color='red')

        center = self.mesh.points.mean(axis=0)
        self.plotter.camera.position = center + np.array([camera_distance, camera_distance, camera_distance])
        self.plotter.camera.focal_point = center
        self.plotter.camera.up = [0, 0, 1]

        self.plotter.show(interactive_update=True)

    def update(self, state: State):
        new_points = state_2_points(state)
        self.mesh.points = new_points
        self.plotter.update()

    def close(self):
        self.plotter.close()