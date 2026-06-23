from tac.core.state import State
from tac.constraints.weld import FacetedWeld
import pyvista as pv


def weld_2_cells(weld: FacetedWeld):
    faces = weld.faces
    cells = []

    for face in faces:
        cells.append(len(face))
        cells.extend(face)

    return cells


def state_2_points(state: State):
    return state.x_current[0].to('cpu').numpy()