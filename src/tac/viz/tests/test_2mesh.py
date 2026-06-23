import pytest
from tac.constraints.weld import FacetedWeld
from tac.viz.convert import weld_2_mesh
from tac.core.state import State
import torch

def test_weld_2_mesh(batch_size):
    state = State(
        torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]),
        inv_mass=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        batch_size=1
    )
    weld = FacetedWeld(state, [[0, 1, 2, 3]], 1.0)
    mesh = weld_2_mesh(weld)
    mesh.plot()
    # print(mesh)