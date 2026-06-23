import pytest
from tac.constraints.weld import FacetedWeld
from tac.viz.renderer import weld_2_cells
from tac.core.state import State
import torch

def test_weld_2_mesh():
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
    cells = weld_2_cells(weld)
    assert cells == [4, 0, 1, 2, 3]