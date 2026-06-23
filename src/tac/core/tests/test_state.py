import torch
from tac.core.state import State
import pytest


@pytest.mark.parametrize("batch_size", [1, 2])
def test_state(batch_size):
    state = State(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), batch_size=batch_size)
    assert state.x_current.shape == (batch_size, 2, 3)
    assert state.x_previous.shape == (batch_size, 2, 3)
    assert state.inv_mass.shape == (batch_size, 2)
