from tac.core.state import State
from tac.constraints.length import Length
import torch
import pytest


@pytest.mark.parametrize("batch_size", [1, 2])
def test_get_length(batch_size):
    state = State(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), batch_size=batch_size)
    assert state.x_current.shape == (batch_size, 2, 3)
    assert state.x_previous.shape == (batch_size, 2, 3)
    assert state.inv_mass.shape == (batch_size, 2)
    length = Length(state, 0, 1, 1.0)
    assert length.L.shape == (1, 1)
    assert length.L[0, 0] == 1.0


def test_apply_length():
    state = State(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), batch_size=2)
    length = Length(state, 0, 1, 1.0)
    state.x_current[0, 0] = torch.tensor([0.0, 0.5, 0.0])
    lam = torch.zeros(2, 1)
    nlam = length.apply(lam, 0.01)
    assert nlam.shape == (2, 1)


def test_apply_length_iterated():
    state = State(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), batch_size=2)
    length = Length(state, 0, 1, 1.0)
    state.x_current[0, 0] = torch.tensor([0.0, 0.5, 0.0])
    state.x_current[1, 0] = torch.tensor([0.05, 0.0, 1.0])
    for _ in range(200):
        lam = torch.zeros(2, 1)
        for _ in range(10):
            lam = length.apply(lam, 0.1)

    assert torch.allclose(length.get_length(), torch.tensor([[1.0], [1.0]]), atol=1e-2)
    assert lam.shape == (2, 1)
