from tac.core.state import State
from tac.constraints.length import Length, VectorizedLength
import torch
import pytest


@pytest.mark.parametrize("batch_size", [1, 2])
def test_get_length(batch_size):
    state = State(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), batch_size=batch_size)
    assert state.x_current.shape == (batch_size, 2, 3)
    assert state.x_previous.shape == (batch_size, 2, 3)
    assert state.inv_mass.shape == (batch_size, 2)
    length = Length(state, 0, 1, 1.0, 0.1)
    assert length.L.shape == (1, 1)
    assert length.L[0, 0] == 1.0


def test_apply_length():
    state = State(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), batch_size=2)
    length = Length(state, 0, 1, 1.0, 0.1)
    state.x_current[0, 0] = torch.tensor([0.0, 0.5, 0.0])
    lam = torch.zeros(2, 1)
    nlam = length.apply(lam, 0.01)
    assert nlam.shape == (2, 1)


def test_apply_length_iterated():
    state = State(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), batch_size=2)
    length = Length(state, 0, 1, 1.0, 0.1)
    state.x_current[0, 0] = torch.tensor([0.0, 0.5, 0.0])
    state.x_current[1, 0] = torch.tensor([0.05, 0.0, 1.0])
    for _ in range(200):
        lam = torch.zeros(2, 1)
        for _ in range(10):
            lam = length.apply(lam, 0.1)

    assert torch.allclose(length.get_length(), torch.tensor([[1.0], [1.0]]), atol=1e-2)
    assert lam.shape == (2, 1)


def test_vectorized_length():
    state = State(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]), batch_size=2)
    vectorized_length = VectorizedLength(state, [0, 2], [1, 3], 1.0, 0.1)
    assert vectorized_length.L.shape == (1, 2, 1)
    assert torch.allclose(vectorized_length.L, torch.tensor([[1.0], [1.0]]), atol=1e-2)

    dist = vectorized_length.get_length()
    assert dist.shape == (2, 2, 1)
    assert torch.allclose(dist, torch.tensor([[1.0], [1.0]]), atol=1e-2)


def test_vectorized_length_apply():
    state = State(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]), batch_size=2)
    vectorized_length = VectorizedLength(state, [0, 2], [1, 3], 1.0, 0.1)
    lam = torch.zeros(2, 2, 1)
    nlam = vectorized_length.apply(lam, 0.01)
    assert nlam.shape == (2, 2, 1)


def test_vectorized_length_parity():
    b = 1
    state_1 = State(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]), batch_size=b)
    state_2 = State(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]), batch_size=b)
    vectorized_length = VectorizedLength(state_1, [0, 2], [1, 3], 1.0, 0.1)
    length_1 = Length(state_2, 0, 1, 1.0, 0.1)
    length_2 = Length(state_2, 2, 3, 1.0, 0.1)


    state_1.x_current[0, 0] = torch.tensor([0.0, 0.5, 0.0])
    state_2.x_current[0, 0] = torch.tensor([0.0, 0.5, 0.0])

    # lambda is per-constraint XPBD state: each Length keeps its own accumulator.
    # (Threading one lambda through both constraints would pollute the second
    # constraint's a_tilde*lam compliance term.)
    lam_a = length_1.apply(torch.zeros(b, 1), 0.1)
    lam_b = length_2.apply(torch.zeros(b, 1), 0.1)

    lam_v = vectorized_length.apply(torch.zeros(b, 2, 1), 0.1)

    # constraints (0,1) and (2,3) share no points, so vectorized (Jacobi) must
    # match the sequential Gauss-Seidel result exactly.
    assert torch.allclose(state_1.x_current, state_2.x_current, atol=1e-6)
    assert torch.allclose(lam_a, lam_v[:, 0], atol=1e-6)
    assert torch.allclose(lam_b, lam_v[:, 1], atol=1e-6)
    assert torch.allclose(length_1.get_length(), vectorized_length.get_length()[:, 0], atol=1e-6)
    assert torch.allclose(length_2.get_length(), vectorized_length.get_length()[:, 1], atol=1e-6)