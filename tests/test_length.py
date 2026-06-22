"""Phase 1-2: length constraint + XPBD solver."""

import torch

from tac.constraints import Length, solve
from tac.core import State


def test_link_holds_length_both_free():
    # Two free points 3 apart, target length 2. A single hard distance constraint on
    # two equal-mass free points converges *exactly* in one iteration; we use several.
    x0 = torch.tensor([[[0.0, 0, 0], [3.0, 0, 0]]])  # (B=1, N=2, 3)
    state = State.at_rest(x0)
    c = Length(0, 1, rest_length=2.0)

    out = solve(state, [c], dt=0.05, iters=5)

    dist = (out.x_current[:, 0] - out.x_current[:, 1]).norm(dim=-1)
    assert torch.allclose(dist, torch.tensor([2.0]), atol=1e-6)
    # symmetric masses -> each point moved by half the error (0.5), midpoint preserved
    assert torch.allclose(out.x_current[:, 0], torch.tensor([[0.5, 0, 0]]), atol=1e-6)
    assert torch.allclose(out.x_current[:, 1], torch.tensor([[2.5, 0, 0]]), atol=1e-6)


def test_pinned_endpoint_takes_no_correction():
    # Point 0 pinned, point 1 free -> only point 1 moves to satisfy the length.
    x0 = torch.tensor([[[0.0, 0, 0], [3.0, 0, 0]]])
    inv_mass = torch.tensor([[0.0, 1.0]])
    state = State.at_rest(x0, inv_mass)
    c = Length(0, 1, rest_length=2.0)

    out = solve(state, [c], dt=0.05, iters=5)

    assert torch.allclose(out.x_current[:, 0], x0[:, 0], atol=1e-7)          # pinned
    assert torch.allclose(out.x_current[:, 1], torch.tensor([[2.0, 0, 0]]), atol=1e-6)


def test_batched_links():
    # Different starting separations per batch element, all driven to the same length.
    B = 4
    seps = torch.tensor([1.0, 2.5, 4.0, 7.0])
    x0 = torch.zeros(B, 2, 3)
    x0[:, 1, 0] = seps
    state = State.at_rest(x0)
    c = Length(0, 1, rest_length=3.0)

    out = solve(state, [c], dt=0.05, iters=5)

    dist = (out.x_current[:, 0] - out.x_current[:, 1]).norm(dim=-1)
    assert torch.allclose(dist, torch.full((B,), 3.0), atol=1e-6)


def test_gradients_flow_to_rest_length():
    # Learnable L: after solving, the link length equals L, so d(dist)/dL == 1.
    L = torch.tensor(2.0, requires_grad=True)
    x0 = torch.tensor([[[0.0, 0, 0], [3.0, 0, 0]]])
    state = State.at_rest(x0)
    c = Length(0, 1, rest_length=L)

    out = solve(state, [c], dt=0.05, iters=5)
    final_dist = (out.x_current[:, 0] - out.x_current[:, 1]).norm()
    final_dist.backward()

    assert L.grad is not None and torch.isfinite(L.grad)
    assert abs(final_dist.item() - 2.0) < 1e-6
    assert abs(L.grad.item() - 1.0) < 1e-4


def test_gradients_flow_to_positions():
    x0 = torch.tensor([[[0.0, 0, 0], [3.0, 0, 0]]], requires_grad=True)
    state = State(x_current=x0, x_previous=x0.detach().clone(), inv_mass=torch.ones(1, 2))
    c = Length(0, 1, rest_length=2.0)

    out = solve(state, [c], dt=0.05, iters=5)
    out.x_current.sum().backward()

    assert x0.grad is not None
    assert torch.isfinite(x0.grad).all()
