"""Phase 1-1: State + Verlet integration (no constraints yet)."""

import torch

from tac.core import State, verlet_step


def test_free_points_move_at_constant_velocity_batched():
    # A free point with no acceleration must keep its velocity exactly.
    # Batched: B batches x N points, each with its own random velocity.
    B, N, dt = 3, 4, 0.1
    torch.manual_seed(0)
    v = torch.randn(B, N, 3)
    x0 = torch.randn(B, N, 3)

    # Encode the initial velocity into the previous position: (x0 - x_prev)/dt == v.
    state = State(x_current=x0.clone(), x_previous=x0 - v * dt, inv_mass=torch.ones(B, N))

    zero_accel = torch.zeros(3)
    for k in range(1, 11):
        state = verlet_step(state, dt, zero_accel)
        # position advances linearly; velocity is unchanged
        assert torch.allclose(state.x_current, x0 + v * dt * k, atol=1e-5)
        assert torch.allclose(state.velocity(dt), v, atol=1e-5)


def test_constant_acceleration_follows_discrete_verlet():
    # Starting from rest, position Verlet under constant accel does NOT give the
    # continuous 0.5*a*t^2. The exact discrete sum is a*dt^2 * (1+2+...+n) =
    # a*dt^2 * n(n+1)/2. This test documents that on purpose.
    B, N, dt = 2, 3, 0.05
    g = torch.tensor([0.0, 0.0, -9.81])
    x0 = torch.randn(B, N, 3)
    state = State.at_rest(x0)  # x_previous == x_current  ->  zero initial velocity

    n_steps = 25
    for _ in range(n_steps):
        state = verlet_step(state, dt, g)

    n = n_steps
    expected = x0 + g * (dt * dt) * (n * (n + 1) / 2)
    assert torch.allclose(state.x_current, expected, atol=1e-5)


def test_pinned_point_does_not_move_under_gravity():
    # inv_mass == 0 means pinned: it must stay put while free points fall.
    dt = 0.1
    x0 = torch.randn(1, 2, 3)
    inv_mass = torch.tensor([[0.0, 1.0]])  # point 0 pinned, point 1 free
    state = State.at_rest(x0, inv_mass)
    g = torch.tensor([0.0, 0.0, -9.81])

    for _ in range(5):
        state = verlet_step(state, dt, g)

    assert torch.allclose(state.x_current[:, 0], x0[:, 0], atol=1e-7)  # pinned unchanged
    assert state.x_current[0, 1, 2] < x0[0, 1, 2]  # free point fell (z decreased)
