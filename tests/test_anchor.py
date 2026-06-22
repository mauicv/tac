"""Phase 1-4: anchor (pin a point to a fixed world position).

First time integration and solving are combined: each step Verlet moves the point under
gravity, then the anchor solve pulls it back. The headline test is "an anchored point
stays put under gravity".
"""

import torch

from tac.constraints import Anchor, solve
from tac.core import State, verlet_step


def test_anchored_point_stays_put_under_gravity():
    target = torch.tensor([0.0, 0.0, 1.0])
    x0 = target.reshape(1, 1, 3).clone()
    state = State.at_rest(x0)
    anchor = Anchor(0, target)  # hard pin

    for _ in range(50):
        state = verlet_step(state, dt=0.02, accel=[0.0, 0, -9.81])
        state = solve(state, [anchor], dt=0.02, iters=4)

    assert torch.allclose(state.x_current[0, 0], target, atol=1e-5)
    # velocity (x_cur - x_prev)/dt is ~0: it isn't drifting, it's held.
    assert state.velocity(0.02).abs().max() < 1e-3


def test_hard_anchor_snaps_free_point_in_one_iteration():
    target = torch.tensor([1.0, 2.0, 3.0])
    x0 = torch.zeros(1, 1, 3)
    state = State.at_rest(x0)
    out = solve(state, [Anchor(0, target)], dt=0.05, iters=1)
    assert torch.allclose(out.x_current[0, 0], target, atol=1e-6)


def test_batched_anchors_to_distinct_targets():
    B = 3
    targets = torch.tensor([[1.0, 0, 0], [0.0, 2, 0], [0.0, 0, -1]])  # (B, 3)
    state = State.at_rest(torch.zeros(B, 1, 3))
    out = solve(state, [Anchor(0, targets)], dt=0.05, iters=2)
    assert torch.allclose(out.x_current[:, 0], targets, atol=1e-6)


def test_soft_anchor_lags_under_gravity():
    # A compliant anchor (alpha > 0) is a spring: under steady gravity the point settles
    # *below* the target, held by a finite restoring pull rather than snapping back.
    target = torch.tensor([0.0, 0.0, 0.0])
    state = State.at_rest(target.reshape(1, 1, 3).clone())
    soft = Anchor(0, target, alpha=1e-3)

    for _ in range(200):
        state = verlet_step(state, dt=0.02, accel=[0.0, 0, -9.81])
        state = solve(state, [soft], dt=0.02, iters=8)

    z = state.x_current[0, 0, 2].item()
    assert z < -1e-4              # sagged below the target
    assert z > -0.2              # but still held, not falling away


def test_gradients_flow_to_target():
    target = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    state = State.at_rest(torch.zeros(1, 1, 3))
    out = solve(state, [Anchor(0, target)], dt=0.05, iters=3)
    out.x_current.sum().backward()
    assert target.grad is not None and torch.isfinite(target.grad).all()
