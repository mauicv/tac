"""Phase 1-5: shared-edge hinge + servo actuator.

Scene (a single world-z edge, as in the build order):
    point 0  P          = [0, 0, 0]   shared edge (axis = z), pinned
    point 1  Q          = [0, 0, 1]   shared edge,             pinned
    point 2  parent_ref = [1, 0, 0]   parent off-axis point,   pinned
    point 3  child_ref  = [1, 0, 0]   child off-axis point,    free  (the downstream subtree)

parent weld = {0,1,2}, child weld = {0,1,3}; they share the edge {0,1}. Driving the joint
rotates point 3 about z; theta reads the signed angle of the child off the parent about z.
"""

import math

import torch

from tac.constraints import Actuator, Hinge, Weld, solve
from tac.core import State, rodrigues, rotate_about_edge


def _scene(B=1):
    pts = torch.tensor(
        [[0.0, 0, 0], [0.0, 0, 1], [1.0, 0, 0], [1.0, 0, 0]]
    ).expand(B, 4, 3).clone()
    inv_mass = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(B, 4).clone()  # only child free
    state = State.at_rest(pts, inv_mass)
    parent = Weld([0, 1, 2], pts[0, [0, 1, 2]])
    child = Weld([0, 1, 3], pts[0, [0, 1, 3]])
    return state, parent, child


# --- rotation primitive ----------------------------------------------------

def test_rodrigues_rotates_basis_vectors():
    z = torch.tensor([[0.0, 0, 1]])
    R = rodrigues(z, torch.tensor([math.pi / 2]))
    x = torch.tensor([[1.0, 0, 0]])
    got = (R @ x.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(got, torch.tensor([[0.0, 1, 0]]), atol=1e-6)  # +x -> +y about +z

    y = torch.tensor([[0.0, 1, 0]])
    R = rodrigues(y, torch.tensor([math.pi / 2]))
    got = (R @ x.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(got, torch.tensor([[0.0, 0, -1]]), atol=1e-6)  # +x -> -z about +y


def test_rotate_about_offset_edge_preserves_axis_distance():
    # Edge parallel to z through (2,0,*); a point swung 180 deg lands mirrored across it.
    P = torch.tensor([[2.0, 0, 0]])
    Q = torch.tensor([[2.0, 0, 1]])
    pts = torch.tensor([[[3.0, 0, 0]]])  # (B=1, M=1, 3), 1 unit out in +x
    out = rotate_about_edge(pts, P, Q, torch.tensor([math.pi]))
    assert torch.allclose(out[0, 0], torch.tensor([1.0, 0, 0]), atol=1e-6)


# --- angle readback --------------------------------------------------------

def test_theta_zero_at_build_pose():
    state, parent, child = _scene()
    hinge = Hinge(parent, child)
    assert torch.allclose(hinge.theta(state), torch.zeros(1), atol=1e-6)


def test_theta_reads_imposed_angle():
    state, parent, child = _scene()
    hinge = Hinge(parent, child)
    for ang in (0.3, 1.0, -0.7, math.pi / 2):
        x = state.x_current.clone()
        x[:, 3] = rotate_about_edge(
            x[:, [3]], x[:, 0], x[:, 1], torch.tensor([ang])
        )[:, 0]
        moved = State(x, state.x_previous, state.inv_mass)
        assert torch.allclose(hinge.theta(moved), torch.tensor([ang]), atol=1e-6)


# --- servo actuator --------------------------------------------------------

def test_actuator_drives_downstream_to_target():
    state, parent, child = _scene()
    hinge = Hinge(parent, child)
    act = Actuator(hinge, theta_target=math.pi / 2)

    out = solve(state, [act], dt=0.02, iters=4)

    assert torch.allclose(hinge.theta(out), torch.tensor([math.pi / 2]), atol=1e-6)
    assert torch.allclose(out.x_current[:, 3], torch.tensor([[0.0, 1, 0]]), atol=1e-6)
    # the axis and the parent did not move (only the downstream subtree rotates)
    assert torch.allclose(out.x_current[:, :3], state.x_current[:, :3], atol=1e-7)


def test_actuator_reaches_target_in_one_pass_when_hard():
    state, parent, child = _scene()
    act = Actuator(Hinge(parent, child), theta_target=0.8)
    out = solve(state, [act], dt=0.02, iters=1)  # alpha = 0 -> exact in one iteration
    assert torch.allclose(act.hinge.theta(out), torch.tensor([0.8]), atol=1e-6)


def test_encoder_set_angle_reproduces_pose():
    # "encoder-set theta reproduces a pose": commanding angle phi lands the child point at
    # the analytic rotation of its build position about the axis.
    state, parent, child = _scene()
    hinge = Hinge(parent, child)
    phi = -1.1
    out = solve(state, [Actuator(hinge, theta_target=phi)], dt=0.02, iters=3)

    expected = rotate_about_edge(
        state.x_current[:, [3]], state.x_current[:, 0], state.x_current[:, 1],
        torch.tensor([phi]),
    )[:, 0]
    assert torch.allclose(out.x_current[:, 3], expected, atol=1e-6)


def test_reverse_flips_rotation_direction():
    state, parent, child = _scene()
    fwd = solve(state, [Actuator(Hinge(parent, child), theta_target=0.6)], dt=0.02, iters=3)
    rev = solve(
        state,
        [Actuator(Hinge(parent, child, reverse=True), theta_target=0.6)],
        dt=0.02, iters=3,
    )
    # same target, opposite swing: the child y-coordinate moves opposite ways
    assert fwd.x_current[0, 3, 1] > 1e-3
    assert rev.x_current[0, 3, 1] < -1e-3


def test_actuator_keeps_child_weld_rigid_alongside_rigidity_constraints():
    state, parent, child = _scene()
    hinge = Hinge(parent, child)
    cons = parent.constraints() + child.constraints() + [Actuator(hinge, theta_target=1.2)]

    out = solve(state, cons, dt=0.02, iters=6)

    # child weld distances preserved (rigid rotation about the shared edge)
    for (i, j) in ((0, 3), (1, 3)):
        d0 = (state.x_current[:, i] - state.x_current[:, j]).norm(dim=-1)
        d1 = (out.x_current[:, i] - out.x_current[:, j]).norm(dim=-1)
        assert torch.allclose(d0, d1, atol=1e-5)
    assert torch.allclose(hinge.theta(out), torch.tensor([1.2]), atol=1e-5)


def test_soft_actuator_lags_target():
    state, parent, child = _scene()
    hinge = Hinge(parent, child)
    out = solve(state, [Actuator(hinge, theta_target=1.0, alpha=5e-3)], dt=0.02, iters=1)
    th = hinge.theta(out).item()
    assert 0.0 < th < 1.0  # compliant servo reaches partway in one pass


def test_batched_independent_targets():
    state, parent, child = _scene(B=3)
    targets = torch.tensor([0.2, -0.5, 1.3])
    out = solve(state, [Actuator(Hinge(parent, child), theta_target=targets)], dt=0.02, iters=4)
    assert torch.allclose(Hinge(parent, child).theta(out), targets, atol=1e-6)


def test_gradients_flow_to_target():
    state, parent, child = _scene()
    target = torch.tensor([0.7], requires_grad=True)
    out = solve(state, [Actuator(Hinge(parent, child), theta_target=target)], dt=0.02, iters=3)
    out.x_current.sum().backward()
    assert target.grad is not None and torch.isfinite(target.grad).all()


def test_hinge_requires_exactly_two_shared_points():
    _, parent, _ = _scene()
    bad = Weld([2, 5, 6], torch.tensor([[1.0, 0, 0], [2.0, 0, 0], [2.0, 1, 0]]))
    try:
        Hinge(parent, bad)  # shares only point 2
        assert False, "expected ValueError"
    except ValueError:
        pass
