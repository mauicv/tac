"""Phase 1-3: weld (rigid cluster) + emergent frame readback.

Two things under test:
  * rigidity  — the pairwise lengths hold the cluster's shape (at rest and under motion).
  * frame     — weld_frame reads a stable, orthonormal body frame off the points, and that
                frame transforms exactly as the body does (translation: unchanged;
                rotation: the same rotation; stays valid while the body moves).
"""

import math

import torch

from tac.constraints import Weld, solve
from tac.core import State, verlet_step, weld_frame


def _triad(B=1):
    # right-triangle triad spanning real area; rest frame == identity.
    pts = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])  # (3, 3)
    return pts.expand(B, 3, 3).clone()


def _rot_xyz(rx, ry, rz):
    """Proper rotation R = Rz @ Ry @ Rx (radians)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _dists(x, points):
    out = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            out.append((x[:, points[i]] - x[:, points[j]]).norm(dim=-1))
    return torch.stack(out, dim=-1)  # (B, n_pairs)


# --- rigidity --------------------------------------------------------------

def test_weld_pulls_perturbed_point_back_to_rest_shape():
    rest = _triad()[0]  # (3, 3)
    x0 = _triad()
    x0[:, 2] = torch.tensor([0.3, 1.4, 0.2])  # yank point 2 off its rest location
    state = State.at_rest(x0)
    weld = Weld([0, 1, 2], rest)

    rest_d = _dists(rest.unsqueeze(0), [0, 1, 2])
    assert not torch.allclose(_dists(x0, [0, 1, 2]), rest_d, atol=1e-3)  # starts wrong

    out = solve(state, weld.constraints(), dt=0.05, iters=20)
    assert torch.allclose(_dists(out.x_current, [0, 1, 2]), rest_d, atol=1e-5)


def test_rigid_and_frame_valid_under_spin():
    # Spin the triad about z (no gravity). x_previous is the triad rotated *back* by a
    # small angle, so Verlet keeps rotating it forward; the weld must hold the shape.
    B = 1
    dth = 0.05
    x_cur = _triad(B)
    centroid = x_cur.mean(dim=1, keepdim=True)
    Rback = _rot_xyz(0, 0, -dth)
    x_prev = (x_cur - centroid) @ Rback.T + centroid
    state = State(x_current=x_cur, x_previous=x_prev, inv_mass=torch.ones(B, 3))

    weld = Weld([0, 1, 2], _triad(B)[0])
    rest_d = _dists(_triad(B), [0, 1, 2])

    prev_R = None
    moved = False
    for _ in range(20):
        state = verlet_step(state, dt=1.0, accel=[0.0, 0, 0])
        state = solve(state, weld.constraints(), dt=1.0, iters=10)

        # rigid: pairwise distances stay at the rest shape every step
        assert torch.allclose(_dists(state.x_current, [0, 1, 2]), rest_d, atol=1e-4)

        # frame stays a valid rotation (orthonormal, det +1) while the body moves
        R, _ = weld.frame(state)
        eye = torch.eye(3).expand(B, 3, 3)
        assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-4)
        assert torch.allclose(torch.linalg.det(R), torch.ones(B), atol=1e-4)

        if prev_R is not None and not torch.allclose(R, prev_R, atol=1e-3):
            moved = True
        prev_R = R
    assert moved  # the frame actually rotated (it was "stable under motion", not static)


def test_free_fall_translates_without_rotating_the_frame():
    # Pure gravity = uniform translation: no torque, so the read-off frame is unchanged.
    state = State.at_rest(_triad())
    weld = Weld([0, 1, 2], _triad()[0])
    R0, _ = weld.frame(state)

    for _ in range(15):
        state = verlet_step(state, dt=0.02, accel=[0.0, 0, -9.81])
        state = solve(state, weld.constraints(), dt=0.02, iters=8)

    assert state.x_current[:, 0, 2] < -1e-3  # it really fell
    R1, origin1 = weld.frame(state)
    assert torch.allclose(R1, R0, atol=1e-5)  # ...but did not rotate


# --- frame readback (helper) ----------------------------------------------

def test_frame_is_orthonormal():
    R, origin = weld_frame(_triad())
    assert torch.allclose(R, torch.eye(3).unsqueeze(0), atol=1e-6)  # this triad -> identity
    assert torch.allclose(origin, torch.zeros(1, 3), atol=1e-6)


def test_frame_recovers_applied_rotation():
    # Rotate the whole triad by a known R; the frame must rotate by exactly that R.
    Rot = _rot_xyz(-0.3, 0.7, 0.4)
    x = _triad()
    R0, _ = weld_frame(x)
    x_rot = x @ Rot.T  # apply Rot to every point (row vectors)
    R1, _ = weld_frame(x_rot)

    recovered = R1 @ R0.transpose(-1, -2)  # R1 = Rot @ R0  =>  Rot = R1 @ R0^T
    assert torch.allclose(recovered[0], Rot, atol=1e-5)


def test_frame_unchanged_by_translation():
    x = _triad()
    R0, o0 = weld_frame(x)
    t = torch.tensor([3.0, -2.0, 5.0])
    R1, o1 = weld_frame(x + t)
    assert torch.allclose(R1, R0, atol=1e-6)
    assert torch.allclose(o1, o0 + t, atol=1e-6)


def test_frame_batched_independent_rotations():
    base = _triad(2)
    Ra, Rb = _rot_xyz(0.2, 0, 0), _rot_xyz(0, 0, 1.1)
    base[0] = base[0] @ Ra.T
    base[1] = base[1] @ Rb.T
    R, _ = weld_frame(base)
    assert torch.allclose(R[0], Ra, atol=1e-5)
    assert torch.allclose(R[1], Rb, atol=1e-5)


def test_frame_gradients_flow():
    x = _triad().requires_grad_(True)
    R, _ = weld_frame(x)
    R.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
