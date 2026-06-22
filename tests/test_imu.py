"""Phase 2 step 7: the `TiltImu` constraint.

Imposed pitch/roll orients the torso weld (its emergent frame stands the measured "up"
vector vertical), the weld stays rigid through the correction, and yaw is left free — a
purely-yawed (level) body is untouched.
"""

import math

import torch

from tac.constraints import TiltImu, Weld, solve, up_body_from_pitch_roll
from tac.core import State
from tac.core.rotate import rotate_about_edge


def _torso():
    """A small torso triad: e1=+x (forward), e2=+y, e3=+z (up) at the level pose."""
    pts = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=torch.float64)
    weld = Weld([0, 1, 2], pts)
    state = State.at_rest(pts.unsqueeze(0).clone())
    return weld, state, pts


def _world_up(B):
    u = torch.zeros(B, 3, dtype=torch.float64)
    u[:, 2] = 1.0
    return u


def test_imposed_pitch_roll_orients_the_torso():
    weld, state, _ = _torso()
    for p, r in [(0.26, 0.0), (0.0, -0.2), (0.3, 0.15), (-0.4, 0.25)]:
        _, st, _ = _torso()
        up_body = up_body_from_pitch_roll(torch.tensor(p, dtype=torch.float64),
                                          torch.tensor(r, dtype=torch.float64)).unsqueeze(0)
        imu = TiltImu(weld, at=0, up_body=up_body, downstream=[0, 1, 2])
        out = solve(st, weld.constraints() + [imu], dt=0.02, iters=20)

        # the corrected frame stands the measured "up" exactly vertical
        R, _ = weld.frame(out)
        aligned = torch.einsum("bij,bj->bi", R, up_body)
        assert torch.allclose(aligned, _world_up(1), atol=1e-7)


def test_correction_keeps_the_weld_rigid():
    weld, state, _ = _torso()
    pairs = [(0, 1), (0, 2), (1, 2)]
    d0 = {q: (state.x_current[:, q[0]] - state.x_current[:, q[1]]).norm() for q in pairs}
    up_body = up_body_from_pitch_roll(torch.tensor(0.35, dtype=torch.float64),
                                      torch.tensor(0.2, dtype=torch.float64)).unsqueeze(0)
    imu = TiltImu(weld, at=0, up_body=up_body, downstream=[0, 1, 2])
    out = solve(state, weld.constraints() + [imu], dt=0.02, iters=20)
    for q in pairs:
        d = (out.x_current[:, q[0]] - out.x_current[:, q[1]]).norm()
        assert torch.allclose(d, d0[q], atol=1e-9)


def test_pure_yaw_is_left_free():
    # A level body rotated only about vertical: up is already vertical, so the tilt error is
    # zero and the constraint must do nothing -- heading (yaw) is untouched.
    weld, state, pts = _torso()
    yaw = 0.7
    axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)  # vertical
    origin = torch.zeros(1, 3, dtype=torch.float64)
    yawed = rotate_about_edge(pts.unsqueeze(0), origin, origin + axis, torch.tensor([yaw]))
    st = State.at_rest(yawed.clone())

    imu = TiltImu(weld, at=0, up_body=_world_up(1), downstream=[0, 1, 2])
    out = solve(st, weld.constraints() + [imu], dt=0.02, iters=20)
    assert torch.allclose(out.x_current, yawed, atol=1e-7)

    # forward axis still points along the imposed heading (yaw preserved, not zeroed)
    R, _ = weld.frame(out)
    fwd = R[0, :, 0]  # e1 = forward
    assert math.isclose(float(torch.atan2(fwd[1], fwd[0])), yaw, abs_tol=1e-7)


def test_correction_axis_is_horizontal_so_no_yaw_is_injected():
    # Direct check of the yaw-free property: start yawed *and* pitched; one hard pass aligns
    # tilt, and because the rotation axis is u_w x v (perp to vertical) the heading is held.
    weld, state, pts = _torso()
    yaw = 0.5
    axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    origin = torch.zeros(1, 3, dtype=torch.float64)
    yawed = rotate_about_edge(pts.unsqueeze(0), origin, origin + axis, torch.tensor([yaw]))
    st = State.at_rest(yawed.clone())

    up_body = up_body_from_pitch_roll(torch.tensor(0.3, dtype=torch.float64),
                                      torch.tensor(0.0, dtype=torch.float64)).unsqueeze(0)
    imu = TiltImu(weld, at=0, up_body=up_body, downstream=[0, 1, 2])
    out = solve(st, weld.constraints() + [imu], dt=0.02, iters=20)

    R, _ = weld.frame(out)
    aligned = torch.einsum("bij,bj->bi", R, up_body)
    assert torch.allclose(aligned, _world_up(1), atol=1e-7)        # tilt fixed
    # heading: project the (now tilted) forward axis onto the horizontal plane
    fwd = R[0, :, 0]
    assert math.isclose(float(torch.atan2(fwd[1], fwd[0])), yaw, abs_tol=1e-6)
