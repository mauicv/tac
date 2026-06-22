"""Phase 1-6: the assembled 2-DOF arm (Phase 1 milestone).

The headline: driving the two joints traces the analytic two-link arc, and the velocity
read off the engine (Delta x / dt) matches the analytic forward-kinematic tip velocity.
"""

import math

import torch

from tac.robots import build_arm, fk, fk_tip_velocity
from tac.robots.arm import BASE_POINTS, ELBOW, TIP
from tac.constraints import solve


def _set_targets(arm, t1, t2):
    B = arm.state.batch
    arm.shoulder_act.theta_target = torch.full((B,), float(t1))
    arm.elbow_act.theta_target = torch.full((B,), float(t2))


def test_build_pose_is_straight():
    arm = build_arm()
    assert torch.allclose(arm.shoulder.theta(arm.state), torch.zeros(1), atol=1e-6)
    assert torch.allclose(arm.elbow.theta(arm.state), torch.zeros(1), atol=1e-6)
    assert torch.allclose(arm.tip(), torch.tensor([[arm.L1 + arm.L2, 0.0, 0.0]]), atol=1e-6)


def test_static_fk_matches_two_link_analytic():
    L1, L2 = 1.0, 0.8
    for t1, t2 in [(0.5, 0.3), (-0.7, 1.2), (1.4, -0.9), (math.pi / 2, math.pi / 2)]:
        arm = build_arm(L1, L2)
        _set_targets(arm, t1, t2)
        out = solve(arm.state, arm.constraints(), dt=0.02, iters=30)

        elbow_a, tip_a = fk(L1, L2, torch.tensor(t1), torch.tensor(t2))
        assert torch.allclose(out.x_current[0, ELBOW], elbow_a, atol=1e-5)
        assert torch.allclose(out.x_current[0, TIP], tip_a, atol=1e-5)
        # joints actually reached their commanded angles
        assert torch.allclose(arm.shoulder.theta(out), torch.tensor([t1]), atol=1e-5)
        assert torch.allclose(arm.elbow.theta(out), torch.tensor([t2]), atol=1e-5)


def test_base_stays_anchored_through_rollout():
    arm = build_arm()
    base0 = arm.state.x_current[:, list(BASE_POINTS)].clone()
    for _ in range(40):
        arm.step(torch.tensor([[0.01, -0.008]]), dt=0.02)
    assert torch.allclose(arm.state.x_current[:, list(BASE_POINTS)], base0, atol=1e-5)


def test_links_stay_rigid_through_rollout():
    arm = build_arm()
    pairs = [(0, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    d0 = {p: (arm.state.x_current[:, p[0]] - arm.state.x_current[:, p[1]]).norm() for p in pairs}
    for _ in range(40):
        arm.step(torch.tensor([[0.01, -0.008]]), dt=0.02)
    for p in pairs:
        d = (arm.state.x_current[:, p[0]] - arm.state.x_current[:, p[1]]).norm()
        assert torch.allclose(d, d0[p], atol=1e-5)


def test_tip_traces_arc_and_velocity_matches_analytic():
    # float64 so the in-plane bound reflects the method, not float32 noise.
    L1, L2 = 1.0, 0.8
    dt = 0.01
    d1, d2 = 0.004, -0.003  # per-step joint deltas (action, gain = 1)
    arm = build_arm(L1, L2, dtype=torch.float64)
    act = torch.tensor([[d1, d2]], dtype=torch.float64)
    f64 = lambda v: torch.tensor(v, dtype=torch.float64)

    prev_tip = arm.tip().clone()
    for k in range(1, 60):
        arm.step(act, dt=dt, iters=25)
        t1, t2 = float(arm.targets[0, 0]), float(arm.targets[0, 1])
        _, tip_a = fk(L1, L2, f64(t1), f64(t2))

        # (a) the tip tracks the analytic two-link arc. In-plane it is essentially exact;
        # z carries a ~7e-4 out-of-plane wobble that is independent of iteration count -- a
        # method artifact (Verlet extrapolates the rotating body off the constraint manifold,
        # and near the z=0 plane the restoring gradient is only second order in that error).
        assert torch.allclose(arm.tip()[0, :2], tip_a[:2], atol=1e-5)   # x, y
        assert abs(arm.tip()[0, 2]) < 1e-3                              # z wobble bound

        # (b) Delta x / dt matches the analytic Jacobian tip velocity (to FD truncation)
        v_meas = (arm.tip() - prev_tip)[0] / dt
        v_analytic = fk_tip_velocity(L1, L2, f64(t1), f64(t2), d1 / dt, d2 / dt)
        assert torch.allclose(v_meas, v_analytic, atol=2e-2)
        prev_tip = arm.tip().clone()


def test_batched_arms_track_independent_commands():
    B = 3
    arm = build_arm(B=B)
    arm.shoulder_act.theta_target = torch.tensor([0.3, -0.6, 1.1])
    arm.elbow_act.theta_target = torch.tensor([0.2, 0.9, -0.4])
    out = solve(arm.state, arm.constraints(), dt=0.02, iters=30)

    _, tip_a = fk(arm.L1, arm.L2, arm.shoulder_act.theta_target, arm.elbow_act.theta_target)
    assert torch.allclose(out.x_current[:, TIP], tip_a, atol=1e-5)


def test_action_gain_scales_command():
    arm = build_arm(gain=0.5)
    arm.command(torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(arm.shoulder_act.theta_target, torch.tensor([0.5]))
    assert torch.allclose(arm.elbow_act.theta_target, torch.tensor([1.0]))
