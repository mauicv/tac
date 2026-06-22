"""Phase 2 step 9: the full gnoci biped, built by hand.

Assembles (11 welds, 10 shared-edge hinges, mirrored), holds its static build pose, and —
the headline the 2-DOF arm could not exercise — the **stacked orthogonal hip** (yaw⊥roll⊥pitch)
resolves to independent target angles with the legs staying rigid (twist included).
"""

import torch

from tac.constraints import solve
from tac.robots import build_gnoci


def _weld_pairs(g):
    pairs = []
    for w in g.welds:
        ps = w.points
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                pairs.append((ps[i], ps[j]))
    return pairs


def _rest_lengths(state, pairs):
    x = state.x_current
    return {p: (x[:, p[0]] - x[:, p[1]]).norm(dim=-1) for p in pairs}


def test_assembles_and_calibrates_to_zero():
    g = build_gnoci()
    assert g.state.num_points == 25
    assert len(g.welds) == 11 and len(g.hinges) == 10 and len(g.actuators) == 10
    assert len(g.contacts) == 4
    # every weld carries a frame (>=3 points); every hinge found exactly 2 shared points
    # (enforced in the Weld/Hinge constructors — a clean build is the assertion).
    assert all(len(w.points) >= 3 for w in g.welds)
    assert torch.allclose(g.joint_angles(), torch.zeros(1, 10, dtype=torch.float64), atol=1e-9)


def test_holds_static_pose():
    # gravity off: the build pose satisfies every weld + servo (angles = targets = 0), so the
    # modeled solve is a fixed point — nothing drifts and the body stays rigid.
    g = build_gnoci()
    x0 = g.state.x_current.clone()
    out = solve(g.state, g.modeled_constraints(), dt=0.02, iters=20)
    assert torch.allclose(out.x_current, x0, atol=1e-7)


def test_stacked_orthogonal_hip_resolves():
    # Drive the LEFT hip's three orthogonal hinges (yaw‖z, roll‖x, pitch‖y) to independent
    # targets at once; each must reach its target and the leg stays rigid. This is the
    # non-parallel stacked-hinge case the arm (parallel ẑ only) could not test.
    g = build_gnoci()
    pairs = _weld_pairs(g)
    d0 = _rest_lengths(g.state, pairs)

    yaw, roll, pitch = 0.3, -0.25, 0.4
    g.actuators[0].theta_target = torch.full((1,), yaw, dtype=torch.float64)   # left yaw
    g.actuators[1].theta_target = torch.full((1,), roll, dtype=torch.float64)  # left roll
    g.actuators[2].theta_target = torch.full((1,), pitch, dtype=torch.float64) # left pitch
    out = solve(g.state, g.modeled_constraints(), dt=0.02, iters=60)
    g.state = out

    ang = g.joint_angles()[0]
    assert abs(float(ang[0]) - yaw) < 1e-4
    assert abs(float(ang[1]) - roll) < 1e-4
    assert abs(float(ang[2]) - pitch) < 1e-4
    # the other 7 joints stayed at 0 (no cross-talk through the chain)
    assert torch.allclose(ang[3:], torch.zeros(7, dtype=torch.float64), atol=1e-4)
    # legs stayed rigid through the orthogonal rotations
    d1 = _rest_lengths(out, pairs)
    for p in pairs:
        assert torch.allclose(d1[p], d0[p], atol=1e-6)


def test_full_leg_pose_reaches_all_five_joints():
    g = build_gnoci()
    targets = [0.2, -0.15, 0.5, -0.7, 0.3]  # yaw, roll, pitch, knee, ankle (left)
    for j, t in enumerate(targets):
        g.actuators[j].theta_target = torch.full((1,), t, dtype=torch.float64)
    out = solve(g.state, g.modeled_constraints(), dt=0.02, iters=80)

    g.state = out
    ang = g.joint_angles()[0]
    for j, t in enumerate(targets):
        assert abs(float(ang[j]) - t) < 1e-4


def test_batched_legs_track_independent_targets():
    B = 4
    g = build_gnoci(B=B)
    # different left-knee target per batch element
    knee = torch.tensor([-0.2, -0.6, -1.0, -1.4], dtype=torch.float64)
    g.actuators[3].theta_target = knee
    out = solve(g.state, g.modeled_constraints(), dt=0.02, iters=60)
    g.state = out
    assert torch.allclose(g.joint_angles()[:, 3], knee, atol=1e-4)
