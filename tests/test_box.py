"""The free rigid box — the simplest model (PLAN Phase 1 sanity baseline).

A box is a single rigid `Weld` of 8 cube corners. Dropped from rest it is in pure
translation, so the weld is never violated and the centre falls *exactly* as a free
particle would — the engine's free-fall reference. Spun, the weld keeps it rigid. Over a
floor it lands flat and rests without sinking or bouncing.
"""

import torch

from tac.core import State, verlet_step
from tac.robots import build_box

DT = 0.01
G = (0.0, 0.0, -9.81)


def _edge_lengths(state: State, weld) -> torch.Tensor:
    """All pairwise corner distances, (B, n_pairs) — constant iff the box is rigid."""
    x = state.x_current
    pts = [int(p) for p in weld.points]
    dists = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dists.append(torch.linalg.vector_norm(x[:, pts[i]] - x[:, pts[j]], dim=-1))
    return torch.stack(dists, dim=-1)


def test_free_fall_matches_free_particle():
    # No floor: the box just falls. Its centre must track a single free particle stepped by
    # the same Verlet integrator (the weld does nothing in pure translation).
    box = build_box(size=0.2, center=(0.1, -0.2, 0.5))
    p = State.at_rest(torch.tensor([[[0.1, -0.2, 0.5]]], dtype=torch.float64))

    for _ in range(100):
        box.step(dt=DT, gravity=G, contact=False, iters=10)
        p = verlet_step(p, DT, G)

    assert torch.allclose(box.center(), p.x_current[:, 0], atol=1e-9)


def test_box_stays_rigid_while_falling():
    box = build_box(size=0.2, center=(0.0, 0.0, 0.5))
    rest = _edge_lengths(box.state, box.weld)
    for _ in range(100):
        box.step(dt=DT, gravity=G, contact=False, iters=10)
    assert torch.allclose(_edge_lengths(box.state, box.weld), rest, atol=1e-9)


def test_box_stays_rigid_while_tumbling():
    # Give the box an initial spin (rotate x_previous about the centre) so the fall is no
    # longer pure translation; the weld must hold every edge length as it tumbles.
    box = build_box(size=0.2, center=(0.0, 0.0, 0.5))
    rest = _edge_lengths(box.state, box.weld)

    x = box.state.x_current
    c = x.mean(dim=1, keepdim=True)
    ang = 0.05  # small yaw per step, baked into x_previous as implicit angular velocity
    cos, sin = torch.cos(torch.tensor(ang)), torch.sin(torch.tensor(ang))
    rot = torch.tensor(
        [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=x.dtype
    )
    x_prev = c + (x - c) @ rot.T   # previous = current rotated back -> spins forward
    box.state = State(x.clone(), x_prev, box.state.inv_mass)

    for _ in range(100):
        box.step(dt=DT, gravity=G, contact=False, iters=20)
    # rigid throughout the tumble (edges within a tight band of their rest lengths)
    assert torch.allclose(_edge_lengths(box.state, box.weld), rest, atol=1e-4)


def test_box_rests_on_floor_without_bouncing():
    # The box falls onto the floor and settles. It approaches its rest height from above and
    # asymptotes a few microns *above* the ideal (a small finite-iteration XPBD steady-state
    # offset: gravity loads the weld each step, contact pins the bottom, and the two balance
    # short of perfect convergence). So we test settling, not exact equality: it lands, comes
    # to rest at the right height, holds a small residual velocity, and never bounces back up.
    floor = -0.23
    rest = floor + 0.2 / 2.0   # centre sits half a box above the floor
    box = build_box(size=0.2, center=(0.0, 0.0, 0.4), floor_z=floor)

    settled = False
    max_z_after_settling = -1e9
    for _ in range(250):
        box.step(dt=DT, gravity=G, contact=True, iters=20)
        cz = float(box.center()[0, 2])
        if abs(cz - rest) < 1e-3:        # entered the resting band
            settled = True
        if settled:
            max_z_after_settling = max(max_z_after_settling, cz)

    assert settled
    # once resting it never climbs back up (a real bounce after a 0.27 m fall would be cm-scale)
    assert max_z_after_settling < rest + 1e-3
    # rests with its centre at the right height and only a small residual velocity
    assert abs(float(box.center()[0, 2]) - rest) < 1e-3
    assert float(box.state.velocity(DT)[0, :, 2].abs().max()) < 5e-3
    # bottom corners sit on the floor plane
    bottom_z = box.state.x_current[0, list(box.contacts), 2]
    assert torch.allclose(bottom_z, torch.full_like(bottom_z, floor), atol=1e-3)


def test_batched_boxes_fall_independently():
    box = build_box(size=0.2, center=(0.0, 0.0, 0.5), B=3)
    # offset each batch element's start so they are genuinely independent
    shift = torch.tensor([[0.0], [0.3], [-0.3]], dtype=torch.float64)  # (3, 1) -> x
    x = box.state.x_current.clone()
    x[..., 0] += shift
    box.state = State(x, x.clone(), box.state.inv_mass)

    rest = _edge_lengths(box.state, box.weld)
    for _ in range(100):
        box.step(dt=DT, gravity=G, contact=False, iters=10)

    # all three stay rigid and share the same z (same gravity, started at rest)
    assert torch.allclose(_edge_lengths(box.state, box.weld), rest, atol=1e-9)
    cz = box.center()[:, 2]
    assert torch.allclose(cz, cz[:1].expand_as(cz), atol=1e-12)
    # x positions stayed distinct (no cross-batch coupling)
    cx = box.center()[:, 0]
    assert abs(float(cx[1] - cx[0]) - 0.3) < 1e-9
