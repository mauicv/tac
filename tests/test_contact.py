"""Phase 2 step 8: the asymmetric floor `Contact` constraint.

A grounded point pins in x,y; a point above the floor is free (can lift); a point falling
onto the floor rests without bouncing; and a touch-sensor override enforces contact even
when the modeled point is above the floor.
"""

import torch

from tac.constraints import Contact, solve
from tac.core import State, verlet_step

FLOOR = -0.23


def _state(xc, xp):
    xc = torch.tensor([[xc]], dtype=torch.float64)   # (1, 1, 3)
    xp = torch.tensor([[xp]], dtype=torch.float64)
    return State(xc, xp, torch.ones(1, 1, dtype=torch.float64))


def test_grounded_point_pins_xy_and_snaps_z():
    # below the floor and drifting in x,y; previous step was at (0.4, 0.1)
    st = _state([0.45, 0.13, FLOOR - 0.02], [0.40, 0.10, FLOOR + 0.01])
    out, _ = Contact(0, FLOOR).project(st, torch.zeros(1, dtype=torch.float64), dt=0.01)

    # z snapped to the floor on both current and previous -> zero z-velocity (no bounce)
    assert torch.allclose(out.x_current[0, 0, 2], torch.tensor(FLOOR, dtype=torch.float64))
    assert torch.allclose(out.x_previous[0, 0, 2], torch.tensor(FLOOR, dtype=torch.float64))
    # xy pinned back to previous -> zero in-plane velocity (no slip)
    assert torch.allclose(out.x_current[0, 0, :2], out.x_previous[0, 0, :2])
    assert torch.allclose(out.x_current[0, 0, :2], torch.tensor([0.40, 0.10], dtype=torch.float64))


def test_point_above_floor_is_free():
    st = _state([0.45, 0.13, FLOOR + 0.05], [0.40, 0.10, FLOOR + 0.06])
    out, _ = Contact(0, FLOOR).project(st, torch.zeros(1, dtype=torch.float64), dt=0.01)
    assert torch.allclose(out.x_current, st.x_current)       # untouched
    assert torch.allclose(out.x_previous, st.x_previous)


def test_grounded_point_can_lift_off():
    # at the floor with an upward implicit velocity (current above previous): not penetrating,
    # so contact leaves it free to rise.
    st = _state([0.4, 0.1, FLOOR + 0.01], [0.4, 0.1, FLOOR])
    out, _ = Contact(0, FLOOR).project(st, torch.zeros(1, dtype=torch.float64), dt=0.01)
    assert torch.allclose(out.x_current, st.x_current)
    assert torch.allclose(out.x_previous, st.x_previous)


def test_falls_and_rests_without_bouncing():
    st = _state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])            # at rest above the floor
    contact = [Contact(0, FLOOR)]
    landed = False
    max_z_after_landing = -1e9
    for _ in range(200):
        st = verlet_step(st, 0.01, (0.0, 0.0, -9.81))
        st = solve(st, contact, dt=0.01, iters=1)
        z = float(st.x_current[0, 0, 2])
        if z <= FLOOR + 1e-9:
            landed = True
        if landed:
            max_z_after_landing = max(max_z_after_landing, z)

    assert landed
    # never rises back above the floor after first contact -> no bounce
    assert max_z_after_landing < FLOOR + 1e-6
    # comes to rest essentially on the floor
    assert abs(float(st.x_current[0, 0, 2]) - FLOOR) < 1e-6
    assert abs(float(st.velocity(0.01)[0, 0, 2])) < 1e-6


def test_touch_sensor_override_enforces_contact_above_floor():
    # measured scope: the switch says grounded, so the floor is enforced even though the
    # modeled point sits above it (pulls it down to the floor, pins xy).
    st = _state([0.45, 0.13, FLOOR + 0.05], [0.40, 0.10, FLOOR + 0.06])
    active = torch.tensor([True])
    out, _ = Contact(0, FLOOR, active=active, scope="measured").project(
        st, torch.zeros(1, dtype=torch.float64), dt=0.01
    )
    assert torch.allclose(out.x_current[0, 0, 2], torch.tensor(FLOOR, dtype=torch.float64))
    assert torch.allclose(out.x_current[0, 0, :2], out.x_previous[0, 0, :2])


def test_batched_mixed_contact():
    # two batch elements: one grounded, one swinging, resolved independently in one call.
    xc = torch.tensor([[[0.45, 0.13, FLOOR - 0.02]], [[0.1, 0.2, FLOOR + 0.1]]], dtype=torch.float64)
    xp = torch.tensor([[[0.40, 0.10, FLOOR + 0.01]], [[0.1, 0.2, FLOOR + 0.12]]], dtype=torch.float64)
    st = State(xc, xp, torch.ones(2, 1, dtype=torch.float64))
    out, _ = Contact(0, FLOOR).project(st, torch.zeros(2, dtype=torch.float64), dt=0.01)

    assert torch.allclose(out.x_current[0, 0, 2], torch.tensor(FLOOR, dtype=torch.float64))  # grounded
    assert torch.allclose(out.x_current[0, 0, :2], out.x_previous[0, 0, :2])                 # pinned
    assert torch.allclose(out.x_current[1], st.x_current[1])                                 # free
