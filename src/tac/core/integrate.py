"""Verlet integration — the only place positions advance in time.

Position ("Stormer-Verlet") update, with velocity carried implicitly by x_previous:

    x_next = 2 * x_current - x_previous + a * dt**2

Because x_previous encodes velocity, a *zero* acceleration gives exact constant-velocity
motion. A constant acceleration accumulates exactly per the discrete recurrence (which is
*not* the continuous 0.5*a*t**2 when starting from rest — see the tests for why).

Pinned points (inv_mass == 0) are held fixed and their implied velocity zeroed.
"""

from __future__ import annotations

import torch

from .state import State

Tensor = torch.Tensor


def verlet_step(state: State, dt: float, accel: Tensor | list[float]) -> State:
    """Advance one timestep `dt` under acceleration `accel`, returning a new State.

    `accel` is broadcast to (B, N, 3): pass a (3,) gravity vector, or a full per-point
    field. Gravity is an *acceleration*, so it is mass-independent here; mass-dependent
    forces (later) would enter as inv_mass * force.
    """
    x_cur, x_prev = state.x_current, state.x_previous
    a = torch.as_tensor(accel, dtype=x_cur.dtype, device=x_cur.device)
    a = a.broadcast_to(x_cur.shape)

    x_next = 2.0 * x_cur - x_prev + a * (dt * dt)

    # Pinned points (inv_mass == 0) do not move; setting x_prev := x_cur zeros velocity.
    pinned = (state.inv_mass == 0).unsqueeze(-1)  # (..., N, 1), broadcasts over D
    x_next = torch.where(pinned, x_cur, x_next)

    return State(x_current=x_next, x_previous=x_cur, inv_mass=state.inv_mass)
