"""XPBD constraint solver.

Gauss-Seidel: each timestep, reset every constraint's Lagrange multiplier, then sweep
the constraint list `iters` times, applying each constraint to the running state (so
later constraints see earlier corrections). Constraint order matters; for now it is just
the list order (explicit ordering / scopes come in later phases).
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.state import State
from .base import Constraint


def solve(state: State, constraints: Sequence[Constraint], dt: float, iters: int) -> State:
    lams = [c.init_lambda(state) for c in constraints]  # reset multipliers each solve
    for _ in range(iters):
        for i, c in enumerate(constraints):
            state, lams[i] = c.project(state, lams[i], dt)
    return state
