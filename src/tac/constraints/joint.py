"""Revolute joint = two welds sharing an edge, plus its servo actuator (PLAN.md "Joints").

A `Hinge` is pure geometry: the two welds share exactly **two points** (an edge), and that
edge *is* the hinge axis and pivot — nothing is stored. The hinge only needs to *read* the
joint angle off the welds' off-axis points:

    n  = (Q - P)/|Q - P|                          axis, from the two shared points
    u  = perp(parent_ref - P, n)                  parent reference, perp to axis
    v  = perp(child_ref  - P, n)                  child reference,  perp to axis
    theta = atan2(n·(u×v), u·v) - theta_build      signed angle u->v about n

The `Actuator` is the servo: an XPBD constraint `C = theta(x) - theta_target`. Each pass it
rotates the **downstream subtree** rigidly about the edge by the XPBD correction angle —
the same rotate-about-edge primitive, applied partially. Because the rotation is about the
shared edge (which the parent owns and which doesn't move), it preserves every distance
inside the subtree, so it never fights the welds' rigidity. With alpha = 0 it reaches the
target in one pass; alpha > 0 lags = servo softness.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..core.rotate import rotate_about_edge
from ..core.state import State
from .base import Constraint
from .weld import Weld

Tensor = torch.Tensor

_EPS = 1e-9


def _normalize(v: Tensor) -> Tensor:
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(_EPS)


def _perp(v: Tensor, n: Tensor) -> Tensor:
    """Component of v perpendicular to unit axis n."""
    return v - (v * n).sum(-1, keepdim=True) * n


class Hinge:
    def __init__(
        self,
        parent: Weld,
        child: Weld,
        reverse: bool = False,
        downstream: Sequence[int] | None = None,
        theta_build=0.0,
    ):
        """A revolute hinge between two welds that share an edge.

        reverse    : flip the axis orientation (P<->Q) so a positive target turns the other
                     way — convenient for mirrored/alternating joints (see gnoci_desc).
        downstream : points the actuator rotates. Defaults to the child's off-axis points;
                     pass the child's points + all descendants for a multi-link chain.
        theta_build: rest-pose offset so theta = 0 at the build pose (see `calibrate`).
        """
        self.parent = parent
        self.child = child

        shared = [p for p in parent.points if p in child.points]
        if len(shared) != 2:
            raise ValueError(
                f"a hinge needs exactly 2 shared points; welds share {len(shared)}"
            )
        P, Q = shared
        if reverse:
            P, Q = Q, P
        self.P, self.Q = P, Q
        self.reverse = reverse

        self.parent_ref = next(p for p in parent.points if p not in shared)
        self.child_ref = next(p for p in child.points if p not in shared)

        if downstream is None:
            downstream = [p for p in child.points if p not in shared]
        self.downstream = [int(p) for p in downstream]
        self.theta_build = theta_build

    def axis_points(self, state: State) -> tuple[Tensor, Tensor]:
        x = state.x_current
        return x[:, self.P], x[:, self.Q]

    def _raw_theta(self, state: State) -> Tensor:
        x = state.x_current
        P, Q = x[:, self.P], x[:, self.Q]
        n = _normalize(Q - P)
        u = _perp(x[:, self.parent_ref] - P, n)
        v = _perp(x[:, self.child_ref] - P, n)
        cross = (n * torch.linalg.cross(u, v)).sum(-1)
        dot = (u * v).sum(-1)
        return torch.atan2(cross, dot)  # (B,)

    def theta(self, state: State) -> Tensor:
        """Signed joint angle (B,), zero at the build pose.

        Wrapped into (-pi, pi] around the build pose: a straight serial chain reads its raw
        angle at +-pi (the parent reference points back toward the previous joint, opposite
        the child), i.e. right on the atan2 branch cut — wrapping keeps small bends either
        way continuous instead of flipping sign across pi.
        """
        tb = torch.as_tensor(
            self.theta_build, dtype=state.x_current.dtype, device=state.x_current.device
        )
        a = self._raw_theta(state) - tb
        return torch.atan2(torch.sin(a), torch.cos(a))

    def calibrate(self, state: State) -> "Hinge":
        """Set theta_build from a build pose so theta(build) == 0. Returns self."""
        self.theta_build = self._raw_theta(state).detach()
        return self


class Actuator(Constraint):
    def __init__(self, hinge: Hinge, theta_target=0.0, alpha: float = 0.0):
        self.hinge = hinge
        self.theta_target = theta_target
        self.alpha = float(alpha)

    def project(self, state: State, lam: Tensor, dt: float) -> tuple[State, Tensor]:
        x = state.x_current
        theta = self.hinge.theta(state)                       # (B,)
        target = torch.as_tensor(self.theta_target, dtype=x.dtype, device=x.device)
        target = target.broadcast_to(theta.shape)
        C = theta - target                                    # (B,)

        # Generalized inverse mass of the angular DOF is 1: rotating the subtree by dphi
        # changes theta by exactly dphi. (A driven servo, not a mass-weighted spring.)
        a_tilde = self.alpha / (dt * dt)
        dlam = (-C - a_tilde * lam) / (1.0 + a_tilde)         # (B,) = correction angle
        lam = lam + dlam

        P, Q = self.hinge.axis_points(state)
        ds = self.hinge.downstream
        new_x = x.clone()
        new_x[:, ds] = rotate_about_edge(x[:, ds], P, Q, dlam)

        return State(new_x, state.x_previous, state.inv_mass), lam
