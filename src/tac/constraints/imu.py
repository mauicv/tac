"""IMU orientation correction on the torso weld (PLAN.md, Decision 11 + Note on `imu`).

An IMU sits on the torso weld at a point. The accelerometer (gravity-fused with the gyro)
measures **which way is up, expressed in the body frame** — a unit vector `up_body`. That
vector carries 2 DOF, and those 2 DOF *are* pitch & roll; the third orientation DOF (yaw,
heading about vertical) is unobservable from accel+gyro and is left free.

`TiltImu` enforces the tilt by **aligning vectors**, not by decomposing Euler angles:

    R, _ = weld.frame(state)          # current body->world orientation (emergent)
    v    = R @ up_body                # the measured "up", carried into world by current R
    # rotate the whole subtree about the IMU point so v lands on world-up (0,0,1):
    axis  = normalize(u_w x v)
    theta = atan2(|u_w x v|, u_w . v) # the tilt error
    rotate_about(subtree, pivot=imu_pt, axis, by -theta)   # XPBD-damped

Because the rotation axis is `u_w x v`, it is always perpendicular to vertical, so the
correction is **pure tilt and injects zero yaw** — "pitch & roll, yaw free" falls straight
out of the geometry, with no Euler convention and no gimbal singularity. Yaw (from a future
magnetometer) will be a separate `HeadingImu` rotating about the *vertical* axis.

Measured scope: the whole robot hangs off the torso, so `downstream` is the torso weld plus
everything below it — the correction re-orients the entire body rigidly about the IMU point,
preserving every joint angle and link length (same property as the actuator's rotate-about-
edge). Position of the floating base is fixed separately by contacts.
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


def up_body_from_pitch_roll(pitch, roll) -> Tensor:
    """World-up expressed in the body frame for a torso at (pitch, roll), yaw-independent.

    ZYX convention (R = Rz(yaw)·Ry(pitch)·Rx(roll)); up_body = Rᵀ·(0,0,1) drops the yaw
    (a rotation about vertical leaves vertical fixed), giving a pure tilt vector:

        up_body = (-sin p,  cos p·sin r,  cos p·cos r)

    Handy for building IMU measurements / tests from angles; the constraint itself only ever
    sees the vector.
    """
    pitch = torch.as_tensor(pitch)
    roll = torch.as_tensor(roll, dtype=pitch.dtype)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    return torch.stack([-sp, cp * sr, cp * cr], dim=-1)


class TiltImu(Constraint):
    scope = "measured"

    def __init__(
        self,
        torso: Weld,
        at: int,
        up_body,
        downstream: Sequence[int],
        alpha: float = 0.0,
    ):
        """Tilt (pitch+roll) correction from an IMU on the torso weld.

        torso      : the weld whose emergent frame is corrected.
        at         : point index the IMU sits on; the rotation pivots about it (stays fixed).
        up_body    : measured world-up in body coords, (3,) or (B, 3). The live measurement
                     (set per step in estimation); normalized internally.
        downstream : points to rotate — the torso weld + everything below it (measured scope).
        alpha      : compliance; 0.0 = align hard each pass, >0 = soft sensor trust (Note 2).
        """
        self.torso = torso
        self.at = int(at)
        self.up_body = up_body
        self.downstream = [int(p) for p in downstream]
        self.alpha = float(alpha)

    def project(self, state: State, lam: Tensor, dt: float) -> tuple[State, Tensor]:
        x = state.x_current
        B = state.batch

        R, _ = self.torso.frame(state)                         # (B, 3, 3) body->world
        ub = torch.as_tensor(self.up_body, dtype=x.dtype, device=x.device)
        if ub.ndim == 1:
            ub = ub.expand(B, 3)
        ub = ub / torch.linalg.vector_norm(ub, dim=-1, keepdim=True).clamp_min(_EPS)

        v = torch.einsum("bij,bj->bi", R, ub)                  # measured up, in world
        u_w = torch.zeros_like(v)
        u_w[:, 2] = 1.0

        cross = torch.linalg.cross(u_w, v)                     # rotation axis (⊥ vertical)
        s = torch.linalg.vector_norm(cross, dim=-1)            # (B,) = sin(tilt)
        c = (u_w * v).sum(-1)                                  # (B,) = cos(tilt)
        theta = torch.atan2(s, c)                              # (B,) tilt error >= 0

        # axis = cross / s; fall back to a horizontal axis only in the (non-physical for a
        # standing torso) degenerate v ~ ±u_w case, where theta is ~0 or ~pi anyway.
        axis = cross / s.clamp_min(_EPS).unsqueeze(-1)
        fallback = torch.zeros_like(axis)
        fallback[:, 0] = 1.0
        axis = torch.where((s > _EPS).unsqueeze(-1), axis, fallback)

        # XPBD: drive C = theta -> 0. Rotating the rigid subtree about `axis` by d turns v
        # toward u_w one-for-one, so the generalized inverse mass of this DOF is 1 (as the
        # actuator). dlam (= -theta at the first hard pass) is the correction angle.
        a_tilde = self.alpha / (dt * dt)
        dlam = (-theta - a_tilde * lam) / (1.0 + a_tilde)
        lam = lam + dlam

        P = x[:, self.at]                                      # pivot: the IMU point
        Q = P + axis                                           # P->Q is the rotation axis
        new_x = x.clone()
        new_x[:, self.downstream] = rotate_about_edge(
            x[:, self.downstream], P, Q, dlam
        )
        return State(new_x, state.x_previous, state.inv_mass), lam
