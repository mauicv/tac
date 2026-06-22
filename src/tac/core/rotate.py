"""Rotation primitive — "rotate a subtree's points about a world edge".

The one rotational operation the engine needs (PLAN.md, Note 4). A revolute joint's axis
is the *shared edge* between two welds; the actuator rotates the downstream subtree about
that edge. Everything stays positions-only: we build a rotation matrix and apply it, never
storing an orientation.

`rodrigues(axis, angle)` gives the rotation matrix about a unit (or to-be-normalized) axis:
    R = I + sin(θ) K + (1 - cos(θ)) K²,   K = skew(axis)
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor

_EPS = 1e-9


def rodrigues(axis: Tensor, angle: Tensor) -> Tensor:
    """Rotation matrices (B, 3, 3) about `axis` (B, 3) by `angle` (B,), right-hand rule.

    `axis` is normalized internally; `angle` is in radians.
    """
    k = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(_EPS)
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    zero = torch.zeros_like(kx)
    K = torch.stack(
        [
            torch.stack([zero, -kz, ky], dim=-1),
            torch.stack([kz, zero, -kx], dim=-1),
            torch.stack([-ky, kx, zero], dim=-1),
        ],
        dim=-2,
    )  # (B, 3, 3) skew-symmetric

    angle = angle.reshape(-1, 1, 1)
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).expand_as(K)
    return eye + torch.sin(angle) * K + (1.0 - torch.cos(angle)) * (K @ K)


def rotate_about_edge(points: Tensor, P: Tensor, Q: Tensor, angle: Tensor) -> Tensor:
    """Rotate `points` (B, M, 3) about the line P->Q by `angle` (B,), returning new points.

    P, Q are (B, 3) endpoints of the edge (a point on the axis + its direction). Distances
    to any point on the axis are preserved, so a rigid cluster sharing the edge stays rigid.
    """
    R = rodrigues(Q - P, angle)                 # (B, 3, 3)
    rel = points - P.unsqueeze(1)               # (B, M, 3), relative to a point on the axis
    rel_rot = torch.einsum("bij,bmj->bmi", R, rel)
    return P.unsqueeze(1) + rel_rot
