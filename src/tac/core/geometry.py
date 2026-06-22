"""Geometry helpers that *read* orientation off point clouds.

Orientation is never stored (see PLAN.md, "Orientation model"); it is emergent from
the positions of a weld's points. `weld_frame` is the read-off: given >=3 non-colinear
points it returns an orthonormal body frame, built by Gram-Schmidt exactly as the plan
specifies:

    e1 = normalize(p1 - p0)
    e2 = normalize((p2 - p0) - ((p2 - p0)·e1) e1)     # component of p2-p0 perp to e1
    e3 = e1 x e2

The frame is well-conditioned only if the triad spans real area: its angular resolution
is ~ (position error) / (triad size), so small triads give noisy frames (PLAN, "sizing").
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor

_EPS = 1e-9


def _normalize(v: Tensor) -> Tensor:
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(_EPS)


def weld_frame(points: Tensor) -> tuple[Tensor, Tensor]:
    """Orthonormal body frame read off the first three points of a weld.

    `points` is (B, K>=3, 3). Returns:
        R       (B, 3, 3)  rotation matrix whose *columns* are [e1 e2 e3] (body axes
                           expressed in world); maps body coords -> world directions.
        origin  (B, 3)     the reference point p0 the frame is anchored at.

    Only p0, p1, p2 are used; extra weld points constrain shape but not the frame.
    """
    if points.shape[1] < 3:
        raise ValueError(f"need >=3 points for a frame; got {points.shape[1]}")

    p0, p1, p2 = points[:, 0], points[:, 1], points[:, 2]
    e1 = _normalize(p1 - p0)
    u = p2 - p0
    e2 = _normalize(u - (u * e1).sum(-1, keepdim=True) * e1)
    e3 = torch.linalg.cross(e1, e2)

    R = torch.stack([e1, e2, e3], dim=-1)  # columns -> (B, 3, 3)
    return R, p0
