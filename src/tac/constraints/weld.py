"""Weld — the rigid-body primitive (PLAN.md "The core abstraction").

A weld is a *set of points held mutually rigid* by a `length` constraint between every
pair. With >=3 non-colinear points the cluster moves and turns as one body and carries a
full orientation, read off its points by `weld_frame` (nothing is stored).

It is not itself a `Constraint`: it *expands* into pairwise `Length` constraints (n points
-> n(n-1)/2 of them) that the solver projects. Keep welds <= ~5 points so the pairwise
count stays small (PLAN, Note 9); a single shape-matching constraint is the escape hatch
for larger clusters, not built yet.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..core.geometry import weld_frame
from ..core.state import State
from .length import Length

Tensor = torch.Tensor


class Weld:
    def __init__(self, points: Sequence[int], rest: Tensor, alpha: float = 0.0):
        """A rigid cluster.

        points : K global point indices (K >= 3) into the State's point array.
        rest   : (K, 3) rest positions of *these* K points, in the same order as
                 `points`. Pairwise distances here become the `length` rest lengths.
        alpha  : compliance shared by the pairwise lengths (0.0 = perfectly rigid).
        """
        self.points = [int(p) for p in points]
        if len(self.points) < 3:
            raise ValueError("a weld needs >= 3 points to carry orientation")

        self.rest = torch.as_tensor(rest)
        if self.rest.shape != (len(self.points), 3):
            raise ValueError(
                f"rest must be (K, 3) with K = {len(self.points)}; got "
                f"{tuple(self.rest.shape)}"
            )
        self.alpha = float(alpha)

    def constraints(self) -> list[Length]:
        """Expand to one `Length` per unordered pair of points."""
        cons: list[Length] = []
        for i in range(len(self.points)):
            for j in range(i + 1, len(self.points)):
                rest_len = torch.linalg.vector_norm(self.rest[i] - self.rest[j])
                cons.append(
                    Length(self.points[i], self.points[j], rest_len, alpha=self.alpha)
                )
        return cons

    def frame(self, state: State) -> tuple[Tensor, Tensor]:
        """Read the emergent (R, origin) frame off the current positions."""
        pts = state.x_current[:, self.points[:3]]  # (B, 3, 3)
        return weld_frame(pts)
