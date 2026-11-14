"""Geometry helpers for turning mesh vertex clouds into physics points."""
from dataclasses import dataclass
import itertools

import numpy as np


# corner i has sign pattern _SIGNS[i] along (axis0, axis1, axis2)
_SIGNS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))   # (8, 3)


def _box_faces() -> list[list[int]]:
    index = {tuple(s): i for i, s in enumerate(_SIGNS)}
    faces = []
    for k in range(3):
        a, b = (k + 1) % 3, (k + 2) % 3          # (a, b, k) cyclic -> CCW about +k
        for s in (-1.0, 1.0):
            loop = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            if s < 0:
                loop = loop[::-1]                # outward normal is -k: reverse winding
            face = []
            for sa, sb in loop:
                sign = [0.0, 0.0, 0.0]
                sign[a], sign[b], sign[k] = sa, sb, s
                face.append(index[tuple(sign)])
            faces.append(face)
    return faces


_FACES = _box_faces()


@dataclass
class OrientedBox:
    center: np.ndarray    # (3,) world frame
    axes: np.ndarray      # (3, 3) rows = principal axes, descending variance, right-handed
    half: np.ndarray      # (3,) half-extents along each axis

    @property
    def corners(self) -> np.ndarray:
        """(8, 3) world-frame corners; corner i = center + Σ_k sign[i,k]·half[k]·axes[k]."""
        return self.center + (_SIGNS * self.half) @ self.axes

    def extent_along(self, direction: np.ndarray) -> float:
        """Half-extent of the box along a unit `direction` (support distance from center)."""
        return float(np.sum(self.half * np.abs(self.axes @ direction)))

    @property
    def faces(self) -> list[list[int]]:
        """6 corner-index quads, wound counter-clockwise seen from outside (normals out)."""
        return _FACES

    @property
    def edges(self) -> list[tuple[int, int]]:
        """12 corner-index pairs differing in exactly one sign (for rendering)."""
        return [(i, j) for i, j in itertools.combinations(range(8), 2)
                if np.sum(_SIGNS[i] != _SIGNS[j]) == 1]


def principal_axes(verts: np.ndarray) -> np.ndarray:
    """(V, 3) -> (3, 3) rows = principal axes, descending variance, right-handed."""
    centered = verts - verts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)   # rows of vt = axes
    if np.linalg.det(vt) < 0:
        vt[2] = -vt[2]
    return vt


def oriented_box(verts: np.ndarray, min_half: float = 0.003) -> OrientedBox:
    """PCA-aligned bounding box of a (V, 3) vertex cloud.

    Contains every vertex. Half-extents are clamped to at least `min_half` so thin
    parts don't produce a (near-)degenerate box.
    """
    axes = principal_axes(verts)
    proj = verts @ axes.T                       # (V, 3) coords in the PCA frame
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    half = np.maximum((hi - lo) / 2, min_half)
    center = ((lo + hi) / 2) @ axes             # back to world frame
    return OrientedBox(center=center, axes=axes, half=half)


def patch_corners(points: np.ndarray, max_points: int = 4) -> np.ndarray:
    """Reduce a near-flat (P, 3) contact patch to at most 4 of its own points.

    Picks the extreme points along the patch's diagonals (±u ± v, u/v = in-plane xy
    principal axes), i.e. its corners. Returns the patch unchanged if it is small enough.
    """
    if len(points) <= max_points:
        return points
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    _, _, vt = np.linalg.svd(xy, full_matrices=False)       # (2, 2) in-plane axes
    uv = xy @ vt.T                                          # (P, 2)
    diag = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
    idx = list(dict.fromkeys(int(np.argmax(uv @ d)) for d in diag))  # unique, in order
    return points[idx]
