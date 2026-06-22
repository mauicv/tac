"""Render two boxes joined at a single edge by a hinge + servo, falling under gravity.

    python examples/render_hinged_boxes.py            # pops the native viewer
    python examples/render_hinged_boxes.py out.rrd    # logs headlessly (open: `rerun out.rrd`)

Builds on the single box (render_box.py): two cubes share exactly one edge (the red hinge
axis); box B swings about it, driven by a servo. Here the pair is **unanchored** and dropped
from a height — both boxes fall together under gravity, box B still articulating on the hinge,
and **every point is a floor contact** (orange), so the assembly lands on whatever corner /
edge / face hits the grey floor first and comes to rest without bouncing. Both welds (blue
edges) stay rigid throughout.
"""

import math
import sys

import torch

from tac.robots import build_hinged_boxes
from tac.viz import Renderer, scene_from_hinged_boxes

FLOOR_Z = -0.23


def main(save: str | None = None) -> None:
    hb = build_hinged_boxes(size=0.2, center=(0.0, 0.0, 0.6), anchor=False, floor_z=FLOOR_Z)
    scene = scene_from_hinged_boxes(hb)

    dt = 0.01
    horizon = 300
    states = [hb.state]
    target = torch.zeros(1, dtype=torch.float64)
    for t in range(horizon):
        # keep articulating the hinge as it falls (a gentle fold/unfold)
        new_target = torch.tensor([0.8 * math.sin(2 * math.pi * t / horizon)], dtype=torch.float64)
        hb.step(new_target - target, dt=dt, gravity=(0.0, 0.0, -9.81), contact=True, iters=30)
        target = new_target
        states.append(hb.state)

    renderer = Renderer(spawn=save is None, save=save)
    renderer.play(scene, states, dt=dt)
    if save:
        print(f"wrote {save} ({len(states)} frames) — open with: rerun {save}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
