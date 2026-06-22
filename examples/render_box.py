"""Render a free rigid box falling under gravity in the rerun viewer.

    python examples/render_box.py            # pops the native viewer and scrubs the rollout
    python examples/render_box.py out.rrd    # logs headlessly to a file (open: `rerun out.rrd`)

The simplest sanity-check in the engine (PLAN Phase 1): a single rigid `Weld` of 8 cube
corners dropped from rest. It should fall straight down in pure translation (the blue weld
edges stay fixed length = rigid), land flat on the grey floor, and rest there without
bouncing or sinking — the contact points (orange) pin to the plane. A clean baseline before
the articulated arm / gnoci.
"""

import sys

from tac.robots import build_box
from tac.viz import Renderer, scene_from_box

FLOOR_Z = -0.23


def main(save: str | None = None) -> None:
    box = build_box(size=0.2, center=(0.0, 0.0, 0.6), floor_z=FLOOR_Z)
    scene = scene_from_box(box)

    dt = 0.01
    horizon = 200
    states = [box.state]
    for _ in range(horizon):
        box.step(dt=dt, gravity=(0.0, 0.0, -9.81), contact=True, iters=10)
        states.append(box.state)

    renderer = Renderer(spawn=save is None, save=save)
    renderer.play(scene, states, dt=dt)
    if save:
        print(f"wrote {save} ({len(states)} frames) — open with: rerun {save}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
