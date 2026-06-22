"""Render a 2-DOF arm rollout in the rerun viewer.

    python examples/render_arm.py            # pops the native viewer and scrubs the rollout
    python examples/render_arm.py out.rrd    # logs headlessly to a file (open: `rerun out.rrd`)

Sense-check: the elbow/tip should swing smoothly through the same arc the analytic FK traces
(test_arm.py), the two hinge axes (red arrows) stay vertical, and the green base anchors
never move.
"""

import math
import sys

import torch

from tac.robots import build_arm
from tac.viz import Renderer, scene_from_arm


def main(save: str | None = None) -> None:
    arm = build_arm(oop=0.3)
    scene = scene_from_arm(arm)

    horizon = 240
    states = [arm.state]
    for t in range(horizon):
        # gentle out-and-back sweep on both joints
        d1 = 0.03 * math.sin(2 * math.pi * t / horizon)
        d2 = -0.04 * math.sin(2 * math.pi * t / horizon)
        arm.step(torch.tensor([[d1, d2]]), dt=0.02)
        states.append(arm.state)

    renderer = Renderer(spawn=save is None, save=save)
    renderer.play(scene, states, dt=0.02)
    if save:
        print(f"wrote {save} ({len(states)} frames) — open with: rerun {save}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
