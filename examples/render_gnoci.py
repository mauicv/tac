"""Render a gnoci biped rollout in the rerun viewer.

    python examples/render_gnoci.py            # pops the native viewer and scrubs the rollout
    python examples/render_gnoci.py out.rrd    # logs headlessly to a file (open: `rerun out.rrd`)

Open-loop sinusoidal wiggle on all ten servos (gravity off, like the arm demo), so the motion
is pure kinematics — useful for sense-checking the hand-typed geometry (PLAN Note 11): the
three hip axes (red arrows) at each hip should be mutually orthogonal, the four foot-contact
points (orange) sit on the floor in the build pose, the IMU site (magenta) rides the torso,
and every limb stays rigid as the joints move.
"""

import math
import sys

import torch

from tac.robots import build_gnoci
from tac.viz import Renderer, scene_from_gnoci
from tqdm import tqdm

def main(save: str | None = None) -> None:
    g = build_gnoci()
    scene = scene_from_gnoci(g)

    horizon = 240
    num_joints = len(g.actuators)  # 10
    states = [g.state]
    for t in tqdm(range(horizon)):
        phase = 2 * math.pi * t / horizon
        # per-joint phase offset so the limbs move out of sync (reads better than a uniform
        # wiggle and exercises each axis independently).
        offsets = torch.arange(num_joints, dtype=torch.float64) * 0.6
        action = 0.04 * torch.sin(phase + offsets).unsqueeze(0)  # (1, 10)
        g.step_sim(action, dt=0.0005, gravity=(0.0, 0.0, -9.81), contact=True, iters=2)
        states.append(g.state)

    renderer = Renderer(spawn=save is None, save=save)
    renderer.play(scene, states, dt=0.0005)
    if save:
        print(f"wrote {save} ({len(states)} frames) — open with: rerun {save}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
