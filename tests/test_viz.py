"""Renderer: structural validation + a headless end-to-end render to an .rrd file.

These don't assert pixels — they assert the Scene's structural net fires on bad geometry,
and that logging a real arm rollout to rerun runs cleanly and produces a file.
"""

import numpy as np
import pytest
import torch

from tac.viz import Scene


def test_validate_rejects_colinear_weld():
    colinear = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])  # all on the x-axis
    with pytest.raises(ValueError, match="colinear"):
        Scene(weld_groups=[[0, 1, 2]], hinge_axes=[]).validate(colinear)


def test_validate_rejects_undersized_weld():
    with pytest.raises(ValueError, match="<3 points"):
        Scene(weld_groups=[[0, 1]], hinge_axes=[]).validate(np.zeros((2, 3)))


def test_validate_rejects_degenerate_axis():
    pts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="distinct"):
        Scene(weld_groups=[[0, 1, 2]], hinge_axes=[(1, 1)]).validate(pts)


def test_scene_from_arm_passes_validation():
    from tac.robots import build_arm
    from tac.viz import scene_from_arm

    scene = scene_from_arm(build_arm())  # raises if the arm's geometry is malformed
    assert len(scene.weld_groups) == 3
    assert len(scene.hinge_axes) == 2
    assert scene.anchor_points == [0, 1, 5]


def test_render_arm_rollout_to_rrd(tmp_path):
    pytest.importorskip("rerun")
    from tac.robots import build_arm
    from tac.viz import Renderer, scene_from_arm

    arm = build_arm()
    scene = scene_from_arm(arm)
    states = [arm.state]
    for _ in range(20):
        arm.step(torch.tensor([[0.02, -0.015]]), dt=0.02)
        states.append(arm.state)

    out = tmp_path / "arm.rrd"
    Renderer(save=str(out)).play(scene, states, dt=0.02)

    assert out.exists() and out.stat().st_size > 0


def test_scene_from_gnoci_passes_validation():
    from tac.robots import build_gnoci
    from tac.viz import scene_from_gnoci

    scene = scene_from_gnoci(build_gnoci())  # raises if any weld/axis is malformed
    assert len(scene.weld_groups) == 11
    assert len(scene.hinge_axes) == 10
    assert len(scene.contact_points) == 4
    assert scene.imu_site is not None and scene.floor_z is not None


def test_render_gnoci_rollout_to_rrd(tmp_path):
    pytest.importorskip("rerun")
    import math

    from tac.robots import build_gnoci
    from tac.viz import Renderer, scene_from_gnoci

    g = build_gnoci()
    scene = scene_from_gnoci(g)
    states = [g.state]
    for t in range(20):
        action = 0.04 * math.sin(2 * math.pi * t / 20) * torch.ones(1, 10, dtype=torch.float64)
        g.step_sim(action, dt=0.02, gravity=(0.0, 0.0, 0.0), contact=False, iters=8)
        states.append(g.state)

    out = tmp_path / "gnoci.rrd"
    Renderer(save=str(out)).play(scene, states, dt=0.02)

    assert out.exists() and out.stat().st_size > 0
