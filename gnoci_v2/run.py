"""gnoci v2 loaded from its MJCF: PCA boxes per body, hinge axis points, actuators,
foot contacts and sites, all built by ModelBuilder. Drops the robot from 2 cm, then
drives a slow sinusoid on the hips and knees. Opens a pyvista window (needs a display).
"""
from pathlib import Path
import math
import time

import jax
import jax.numpy as jnp

from tac.core.sim import Simulator
from tac.io import load_mjcf
from tac.model.builder import ModelBuilder
from tac.viz.renderer import Renderer

MJCF = Path(__file__).parent / 'desc' / 'robot.xml'
DT = 0.002
NUM_ITER = 20
STEPS_PER_FRAME = 10           # 20 ms of sim per rendered frame
DROP = 0.02                    # start this far above the floor (m)
AMPLITUDE = 0.15               # rad
PERIOD = 2.0                   # s


def build_env():
    spec = load_mjcf(MJCF)
    model = ModelBuilder.from_spec(spec).build()

    state = model.state
    lift = jnp.array([0.0, 0.0, DROP])
    state = state.replace(x_current=state.x_current + lift, x_previous=state.x_previous + lift)

    simulator = Simulator(model.constraints, model.effects, num_iter=NUM_ITER, dt=DT)

    def scanstep(state, _):
        return simulator.step(state), None

    @jax.jit
    def step_fn(state):
        state, _ = jax.lax.scan(scanstep, state, None, length=STEPS_PER_FRAME)
        return state

    return model, state, step_fn


def control_at(model, t: float) -> jnp.ndarray:
    """(1, n_act) target joint angles: hips and knees swing together, hip_lat held."""
    u = [0.0] * len(model.actuators)
    s = AMPLITUDE * math.sin(2 * math.pi * t / PERIOD)
    for name in ('left_hip_lat', 'right_hip_lat'):
        u[model.actuators[name]] = s
    for name in ('left_knee', 'right_knee'):
        u[model.actuators[name]] = -s
    return jnp.array([u])


if __name__ == "__main__":
    start = time.perf_counter()
    model, state, step_fn = build_env()
    state = step_fn(state)                                    # compile
    print(f"{state.x_current.shape[1]} points, {len(model.constraints)} constraint sets, "
          f"startup {time.perf_counter() - start:.1f}s")

    renderer = Renderer(state, lines=model.lines, faces=model.faces, camera_distance=0.7, opacity=0.4)

    t = 0.0
    for _ in range(2000):
        state = state.update_control(control_at(model, t))
        state = step_fn(state)
        t += DT * STEPS_PER_FRAME
        renderer.update(state)
