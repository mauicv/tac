from tac.core.state import State
from tac.core.sim import Simulator
from tac.viz.renderer import Renderer
from tac.effects.inertial import Inertia, Gravity
from tac.constraints.builders import LengthBuilder, JointBuilder
from tac.constraints.hard_floor import HardFloorConstraint
import jax.numpy as jnp
import jax
import math
import time

dt    = 0.01
alpha = 0.00

def build_env():
    state = State.create(
        x_current=jnp.array([
            # Solid
            [0.0, -0.5, 0.0],   # 0
            [0.0, 0.5, 0.0],    # 1
            [1.0, 0.0, -0.5],   # 2
            [1.0, 0.0, 0.5],    # 3
            [2.0, -0.5, 0.0],   # 4
            [2.0, 0.5, 0.0],    # 5

            # Floor
            [-5.0,-5.0, -1.0],  # 6
            [5.0, -5.0, -1.0],  # 7
            [-5.0, 5.0, -1.0],  # 8
            [5.0, 5.0, -1.0],   # 9
        ])[None],
        inv_mass=jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])[None],
        control_size=1,
    )

    lines = []

    lb = LengthBuilder()
    for i in range(4):
        for j in range(i+1, 4):
            lb.add_constraint([i, j], alpha)
            lines.append([i, j])
        for i in range(2, 6):
            for j in range(i+1, 6):
                lb.add_constraint([i, j], alpha)
                lines.append([i, j])
    lengths = lb.build(state)

    jb = JointBuilder()
    jb.add_constraint([2, 3, 0, 4], alpha, control=0)
    jb.add_constraint([2, 3, 1, 5], alpha, control=0)
    joints = jb.build(state)

    hard_floor_constraint = HardFloorConstraint.create(points=[0, 1, 2, 3, 4, 5], floor_z=-1.0)

    simulator = Simulator(
        constraints=[
            *joints,
            *lengths,
            hard_floor_constraint
        ],
        effects=[
            Inertia(jnp.arange(6), jnp.array([0.0])),
            Gravity(jnp.arange(6), jnp.array([0.0, 0.0, -9.81]))
        ],
        num_iter=1,
        dt=dt,
    )
        
    faces = [[6, 7, 9, 8]]

    renderer = Renderer(state, lines, faces)

    @jax.jit
    def step_fn(state):
        for _ in range(1):
            state = simulator.step(state)
        return state

    return state, step_fn, renderer

if __name__ == "__main__":
    state, step_fn, renderer = build_env()
    state = state.update_x(jnp.array([0]), jnp.array([[0.001, -0.5 + 0.001, 0.0]]))

    i = 0
    for _ in range(1000):
        for _ in range(10):
            state = step_fn(state)
        state = state.update_control(jnp.array([[math.sin(i / 100)]]))
        i += 1
        renderer.update(state)
