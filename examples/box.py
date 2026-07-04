from tac.core.state import State
from tac.constraints.builders import LengthBuilder
from tac.constraints.hard_floor import HardFloorConstraint
from tac.core.sim import Simulator
from tac.effects.inertial import Inertia, Gravity
import random
import jax.numpy as jnp
from tac.viz.renderer import Renderer
import time
import jax

DT = 0.001
alpha = 0.0


def build_env():
    points = jnp.array([
        # Cube
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0],  # 7
        # # Floor
        [-5.0,-5.0, -1.0],
        [5.0, -5.0, -1.0],
        [-5.0, 5.0, -1.0],
        [5.0, 5.0, -1.0],
    ])
    inv_mass = jnp.array([
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # Cube
        0.0, 0.0, 0.0, 0.0                      # Floor
    ])
    state = State.create(
        x_current=points[None],
        inv_mass=inv_mass[None],
    )


    lb = LengthBuilder()
    for i in range(8):
        for j in range(i+1, 8):
            lb.add_constraint([i, j], alpha)

    hard_floor_constraint = HardFloorConstraint.create(points=[0, 1, 2, 3, 4, 5, 6, 7], floor_z=-1.0)

    length_constraints = lb.build(state)

    simulator = Simulator(
        constraints=[
            *length_constraints,
            hard_floor_constraint,
        ],
        effects=[
            Inertia(
                jnp.array([0, 1, 2, 3, 4, 5, 6, 7]),
                friction=jnp.array([0.1])
            ),
            Gravity(
                jnp.array([0, 1, 2, 3, 4, 5, 6, 7]), 
                jnp.array([0.0, 0.0, -9.81])
            )
        ],
        num_iter=1,
        dt=DT,
    )

    lines = []
    faces = [[8, 9, 11, 10], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]]

    renderer = Renderer(state, lines, faces, camera_distance=7)

    def scanstep(state, _):
        state = simulator.step(state)
        return state, None

    @jax.jit
    def step_fn(state):
        state, _ = jax.lax.scan(scanstep, state, None, length=40)
        return state

    return state, step_fn, renderer



if __name__ == "__main__":
    state, step_fn, renderer = build_env()
    state = state.update_x(jnp.array([0]), jnp.array([[-0.001, -0.003, 0.03]]))

    for _ in range(150):
        state = step_fn(state)
        renderer.update(state)
    