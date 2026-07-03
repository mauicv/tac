from tac.core.state import State
from tac.constraints.builders import FloorContactBuilder, LengthBuilder
from tac.core.sim import Simulator
from tac.effects.inertial import Inertia, Gravity
import random
import jax.numpy as jnp
from tac.viz.renderer import Renderer
import time
import jax

DT = 0.04
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

    fc = FloorContactBuilder(-1.0)
    fc.add_constraint([0], alpha)
    fc.add_constraint([1], alpha)
    fc.add_constraint([2], alpha)
    fc.add_constraint([3], alpha)
    fc.add_constraint([4], alpha)
    fc.add_constraint([5], alpha)
    fc.add_constraint([6], alpha)
    fc.add_constraint([7], alpha)
    floor_contact_constraints = fc.build(state)

    length_constraints = lb.build(state)

    simulator = Simulator(
        constraints=[
            *length_constraints,
            *floor_contact_constraints,
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

    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    faces = [[8, 9, 11, 10]]

    renderer = Renderer(state, lines, faces)

    @jax.jit
    def step_fn(state):
        for _ in range(2):
            state = simulator.step(state)
        return state

    return state, step_fn, renderer



if __name__ == "__main__":
    state, step_fn, renderer = build_env()
    state = state.update_x(jnp.array([0]), jnp.array([[0.01, 0.03, 0.1]]))

    for _ in range(1000):
        start_time = time.perf_counter()
        state = step_fn(state)
        end_time = time.perf_counter()
        print(f'Step time: {end_time - start_time} seconds')
        renderer.update(state)
    