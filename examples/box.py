from tac.core.state import State
from tac.constraints.builders import FloorContactBuilder, LengthBuilder
from tac.core.sim import Simulator
from tac.effects.inertial import Inertia, Gravity
import random
import jax.numpy as jnp
# from tac.viz.renderer import Renderer


DT = 0.004
alpha = 0.000


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

    length_constraints = (
        LengthBuilder()
            .add_constraint([0, 1], alpha)
            .add_constraint([1, 2], alpha)
            .add_constraint([2, 3], alpha)
            .add_constraint([3, 0], alpha)
            .add_constraint([4, 5], alpha)
            .add_constraint([5, 6], alpha)
            .add_constraint([6, 7], alpha)
            .add_constraint([7, 4], alpha)
            .add_constraint([0, 4], alpha)
            .add_constraint([1, 5], alpha)
            .add_constraint([2, 6], alpha)
            .add_constraint([3, 7], alpha)
            .build(state)
        )

    # floor_contact_constraints = (
    #     FloorContactBuilder(-1.0)
    #         .add_constraint([8, 9, 11, 10], alpha)
    #         .build(state)
    #     )

    simulator = Simulator(
        constraints=[
            *length_constraints
        ],
        effects=[
            Inertia(
                jnp.array([0, 1, 2, 3, 4, 5, 6, 7]),
                friction=jnp.array([0.0])
            ),
            Gravity(
                jnp.array([0, 1, 2, 3, 4, 5, 6, 7]), 
                jnp.array([0.0, 0.0, -9.81])
            )
        ],
        num_iter=1,
        dt=DT,
    )

    # renderer = Renderer(state, [length_constraints, floor_contact_constraints])
    return state, simulator



if __name__ == "__main__":
    state, simulator = build_env()
    state = simulator.step(state)
    
    # for _ in range(300):
    #     box.step()
    # box.close()
