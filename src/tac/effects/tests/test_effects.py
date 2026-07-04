from tac.effects.inertial import Inertia, Gravity
from tac.core.state import State
import jax.numpy as jnp

def test_inertia():
    points = jnp.array([0, 1, 2])
    friction = jnp.array([0.1])
    inertia = Inertia(points, friction)
    state = State(
        x_current=jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        x_previous=jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        inv_mass=jnp.array([1.0, 1.0, 1.0])
    )
    new_state = inertia.apply(state, 0.1)

    print(new_state.x_current)
    assert new_state.x_current[0, 0] == 0.0
    assert new_state.x_current[1, 0] == 1.0
    assert new_state.x_current[2, 0] == 2.0