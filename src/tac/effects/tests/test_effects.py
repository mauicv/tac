from tac.effects.inertial import Inertia, Gravity
from tac.core.state import State
import jax.numpy as jnp


def test_inertia_zero_velocity_is_noop():
    # x_current == x_previous -> zero Verlet velocity -> position unchanged
    x = jnp.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]])   # (1, 3, 3)
    state = State.create(x_current=x, inv_mass=jnp.ones((1, 3)))
    inertia = Inertia(points=jnp.array([0, 1, 2]), friction=jnp.array([0.0]))

    new_state = inertia.apply(state, 0.1)
    assert jnp.allclose(new_state.x_current, x)
    assert jnp.allclose(new_state.x_previous, x)


def test_inertia_carries_velocity():
    # point moving at v = (1, 0, 0)/step should be carried forward by v
    x_prev = jnp.zeros((1, 1, 3))
    x_curr = jnp.array([[[1.0, 0.0, 0.0]]])
    state = State.create(x_current=x_curr, inv_mass=jnp.ones((1, 1))).replace(x_previous=x_prev)
    inertia = Inertia(points=jnp.array([0]), friction=jnp.array([0.0]))

    new_state = inertia.apply(state, 0.1)
    assert jnp.allclose(new_state.x_current[0, 0], jnp.array([2.0, 0.0, 0.0]))   # advanced by v
    assert jnp.allclose(new_state.x_previous[0, 0], jnp.array([1.0, 0.0, 0.0]))  # = old x_current


def test_gravity_adds_accel_and_leaves_x_previous():
    x = jnp.zeros((1, 1, 3))
    state = State.create(x_current=x, inv_mass=jnp.ones((1, 1)))
    g = jnp.array([0.0, 0.0, -9.81])
    dt = 0.1

    new_state = Gravity(points=jnp.array([0]), gravity=g).apply(state, dt)
    assert jnp.allclose(new_state.x_current[0, 0], g * dt * dt)   # g * dt^2 added to x_current
    assert jnp.allclose(new_state.x_previous, x)                  # gravity must NOT touch x_previous
