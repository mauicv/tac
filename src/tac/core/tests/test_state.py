import jax.numpy as jnp
from tac.core.state import State
import pytest


@pytest.mark.parametrize("batch_size", [1, 2])
def test_state(batch_size):
    pts = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])          # (2, 3)
    x_current = jnp.broadcast_to(pts, (batch_size, 2, 3))
    inv_mass = jnp.ones((batch_size, 2))
    state = State.create(x_current=x_current, inv_mass=inv_mass, control_size=3)

    assert state.x_current.shape == (batch_size, 2, 3)
    assert state.x_previous.shape == (batch_size, 2, 3)          # create seeds x_previous = x_current
    assert state.inv_mass.shape == (batch_size, 2)
    assert state.control.shape == (batch_size, 3)


def test_state_control_defaults_to_one_column():
    state = State.create(x_current=jnp.zeros((1, 2, 3)), inv_mass=jnp.ones((1, 2)))
    # keep >= 1 control column so the per-constraint gather never hits an empty axis
    assert state.control.shape == (1, 1)
