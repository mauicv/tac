from tac.constraints.autograd import AutogradConstraint
from tac.constraints.builders import ConstraintBuilder
from tac.core.state import State
import jax.numpy as jnp
import jax.random as random
import pytest


@pytest.fixture
def state():
    key = random.PRNGKey(0)
    r, key = random.split(key, 2)
    points = random.normal(r, (4, 10, 3))
    r, key = random.split(key, 2)
    inv_mass=jnp.ones((4, 10))
    return State(x_current=points, x_previous=points, inv_mass=inv_mass)

def length_C(x: jnp.ndarray) -> jnp.ndarray:
    d = x[..., 0] - x[..., 1]
    return jnp.linalg.norm(d, axis=-1)

def test_autograd_constraint(state):
    key = random.PRNGKey(0)
    r, key = random.split(key, 2)
    idx = random.randint(r, (10, 2), 0, 10)
    alpha = jnp.ones((10))
    constraint = AutogradConstraint.create(state, idx, alpha, length_C)
    assert constraint.lambda_shape == (10, 1)


def test_autograd_constraint_apply(state):
    key = random.PRNGKey(0)
    r, key = random.split(key, 2)
    idx = random.randint(r, (10, 2), 0, 10)
    alpha = jnp.ones((10))
    constraint = AutogradConstraint.create(state, idx, alpha, length_C)
    lam = jnp.ones((4, 10))
    dt = 0.01
    state, lam = constraint.apply(state, lam, dt)
    assert state.x_current.shape == (4, 10, 3)
    assert lam.shape == (4, 10)


def test_constraint_builder(state):
    builder = ConstraintBuilder(length_C)
    builder.add_constraint([0, 1], 1.0)
    builder.add_constraint([1, 2], 1.0)
    builder.add_constraint([2, 3], 1.0)
    assert builder.constraint_sets == [
        {'points': [[0, 1], [2, 3]], 'alphas': [1.0, 1.0], 'covered_points': {0, 1, 2, 3}},
        {'points': [[1, 2]], 'alphas': [1.0], 'covered_points': {1, 2}}
    ]

    constraints = builder.build(state)
    assert len(constraints) == 2
    assert [c.lambda_shape for c in constraints] == [(2, 1), (1, 1)]