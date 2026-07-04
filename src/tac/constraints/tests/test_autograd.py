from tac.constraints.autograd import AutogradConstraint
from tac.constraints.builders import ConstraintBuilder
from tac.core.state import State
import jax.numpy as jnp
import jax.random as random
import pytest


@pytest.fixture
def state():
    points = random.normal(random.PRNGKey(0), (4, 10, 3))
    inv_mass = jnp.ones((4, 10))
    return State.create(x_current=points, inv_mass=inv_mass)


def length_C(x: jnp.ndarray) -> jnp.ndarray:                     # (B, n, k, 3) -> (B, n)
    return jnp.linalg.norm(x[..., 0, :] - x[..., 1, :], axis=-1)


def test_autograd_constraint(state):
    idx = random.randint(random.PRNGKey(1), (10, 2), 0, 10)
    alpha = jnp.ones((10,))
    constraint = AutogradConstraint.create(state, idx, alpha, length_C)
    assert constraint.lambda_shape == (10, 1)
    assert constraint.control.shape == (10,)                     # defaults to all -1 (uncontrolled)
    assert bool(jnp.all(constraint.control == -1))


def test_autograd_constraint_apply(state):
    idx = random.randint(random.PRNGKey(1), (10, 2), 0, 10)
    alpha = jnp.ones((10,))
    constraint = AutogradConstraint.create(state, idx, alpha, length_C)
    lam = jnp.zeros((4, *constraint.lambda_shape))               # (B, n, 1)
    dt = 0.01

    new_state, new_lam = constraint.apply(state, lam, dt)
    assert new_state.x_current.shape == (4, 10, 3)
    assert new_lam.shape == (4, 10, 1)
    assert bool(jnp.all(jnp.isfinite(new_state.x_current)))


def test_constraint_builder(state):
    builder = ConstraintBuilder(length_C)
    builder.add_constraint([0, 1], 1.0)
    builder.add_constraint([1, 2], 1.0)
    builder.add_constraint([2, 3], 1.0)

    # point-disjoint greedy colouring: [0,1] and [2,3] share no points -> same set;
    # [1,2] collides with both -> its own set.
    assert builder.constraint_sets == [
        {'points': [[0, 1], [2, 3]], 'alphas': [1.0, 1.0], 'controls': [-1, -1],
         'covered_points': {0, 1, 2, 3}, 'name': None},
        {'points': [[1, 2]], 'alphas': [1.0], 'controls': [-1],
         'covered_points': {1, 2}, 'name': None},
    ]

    constraints = builder.build(state)
    assert len(constraints) == 2
    assert [c.lambda_shape for c in constraints] == [(2, 1), (1, 1)]
