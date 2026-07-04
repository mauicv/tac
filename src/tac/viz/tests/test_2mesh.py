import jax.numpy as jnp
from tac.viz.renderer import faces_2_cells, lines_2_cells, state_2_points
from tac.core.state import State


def test_faces_2_cells():
    # each face is prefixed by its vertex count (pyvista cell format)
    assert faces_2_cells([[0, 1, 2, 3]]) == [4, 0, 1, 2, 3]
    assert faces_2_cells([[0, 1, 2], [3, 4, 5, 6]]) == [3, 0, 1, 2, 4, 3, 4, 5, 6]


def test_lines_2_cells():
    assert lines_2_cells([[0, 1], [2, 3]]) == [2, 0, 1, 2, 2, 3]


def test_state_2_points_takes_batch_0():
    x = jnp.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])          # (1, 2, 3)
    state = State.create(x_current=x, inv_mass=jnp.ones((1, 2)))
    pts = state_2_points(state)
    assert pts.shape == (2, 3)
