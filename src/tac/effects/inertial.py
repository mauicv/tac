from tac.core.state import State
from flax import struct
import jax.numpy as jnp


@struct.dataclass
class Inertia:
    points: jnp.ndarray
    friction: jnp.ndarray

    def __post_init__(self):
        assert jnp.issubdtype(self.points.dtype, jnp.integer), "points must be int array"
        assert self.friction.shape == (1, ), "friction must be a scalar"

    def apply(self, state: State, dt: float):
        idx = self.points
        x_curr = state.x_current[idx]
        x_prev = state.x_previous[idx]
        v = x_curr - x_prev
        v = v - self.friction * v * dt
        new_x_current = state.x_current.at[idx].add(v)
        return state.replace(
            x_previous=state.x_current,
            x_current=new_x_current
        )

@struct.dataclass
class Gravity:
    points: jnp.ndarray
    gravity: jnp.ndarray

    def __post_init__(self):
        assert jnp.issubdtype(self.points.dtype, jnp.integer), "points must be int array"
        assert self.gravity.shape == (3, ), "gravity must be a 3D vector"

    def apply(self, state: State, dt: float):
        idx = self.points
        x_curr = state.x_current[idx]
        update = self.gravity * dt * dt
        new_x_current = state.x_current.at[idx].add(update)
        return state.replace(
            x_previous=state.x_current,
            x_current=new_x_current
        )