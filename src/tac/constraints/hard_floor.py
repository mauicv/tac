from tac.core.state import State
from flax import struct
import jax.numpy as jnp


@struct.dataclass
class HardFloorConstraint:
    floor_z: jnp.ndarray                            # dynamic so it can be learned/differentiated
    points: jnp.ndarray
    name: str = struct.field(pytree_node=False)
    lambda_shape: tuple[int, int]

    @classmethod
    def create(
            cls,
            points: jnp.ndarray,
            floor_z: float,
            name: str = "hard_floor",
        ):
        return cls(
            points=jnp.asarray(points, dtype=jnp.int32),
            floor_z=jnp.asarray(floor_z, dtype=jnp.float32),
            name=name,
            lambda_shape=(len(points), 1)
        )

    def apply(self, state: State, lam: jnp.ndarray, dt: float, **kwargs):
        p = self.points
        x = state.x_current[:, p]                                    # (B, n, 3)
        xp = state.x_previous[:, p]                                  # (B, n, 3)
        z = x[..., 2]                                                # (B, n)
        in_contact = z < self.floor_z                                # (B, n)
        z_new = jnp.where(in_contact, self.floor_z, z)               # (B, n)
        xp_xy_new = jnp.where(in_contact[..., None], x[..., 0:2], xp[..., 0:2])   # (B, n, 2)
        x_current = state.x_current.at[:, p, 2].set(z_new)
        x_previous = state.x_previous.at[:, p, 0:2].set(xp_xy_new)
        state = state.replace(x_current=x_current, x_previous=x_previous)
        return state, lam
