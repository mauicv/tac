from flax import struct
import jax.numpy as jnp


@struct.dataclass
class State:
    x_current: jnp.ndarray   # (n, 3)
    x_previous: jnp.ndarray      # (n, 3)
    inv_mass: jnp.ndarray      # (n, 1)

    def __post_init__(self):
        assert len(self.x_current.shape) == 3
        assert len(self.x_previous.shape) == 3
        assert len(self.inv_mass.shape) == 2
        assert self.x_current.shape[1] == self.x_previous.shape[1] == self.inv_mass.shape[1]
        assert self.x_current.shape[0] == self.x_previous.shape[0] == self.inv_mass.shape[0]

    @classmethod
    def create(cls, x_current, inv_mass):
        return cls(x_current=x_current, x_previous=x_current, inv_mass=inv_mass)

    def _get_x_w(self, idx: jnp.ndarray):
        n, k = idx.shape
        x = self.x_current[:, idx.flatten()].reshape(-1, n, k, 3)
        w = self.inv_mass[:, idx.flatten()].reshape(-1, n, k, 1)
        return x, w

    def update_x(self, idx: jnp.ndarray, update: jnp.ndarray):
        update = update.reshape(-1, 3)
        idx = idx.flatten()
        x_current = self.x_current.at[:, idx].set(update)
        r = self.replace(x_current=x_current)
        return r