from flax import struct
import jax.numpy as jnp


@struct.dataclass
class State:
    x_current: jnp.ndarray   # (n, 3)
    x_previous: jnp.ndarray      # (n, 3)
    inv_mass: jnp.ndarray      # (n, 1)
    control: jnp.ndarray      # (B, control_size)

    def __post_init__(self):
        assert len(self.x_current.shape) == 3
        assert len(self.x_previous.shape) == 3
        assert len(self.inv_mass.shape) == 2
        assert self.x_current.shape[1] == self.x_previous.shape[1] == self.inv_mass.shape[1], f"x_current.shape: {self.x_current.shape}, x_previous.shape: {self.x_previous.shape}, inv_mass.shape: {self.inv_mass.shape}"
        assert self.x_current.shape[0] == self.x_previous.shape[0] == self.inv_mass.shape[0], f"x_current.shape: {self.x_current.shape}, x_previous.shape: {self.x_previous.shape}, inv_mass.shape: {self.inv_mass.shape}"

    @classmethod
    def create(cls, x_current, inv_mass, control_size=None):
        B = x_current.shape[0]
        # (B, control_size), per-batch controls. Keep at least one column so the
        # per-constraint gather never indexes an empty axis (uncontrolled
        # constraints read this column and get masked to zero anyway).
        control = jnp.zeros((B, control_size or 1))
        return cls(x_current=x_current, x_previous=x_current, inv_mass=inv_mass, control=control)

    def update_control(self, control: jnp.ndarray):
        return self.replace(control=control)

    def _get_x_w(self, idx: jnp.ndarray):
        n, k = idx.shape
        x = self.x_current[:, idx.flatten()].reshape(-1, n, k, 3)
        w = self.inv_mass[:, idx.flatten()].reshape(-1, n, k, 1)
        return x, w

    def update_x(self, idx: jnp.ndarray, update: jnp.ndarray):
        B = self.x_current.shape[0]
        update = update.reshape(B, -1, 3)       # (B, m*k, 3) - keep per-batch updates distinct
        idx = idx.flatten()                      # (m*k,)
        x_current = self.x_current.at[:, idx].set(update)
        r = self.replace(x_current=x_current)
        return r