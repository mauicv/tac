from tac.core.state import State
from flax import struct
import jax.numpy as jnp
import jax
from tac.core.state import State
from typing import Callable


@struct.dataclass
class AutogradConstraint():
    idx: jnp.ndarray
    alpha: jnp.ndarray
    scale: jnp.ndarray
    control: jnp.ndarray
    rest: jnp.ndarray
    
    lambda_shape: tuple[int, int]
    C: Callable = struct.field(pytree_node=False)
    grad_C: Callable = struct.field(pytree_node=False)
    name: str = struct.field(pytree_node=False)
    angular_wrap: bool = struct.field(pytree_node=False)

    @classmethod
    def create(
            cls,
            state: State,
            points: jnp.ndarray,
            alpha: jnp.ndarray,
            C: Callable,
            rest: jnp.ndarray = None,
            name: str = None,
            angular_wrap: bool = False,
            control: jnp.ndarray = None,
        ):
        N = state.x_current.shape[1]
        points = jnp.asarray(points, dtype=jnp.int32)
        alpha = jnp.asarray(alpha, dtype=jnp.float32)
        n, k = points.shape
        idx = points.reshape(-1)
        counts = jnp.bincount(idx, length=N)
        scale = (1.0 / counts[idx]).reshape(n, k, 1)
        lambda_shape = (n, 1)
        if rest is None:
            x = state.x_current[:, idx].reshape(-1, n, k, 3)
            rest = C(x)[0]
        else:
            rest = jnp.asarray(rest, dtype=jnp.float32)
        grad_C = jax.grad(lambda x: C(x).sum())

        # control: per-constraint index into state.control; -1 means uncontrolled
        # (masked out at apply time). Defaults to all -1.
        if control is None:
            control = -jnp.ones(n, dtype=jnp.int32)
        else:
            control = jnp.asarray(control, dtype=jnp.int32)

        return cls(
            idx=points,
            alpha=alpha,
            scale=scale,
            control=control,
            rest=rest,
            lambda_shape=lambda_shape,
            grad_C=grad_C,
            C=C,
            name=name,
            angular_wrap=angular_wrap,
        )

    def apply(self, state: State, lam: jnp.ndarray, dt: float, **kwargs):
        B = state.x_current.shape[0]
        _alpha = self.alpha[None, :, None]/dt**2
        x, w = state._get_x_w(self.idx)                                             # (B, N, k, 3), (B, N, k, 1)
        C = self.C(x) - self.rest                                                   # (B, m)
        # per-constraint control offset gathered per batch element from
        # state.control; sentinel -1 (uncontrolled) is clipped in-bounds then
        # masked to zero.
        idx = jnp.clip(self.control, 0)                                             # (m,)
        offset = state.control[:, idx]                                              # (B, m)
        offset = jnp.where(self.control >= 0, offset, 0.0)                          # (B, m)
        C = C + offset                                                              # (B, m)
        if self.angular_wrap:
            # angular residual: take the shortest way round the ±pi seam so an
            # actuator moves 20deg across the seam instead of 340deg the long way.
            C = (C + jnp.pi) % (2 * jnp.pi) - jnp.pi
        grad = self.grad_C(x)                                                       # (B, m, k, 3)
        g2 = (grad * grad).sum(-1)[..., None]                                       # (B, m, k, 1)
        w_sum = (w * g2).sum(-2)                                                    # (B, m, 1)
        residual = - (C[..., None] + _alpha * lam)                                  
        denom = jnp.clip(w_sum + _alpha, a_min=1e-6)
        d_lambda = residual / denom                                                 # (B, m, 1)
        upd = x + w * grad * d_lambda[:, :, :, None] * self.scale[None, :, :, :]    # (B, m, k, 3)
        state = state.update_x(self.idx, upd)
        return state, lam + d_lambda