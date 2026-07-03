from tac.core.state import State
from flax import struct
import jax.numpy as jnp
import jax
from tac.core.state import State
from typing import Callable


# from tac.util.log_recorder import recorder


@struct.dataclass
class AutogradConstraint():
    idx: jnp.ndarray
    alpha: jnp.ndarray
    scale: jnp.ndarray
    offset: jnp.ndarray
    rest: jnp.ndarray
    
    lambda_shape: tuple[int, int]
    C: Callable = struct.field(pytree_node=False)
    grad_C: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
            cls,
            state: State,
            points: jnp.ndarray,
            alpha: jnp.ndarray,
            C: Callable
        ):
        N = state.x_current.shape[1]
        points = jnp.asarray(points, dtype=jnp.int32)
        alpha = jnp.asarray(alpha, dtype=jnp.float32)
        n, k = points.shape
        idx = points.reshape(-1)
        counts = jnp.bincount(idx, length=N)
        scale = (1.0 / counts[idx]).reshape(n, k, 1)
        lambda_shape = (n, 1)
        offset = jnp.zeros((n, k, 1))
        x = state.x_current[:, idx].reshape(-1, n, k, 3)
        rest = C(x)[0]
        grad_C = jax.grad(lambda x: C(x).sum())

        return cls(
            idx=points,
            alpha=alpha,
            scale=scale,
            offset=offset,
            rest=rest,
            lambda_shape=lambda_shape,
            grad_C=grad_C,
            C=C
        )

    def apply(self, state: State, lam: jnp.ndarray, dt: float, **kwargs):
        B = state.x_current.shape[0]
        _alpha = self.alpha[None, :, None]/dt**2
        x, w = state._get_x_w(self.idx)                                             # (B, N, k, 3), (B, N, k, 1)
        C = self.C(x) - self.rest                                                   # (B, m, 1)
        grad = self.grad_C(x)                                                       # (B, m, k, 3)
        g2 = (grad * grad).sum(-1)[..., None]                                       # (B, m, k, 1)
        w_sum = (w * g2).sum(-2)                                                    # (B, m, 1)
        residual = - (C[..., None] + _alpha * lam)                                  
        denom = jnp.clip(w_sum + _alpha, a_min=1e-6)
        d_lambda = residual / denom                                                 # (B, m, 1)
        upd = x + w * grad * d_lambda[:, :, :, None] * self.scale[None, :, :, :]    # (B, m, k, 3)
        state = state.update_x(self.idx, upd)
        return state, lam + d_lambda