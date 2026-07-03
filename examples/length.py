from tac.core.state import State
from tac.constraints.builders import LengthBuilder
from tac.core.sim import Simulator
from tac.effects.inertial import Inertia
import jax.numpy as jnp
from matplotlib import pyplot as plt
import jax

DT=0.01

def build_env(alpha: float = 1.0):
    state = State.create(
        jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])[None],
        inv_mass=jnp.array([1.0, 1.0])[None],
    )
    length = LengthBuilder().add_constraint([0, 1], alpha).build(state)
    inertia = Inertia(jnp.array([0, 1]), jnp.array([0.0]))

    simulator = Simulator(
        constraints=length,
        effects=[inertia],
        num_iter=2,
        dt=DT,
    )

    state = state.update_x(jnp.array([0]), jnp.array([-0.15, 0.0, 0.0]))
    state = state.update_x(jnp.array([1]), jnp.array([1.15, 0.0, 0.0]))

    @jax.jit
    def step_fn(state):
        for _ in range(1):
            state = simulator.step(state)
        return state

    return state, step_fn

if __name__ == "__main__":
    state, step_fn = build_env(alpha=0.001)
    data = []
    for _ in range(75):
        state = step_fn(state)
        data.append(state.x_current[0, 0, 0].item())
    plt.plot(data)
    plt.show()
