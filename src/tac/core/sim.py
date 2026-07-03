from tac.core.state import State
from tac.constraints.autograd import AutogradConstraint
from typing import List, Union
from tac.effects.inertial import Inertia, Gravity
import jax.numpy as jnp


class Simulator:
    def __init__(
            self,
            constraints: List[AutogradConstraint],
            effects: List[Union[Inertia, Gravity]],
            num_iter: int = 10,
            dt: float = 0.01,
        ):
        self.effects = effects
        self.constraints = constraints
        self.num_iter = num_iter
        self.dt = dt

    def constraint_step(
            self,
            state: State,
        ):
        b = state.x_current.shape[0]
        lambdas = [jnp.zeros((b, *constraint.lambda_shape)) for constraint in self.constraints]
        for _ in range(self.num_iter):
            new_lambdas = []
            for lambda_val, constraint in zip(lambdas, self.constraints):
                # with self.perf_rec(f"constraint_apply_{constraint.__class__.__name__}", constraint, lambda_val):
                state, lambda_val = constraint.apply(state, lambda_val, self.dt)
                new_lambdas.append(lambda_val)
            lambdas = new_lambdas
        return state

    def effect_step(
            self,
            state: State,
        ):
        for effect in self.effects:
            # with self.perf_rec(f"effect_apply_{effect.__class__.__name__}", effect, None):
            state = effect.apply(state, self.dt)
        return state

    def step(
            self,
            state: State,
        ):
        state = self.effect_step(state=state)
        state = self.constraint_step(state=state)
        # if self.verbose:
        #     self.perf_rec.print_stats()
        return state