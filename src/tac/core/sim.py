from tac.core.state import State
from typing import List
from typing import Union
from tac.constraints.length import Length
import torch
from tac.effects.inertial import Inertia, Gravity


def inertial_step(
        state: State,
        dt: float
    ):
    v = state.x_current - state.x_previous
    new_x = state.x_current + v
    state.x_previous = state.x_current.clone()  # old current becomes previous
    state.x_current = new_x
    return state


def constraint_step(
        state: State,
        constraints: List[Union[Length]],
        num_iter: int,
        dt: float):
    batch_size = state.x_current.shape[0]
    lam = torch.zeros(batch_size, len(constraints))
    for _ in range(num_iter):
        for ind, constraint in enumerate(constraints):
            lam[:, [ind]] = constraint.apply(lam[:, [ind]], dt)
    return state


class Simulator:
    def __init__(
            self,
            constraints: List[Union[Length]],
            effects: List[Union[Inertia, Gravity]],
            num_iter: int = 10,
            dt: float = 0.01
        ):
        self.effects = effects
        self.constraints = constraints
        self.num_iter = num_iter
        self.dt = dt

    def step(self, state: State):
        for effect in self.effects:
            state = effect.apply(state, self.dt)
        state = constraint_step(state, self.constraints, self.num_iter, self.dt)
        return state