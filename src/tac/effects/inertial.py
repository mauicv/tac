from tac.core.state import State
from typing import List
import torch


class Inertia:
    def __init__(self, points: List[int]):
        self.points = points

    def apply(self, state: State, dt: float):
        v = state.x_current[:, self.points] - state.x_previous[:, self.points]
        new_x = state.x_current[:, self.points] + v
        state.x_previous[:, self.points] = state.x_current[:, self.points].clone()  # old current becomes previous
        state.x_current[:, self.points] = new_x
        return state


class Gravity:
    def __init__(self, points: List[int], gravity: List[float]):
        self.gravity = gravity
        self.points = points

    def apply(self, state: State, dt: float):
        state.x_current[:, self.points] += torch.tensor(self.gravity) * dt * dt
        return state