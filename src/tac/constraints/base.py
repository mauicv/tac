from typing import List
from tac.core.state import State


class Constraint:
    constraint_types: List[str] = []

    def __init__(self, state: State, alpha: float, dt: float):
        self.state = state
        self.alpha = alpha
        self._dt = dt
        self.a_tilde = self.alpha / (self.dt * self.dt)

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, dt):
        self._dt = dt
        self.a_tilde = self.alpha / (self.dt * self.dt)

    def apply(self, lam, dt, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
