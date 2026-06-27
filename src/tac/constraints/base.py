from typing import List
from tac.core.state import State


class Constraint:
    constraint_types: List[str] = []

    def __init__(self, state: State, alpha: float):
        self.state = state
        self.alpha = alpha
        self._dt = None
        self.a_tilde = None

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, dt):
        if self._dt is None:
            self._dt = dt
            self.a_tilde = self.alpha / (self.dt * self.dt)

    def apply(self, lam, dt, **kwargs):
        self.dt = dt
        return self._apply(lam, dt, **kwargs)

    def _apply(self, lam, dt, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def _get_points(self):
        raise NotImplementedError("Optimizable constraints must implement this method")