from tac.core.state import State
from typing import List
from typing import Union
from tac.constraints.length import Length
import torch
from tac.effects.inertial import Inertia, Gravity
from tac.constraints.joint import Joint
from tac.constraints.contact import FloorContact
from tac.constraints.imu import PitchRollIMU


import time
from collections import defaultdict


class PerformanceMonitor:
    def __init__(self):
        self.events = defaultdict(list)

    def __call__(self, event_name):
        self._current = event_name
        return self

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self._start
        self.events[self._current].append(elapsed)
        return False



class Simulator:
    def __init__(
            self,
            simulated_constraints: List[Union[Length, Joint, FloorContact]],
            measured_constraints: List[Union[PitchRollIMU, Joint, FloorContact]],
            effects: List[Union[Inertia, Gravity]],
            num_iter: int = 10,
            dt: float = 0.01,
            batch_size: int = 1
        ):
        self.effects = effects
        self.simulated_constraints = simulated_constraints
        self.measured_constraints = measured_constraints
        self.constraints = [*simulated_constraints, *measured_constraints]
        self.batch_size = batch_size
        self.lambdas = [torch.zeros(batch_size, *constraint.shape) for constraint in self.constraints]
        self.num_iter = num_iter
        self.dt = dt

        self.perf_rec = PerformanceMonitor()

    def zero_lambdas(self):
        for lambda_val in self.lambdas:
            lambda_val.zero_()

    def constraint_step(
            self,
            state: State,
            measurements: dict = None
        ):
        self.zero_lambdas()
        if measurements is None:
            measurements = {}

        for _ in range(self.num_iter):
            for ind, (lambda_val, constraint) in enumerate(zip(self.lambdas, self.constraints)):
                with self.perf_rec(f"constraint_apply_{constraint.__class__.__name__}"):
                    lambda_val = constraint.apply(lambda_val, self.dt, **measurements)
        return state

    def step(
            self,
            state: State,
            measurements: dict = None
        ):
        for effect in self.effects:
            state = effect.apply(state, self.dt)
        state = self.constraint_step(state=state, measurements=measurements)
        return state