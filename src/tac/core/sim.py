from tac.core.state import State
from typing import List
from typing import Union
from tac.constraints.length import Length
import torch
from tac.effects.inertial import Inertia, Gravity
from tac.constraints.joint import Joint
from tac.constraints.contact import FloorContact
from tac.constraints.imu import PitchRollIMU
from tac.perf import PerformanceMonitor
from tac.core.optim import _optimize_constraints
from typing import Literal


# TODO: just have constraints not simulated and measured.
class Simulator:
    def __init__(
            self,
            state: State,
            constraints: List[Union[Length, Joint, FloorContact, PitchRollIMU]],
            effects: List[Union[Inertia, Gravity]],
            num_iter: int = 10,
            dt: float = 0.01,
            batch_size: int = 1,
            constraint_optimisation: Literal['None', 'Jacobi', 'Gauss-Seidel'] = 'Gauss-Seidel',
            verbose: bool = True,
        ):
        self.effects = effects
        self.constraints = constraints
        self.batch_size = batch_size
        self.num_iter = num_iter
        self.dt = dt

        self.perf_rec = PerformanceMonitor(print_every=num_iter)
        self.verbose = verbose
        self.state = state

        if constraint_optimisation != 'None':
            self.constraints = _optimize_constraints(self.state, self.constraints, constraint_optimisation)

        self.lambdas = [torch.zeros(batch_size, *constraint.shape) for constraint in self.constraints]

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
                with self.perf_rec(f"constraint_apply_{constraint.__class__.__name__}-{constraint.shape }-{ind}"):
                    lambda_val = constraint.apply(lambda_val, self.dt, **measurements)
        return state

    def effect_step(
            self,
            state: State,
        ):
        for effect in self.effects:
            with self.perf_rec(f"effect_apply_{effect.__class__.__name__}"):
                state = effect.apply(state, self.dt)
        return state

    def step(
            self,
            state: State,
            measurements: dict = None
        ):
        state = self.effect_step(state=state)
        state = self.constraint_step(state=state, measurements=measurements)
        if self.verbose:
            self.perf_rec.print_stats()
        return state