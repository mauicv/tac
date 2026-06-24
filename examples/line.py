from tac.core.state import State
from tac.constraints.length import Length
from tac.core.sim import Simulator
from tac.effects.inertial import Inertia
import torch

from matplotlib import pyplot as plt

class TwoPoints:
    def __init__(self, alpha: float = 1.0):
        self.state = State(
            torch.tensor([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]),
            inv_mass=torch.tensor([1.0, 1.0]),
            batch_size=1
        )
        self.length = Length(self.state, 0, 1, alpha)
        self.state.x_current[:, 0] = torch.tensor([-0.15, 0.0, 0.0])
        self.state.x_current[:, 1] = torch.tensor([1.15, 0.0, 0.0])

    def get_simulated_constraint_set(self):
        return [self.length]


if __name__ == "__main__":
    two_points = TwoPoints(alpha=0.001)
    inertia = Inertia([0, 1])
    simulator = Simulator(
        simulated_constraints=two_points.get_simulated_constraint_set(),
        measured_constraints=[],
        effects=[inertia],
        num_iter=2,
        dt=0.01
    )
    lengths = []
    for _ in range(100):
        two_points.state = simulator.step(two_points.state)
        lengths.append(two_points.length.get_length()[0,0].item())
    plt.plot(lengths)
    plt.show()