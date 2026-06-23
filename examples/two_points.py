from tac.core.state import State
from tac.constraints.length import Length
from tac.core.sim import Simulator
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

    def get_constraint_set(self):
        return [self.length]


if __name__ == "__main__":
    two_points = TwoPoints(alpha=0.001)
    simulator = Simulator(two_points.get_constraint_set(), num_iter=2, dt=0.01)
    lengths = []
    x1 = []
    x2 = []
    for _ in range(100):
        two_points.state = simulator.step(two_points.state)
        lengths.append(two_points.length.get_length()[0,0].item())
        x1.append(two_points.state.x_current[0, 0, 0].item())
        x2.append(two_points.state.x_current[0, 1, 0].item())
    plt.plot(lengths)
    plt.plot(x1)
    plt.plot(x2)
    plt.show()