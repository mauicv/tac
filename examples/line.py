from tac.core.state import State
from tac.constraints.length import Length
from tac.core.sim import Simulator
from tac.effects.inertial import Inertia
import torch

from matplotlib import pyplot as plt

DT=0.01

class TwoPoints:
    def __init__(self, alpha: float = 1.0):
        state = State(
            torch.tensor([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]),
            inv_mass=torch.tensor([1.0, 1.0]),
            batch_size=1
        )
        state.x_current[:, 0] = torch.tensor([-0.15, 0.0, 0.0])
        state.x_current[:, 1] = torch.tensor([1.15, 0.0, 0.0])
        length = Length(state, 0, 1, alpha)
        inertia = Inertia([0, 1])

        self.state = state
        self.simulator = Simulator(
            state=state,
            constraints=[length],
            effects=[inertia],
            num_iter=2,
            dt=DT,
            constraint_optimisation='None'
        )

    def step(self):
        self.state = self.simulator.step(self.state)
        return self.state


if __name__ == "__main__":
    two_points = TwoPoints(alpha=0.001)
    lengths = []
    for _ in range(100):
        two_points.step()
        lengths.append(two_points.simulator.constraints[0].get_length()[0,0].item())
    plt.plot(lengths)
    plt.show()