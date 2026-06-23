from tac.core.state import State
from tac.constraints.length import Length
from tac.constraints.weld import FacetedWeld
from tac.core.sim import Simulator
from tac.viz.renderer import Renderer
import torch


class Box:
    def __init__(self):
        self.state = State(
            torch.tensor([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0]
            ]),
            inv_mass=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            batch_size=1
        )
        self.weld = FacetedWeld(
            self.state,
            [
                [0, 1, 3, 2],
                [0, 1, 5, 4],
                [0, 2, 6, 4],
                [1, 3, 7, 5],
                [2, 3, 7, 6],
                [4, 5, 7, 6]
            ],
            1.0
        )

    def get_constraint_set(self):
        return self.weld.get_constraint_set()

    def render(self):
        mesh = weld_2_mesh(self.weld)
        mesh.plot(show_edges=True)


if __name__ == "__main__":
    box = Box()
    box.state.x_previous[:, 0] = torch.tensor([0.0, 0.1, -0.1])
    simulator = Simulator(box.get_constraint_set(), num_iter=2, dt=1)
    renderer = Renderer(box.state, [box.weld])
    for _ in range(100):
        box.state = simulator.step(box.state)
        renderer.update(box.state)
    renderer.close()
