from tac.core.state import State
from tac.constraints.length import Length
from tac.constraints.contact import FloorContact
from tac.constraints.weld import FacetedWeld
from tac.core.sim import Simulator
from tac.viz.renderer import Renderer
from tac.effects.inertial import Inertia, Gravity
import random
import torch


class Box:
    def __init__(self):
        self.state = State(
            torch.tensor([
                # Cube
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                # Floor
                [-5.0,-5.0, -1.0],
                [5.0, -5.0, -1.0],
                [-5.0, 5.0, -1.0],
                [5.0, 5.0, -1.0],
            ]),
            inv_mass=torch.tensor([
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # Cube
                0.0, 0.0, 0.0, 0.0                      # Floor
            ]),
            batch_size=1
        )
        self.cube_weld = FacetedWeld(
            self.state,
            [
                [0, 1, 3, 2],
                [0, 1, 5, 4],
                [0, 2, 6, 4],
                [1, 3, 7, 5],
                [2, 3, 7, 6],
                [4, 5, 7, 6]
            ],
            0.001
        )

        self.floor_weld = FacetedWeld(
            self.state,
            [
                [8, 9, 11, 10],
            ],
            1.0
        )

        self.contacts = [FloorContact(self.state, i, -1, 1.0) for i in range(8)]

    def get_constraint_set(self):
        return [*self.cube_weld.get_constraint_set(), *self.contacts]

    def push_point(self, point: int, force: torch.Tensor):
        self.state.x_current[:, point] += force


if __name__ == "__main__":
    box = Box()

    for i in range(8):
        i = random.randint(0, 7)
        box.push_point(i, torch.tensor([0.0, 0.0, -0.05]))
    
    gravity = Gravity([0, 1, 2, 3, 4, 5, 6, 7], [0.0, 0.0, -9.81])
    inertia = Inertia([0, 1, 2, 3, 4, 5, 6, 7])

    simulator = Simulator(box.get_constraint_set(), [inertia, gravity], num_iter=2, dt=0.01)
    renderer = Renderer(box.state, [box.cube_weld, box.floor_weld])
    for _ in range(100):
        for _ in range(5):
            box.state = simulator.step(box.state)
        renderer.update(box.state)
    renderer.close()
