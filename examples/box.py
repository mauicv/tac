from tac.core.state import State
from tac.constraints.length import Length, VectorizedLength
from tac.constraints.contact import FloorContact
from tac.constraints.weld import FacetedWeld
from tac.core.sim import Simulator
from tac.viz.renderer import Renderer
from tac.effects.inertial import Inertia, Gravity
import random
import torch


DT = 0.005
alpha = 0.0001

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
            alpha,
            DT
        )

        self.floor_weld = FacetedWeld(
            self.state,
            [
                [8, 9, 11, 10],
            ],
            alpha,
            DT
        )

        self.contacts = [FloorContact(self.state, i, -1, alpha, DT) for i in range(8)]
        self.vectorized_length = VectorizedLength.from_lengths(self.state, self.cube_weld.get_constraint_set(), alpha, DT)

    def get_simulated_constraint_set(self):
        return [self.vectorized_length, *self.contacts]

    def push_point(self, point: int, force: torch.Tensor):
        self.state.x_current[:, point] += force


if __name__ == "__main__":
    box = Box()

    for i in range(8):
        i = random.randint(0, 7)
        box.push_point(i, torch.tensor([0.0, 0.001, 0.002]))
    
    gravity = Gravity([0, 1, 2, 3, 4, 5, 6, 7], [0.0, 0.0, -9.81])
    inertia = Inertia([0, 1, 2, 3, 4, 5, 6, 7])

    simulator = Simulator(
        simulated_constraints=box.get_simulated_constraint_set(),
        measured_constraints=[],
        effects=[inertia, gravity],
        num_iter=2,
        dt=DT
    )
    renderer = Renderer(box.state, [box.cube_weld, box.floor_weld])
    for _ in range(1000):
        for _ in range(10):
            box.state = simulator.step(box.state)
        renderer.update(box.state)
    renderer.close()
