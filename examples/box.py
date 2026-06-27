from tac.core.state import State
from tac.constraints.length import Length, VectorizedLength
from tac.constraints.contact import FloorContact
from tac.constraints.weld import FacetedWeld
from tac.core.sim import Simulator
from tac.viz.renderer import Renderer
from tac.effects.inertial import Inertia, Gravity
import random
import torch


DT = 0.01
alpha = 0.000

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
        cube_weld = FacetedWeld(
            self.state,
            [
                [0, 1, 3, 2],
                [0, 1, 5, 4],
                [0, 2, 6, 4],
                [1, 3, 7, 5],
                [2, 3, 7, 6],
                [4, 5, 7, 6]
            ],
            alpha
        )

        floor_weld = FacetedWeld(
            self.state,
            [
                [8, 9, 11, 10],
            ],
            alpha
        )

        self.simulator = Simulator(
            state=self.state,
            constraints=[
                *[FloorContact(self.state, i, -1, alpha) for i in range(8)],
                *cube_weld.get_constraint_set(),
            ],
            effects=[
                Inertia([0, 1, 2, 3, 4, 5, 6, 7]),
                Gravity([0, 1, 2, 3, 4, 5, 6, 7], [0.0, 0.0, -9.81])
            ],
            num_iter=2,
            dt=DT,
            constraint_optimisation='Gauss-Seidel'
            # constraint_optimisation='Jacobi'
        )

        self.renderer = Renderer(self.state, [cube_weld, floor_weld])

    def push_point(self, point: int, force: torch.Tensor):
        self.state.x_current[:, point] += force

    def step(self):
        for _ in range(10):
            self.state = self.simulator.step(self.state)
        self.renderer.update(self.state)

    def close(self):
        self.renderer.close()


if __name__ == "__main__":
    box = Box()

    for i in range(8):
        i = random.randint(0, 7)
        box.push_point(i, torch.tensor([0.0, 0.001, 0.002]))
    
    for _ in range(300):
        box.step()
    box.close()
