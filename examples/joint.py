from tac.core.state import State
from tac.constraints.length import Length
from tac.constraints.contact import FloorContact
from tac.constraints.weld import FacetedWeld
from tac.constraints.joint import Joint
from tac.core.sim import Simulator
from tac.viz.renderer import Renderer
from tac.effects.inertial import Inertia, Gravity
import random
import torch
import math

DT=0.01

class JointStructure:
    def __init__(self):
        self.state = State(
            torch.tensor([
                # Solid 1
                [0.0, -0.5, 0.0],
                [0.0, 0.5, 0.0],

                # Solid 1 and 2
                [1.0, 0.0, -0.5],
                [1.0, 0.0, 0.5],
                
                # Solid 2
                [2.0, -0.5, 0.0],
                [2.0, 0.5, 0.0],

                # Floor
                [-5.0,-5.0, -1.0],
                [5.0, -5.0, -1.0],
                [-5.0, 5.0, -1.0],
                [5.0, 5.0, -1.0],
            ]),
            inv_mass=torch.tensor([
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,       # Solid 1 and 2
                0.0, 0.0, 0.0, 0.0                  # Floor

            ]),
            batch_size=1
        )
        self.solid_1_weld = FacetedWeld(
            self.state,
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3],
            ],
            0.001, DT
        )

        self.solid_2_weld = FacetedWeld(
            self.state,
            [
                [2, 3, 4],
                [2, 3, 5],
                [2, 4, 5],
                [3, 4, 5],
            ],
            0.001, DT
        )

        self.joint = Joint(
            self.state,
            [2, 3],   # axis = the shared hinge edge
            [0, 1],   # from = solid 1's free edge
            [4, 5],   # to   = solid 2's free edge
            0.01, DT
        )

        self.floor_weld = FacetedWeld(
            self.state,
            [
                torch.tensor([0, 1, 3, 2]) + 6,
            ],
            1.0, DT
        )

        self.contacts = [FloorContact(self.state, i, -1, 1.0, DT) for i in range(6)]

    def get_simulated_constraint_set(self):
        return [*self.solid_1_weld.get_constraint_set(), *self.solid_2_weld.get_constraint_set(), self.joint, *self.contacts]

    def push_point(self, point: int, force: torch.Tensor):
        self.state.x_current[:, point] += force


if __name__ == "__main__":
    joint = JointStructure()
    inertia = Inertia([0, 1, 2, 3, 4, 5])
    gravity = Gravity([0, 1, 2, 3, 4, 5], [0.0, 0.0, -9.81])
    # for _ in range(2):
    #     joint.push_point(random.randint(0, 5), torch.tensor([0.0, 0.0, -0.5]))

    simulator = Simulator(
        simulated_constraints=joint.get_simulated_constraint_set(),
        measured_constraints=[],
        effects=[
            inertia,
            # gravity
        ],
        num_iter=3,
        dt=DT
    )
    renderer = Renderer(joint.state, [joint.solid_1_weld, joint.solid_2_weld, joint.floor_weld])

    for t in range(200):
        joint.joint.offset = math.sin(t*0.1)
        for _ in range(5):
            joint.state = simulator.step(joint.state)
        renderer.update(joint.state)
    renderer.close()
