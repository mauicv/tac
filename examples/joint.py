from tac.core.state import State
from tac.constraints.length import Length, VectorizedLength
from tac.constraints.contact import FloorContact
from tac.constraints.weld import FacetedWeld
from tac.constraints.joint import Joint, VectorizedJoint
from tac.core.sim import Simulator
from tac.viz.renderer import Renderer
from tac.effects.inertial import Inertia, Gravity
import random
import torch
import math

DT=0.005
ALPHA = 0.0001

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
        solid_1_weld = FacetedWeld(
            self.state,
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3],
            ],
            ALPHA, DT
        )

        solid_2_weld = FacetedWeld(
            self.state,
            [
                [2, 3, 4],
                [2, 3, 5],
                [2, 4, 5],
                [3, 4, 5],
            ],
            ALPHA, DT
        )

        # TODO: note that a single joint leads to drift? Should look into this more.
        joints = [
            Joint(self.state, [2, 3], 0, 4, ALPHA, DT),
            Joint(self.state, [2, 3], 1, 5, ALPHA, DT)
        ]
        self.vectorized_joints = VectorizedJoint.from_joints(self.state, joints, ALPHA, DT)
        vectorized_lengths = VectorizedLength.from_lengths(
            self.state,
            [*solid_1_weld.get_constraint_set(), *solid_2_weld.get_constraint_set()],
            ALPHA,
            DT
        )

        floor_weld = FacetedWeld(
            self.state,
            [
                torch.tensor([0, 1, 3, 2]) + 6,
            ],
            1.0, DT
        )

        contacts = [FloorContact(self.state, i, -1, 1.0, DT) for i in range(6)]
        simulator = Simulator(
            constraints=[
                vectorized_lengths,
                self.vectorized_joints,
                *contacts
            ],
            effects=[
                Inertia([0, 1, 2, 3, 4, 5]),
                Gravity([0, 1, 2, 3, 4, 5], [0.0, 0.0, -9.81])
            ],
            num_iter=3,
            dt=DT
        )
        self.simulator = simulator
        self.renderer = Renderer(self.state, [solid_1_weld, solid_2_weld, floor_weld])

    def push_point(self, point: int, force: torch.Tensor):
        self.state.x_current[:, point] += force

    def step(self, action: float):
        self.vectorized_joints.offset = action
        for _ in range(10):
            self.state = self.simulator.step(self.state)
        self.renderer.update(self.state)

    def close(self):
        self.renderer.close()

if __name__ == "__main__":
    joint = JointStructure()
    for _ in range(2):
        joint.push_point(random.randint(0, 5), torch.tensor([0.0, 0.0, -0.005]))

    for t in range(1000):
        joint.step(math.sin(t * 0.01))
    joint.close()
