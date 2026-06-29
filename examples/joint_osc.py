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

DT=0.01
ALPHA = 0.0
JOINT_ALPHA = 0.0

class JointStructure:
    def __init__(self):
        self.state = State(
            torch.tensor([
                # Solid 1
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],

            ]),
            inv_mass=torch.tensor([
                0.001, 0.001, 1.0, 1.0,
            ]),
            batch_size=1
        )
        lengths = [
            Length(self.state, 0, 1, ALPHA),
            Length(self.state, 0, 2, ALPHA),
            Length(self.state, 0, 3, ALPHA),
            Length(self.state, 1, 2, ALPHA),
            Length(self.state, 1, 3, ALPHA),
        ]

        joints = [
            Joint(self.state, [0, 1], 2, 3, JOINT_ALPHA),
        ]

        simulator = Simulator(
            state=self.state,
            constraints=[
                *lengths,
                *joints,
            ],
            effects=[
                Inertia([2, 3]),
            ],
            num_iter=3,
            dt=DT,
            constraint_optimisation='None'
        )
        self.simulator = simulator
        self.renderer = Renderer(self.state, lengths + joints)

    def perturb(self, amount=0.1):
        self.state.x_current[:, 3, 0] += amount

    def step(self):
        for _ in range(10):
            self.state = self.simulator.step(self.state)
        self.renderer.update(self.state)

    def close(self):
        self.renderer.close()

if __name__ == "__main__":
    joint = JointStructure()
    joint.perturb(amount=0.01)
    for t in range(1000):
        joint.step()
    joint.close()
