from tac.core.state import State
from tac.constraints.length import Length, VectorizedLength
from tac.constraints.joint import VectorizedJoint
from tac.constraints.contact import FloorContact
from tac.constraints.weld import FacetedWeld
from tac.core.sim import Simulator
from tac.viz.renderer import Renderer
from tac.effects.inertial import Inertia, Gravity
from tac.loading.stl import load_stl
from tac.loading.tac import load_tac
import random
import torch
import pyvista as pv
from tqdm import tqdm


ALPHA = 0.00025
DT = 0.005

class Gnoci:
    def __init__(self, load_from_tac: bool = True):
        if load_from_tac:
            state, welds, joints = load_tac("examples/gnoci/gnoci.json")
        else:
            stl_parsed, state, welds, joints = load_stl("examples/gnoci/stl")
        # Add floor
        n = state.x_current.shape[1]
        min_z = state.x_current[:, :, 2].min()
        state.add(torch.tensor([
                [-1.0,-1.0, min_z],
                [1.0, -1.0, min_z],
                [-1.0, 1.0, min_z],
                [1.0, 1.0, min_z],
            ]),
            inv_mass=torch.tensor([0.0, 0.0, 0.0, 0.0])
        )
        self.state = state
        self.welds = welds[0]
        self.joints = joints
        self.floor_weld = FacetedWeld(
            self.state,
            [
                [n, n+1, n+3, n+2],
            ],
            1.0,
            DT
        )
        self.contacts = [FloorContact(self.state, i, min_z, 1.0, DT) for i in range(n)]
        self.renderer = Renderer(
            state,
            [
                welds[0],
                # self.floor_weld
            ]
        )
        self.vectorized_lengths = VectorizedLength.from_lengths(self.state, self.welds.get_constraint_set(), ALPHA, DT)
        # self.vectorized_joints = VectorizedJoint.from_joints(self.state, self.joints, ALPHA, DT)

        self.simulator = Simulator(
            simulated_constraints=[
                self.vectorized_lengths,
                # *self.joints,
                *self.contacts
            ],
            measured_constraints=[],
            effects=[
                Gravity(list(range(n)), [0.0, 0.0, -9.81]),
                Inertia(list(range(n))),
            ],
            num_iter=1,
            dt=DT
        )

    def step(self):
        for _ in range(10):
            self.state = self.simulator.step(self.state)
        self.renderer.update(self.state)

    def close(self):
        self.renderer.close()


if __name__ == "__main__":
    gnoci = Gnoci()

    for _ in range(1000):
        gnoci.step()

    gnoci.close()
