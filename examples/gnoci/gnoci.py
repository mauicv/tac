from tac.core.state import State
from tac.constraints.length import Length
from tac.constraints.contact import FloorContact
from tac.constraints.weld import FacetedWeld
from tac.core.sim import Simulator
from tac.viz.renderer import Renderer
from tac.effects.inertial import Inertia, Gravity
from tac.loading.stl import load_stl
import random
import torch
import pyvista as pv
from tqdm import tqdm


class Gnoci:
    def __init__(self):
        stl_parsed, state, welds, joints = load_stl("examples/gnoci/stl")
        # Add floor
        n = len(state.x_current)
        min_z = state.x_current[:, :, 2].min()
        state.add(torch.tensor([
                [-5.0,-5.0, min_z],
                [5.0, -5.0, min_z],
                [-5.0, 5.0, min_z],
                [5.0, 5.0, min_z],
            ]),
            inv_mass=torch.tensor([0.0, 0.0, 0.0, 0.0])
        )
        self.state = state
        self.floor_weld = FacetedWeld(
            self.state,
            [
                [n, n+1, n+3, n+2],
            ],
            1.0
        )
        self.welds = welds
        self.joints = joints
        self.contacts = [FloorContact(self.state, i, min_z, 1.0) for i in range(n)]
        self.renderer = Renderer(state, [self.floor_weld, welds])
        self.simulator = Simulator(
            simulated_constraints=[*welds.get_constraint_set(), *joints, *self.contacts],
            measured_constraints=[],
            effects=[
                Gravity(list(range(n)), [0.0, 0.0, -9.81]),
                Inertia(list(range(n))),
            ],
            num_iter=2,
            dt=0.01
        )

    def step(self):
        for _ in range(5):
            self.state = self.simulator.step(self.state)
        self.renderer.update(self.state)

    def close(self):
        self.renderer.close()


if __name__ == "__main__":
    gnoci = Gnoci()
    for _ in tqdm(range(100)):
        gnoci.step()
    gnoci.close()
