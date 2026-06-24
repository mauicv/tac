from tac.core.state import State
from tac.constraints.length import Length
from tac.constraints.contact import FloorContact
from tac.constraints.weld import FacetedWeld
from tac.core.sim import Simulator
from tac.viz.renderer import Renderer
from tac.effects.inertial import Inertia, Gravity
from tac.constraints.imu import PitchRollIMU
from tac.constraints.weld import LengthSet
import random
import torch


class IMU:
    def __init__(self):
        self.state = State(
            torch.tensor([
                # Object
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [-0.5, 0.5, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.0, 1.0],
                
                # Interior Point
                [0.0, 0.0, 0.5],

                # Floor
                [-5.0,-5.0, -1.0],
                [5.0, -5.0, -1.0],
                [-5.0, 5.0, -1.0],
                [5.0, 5.0, -1.0],
            ]),
            inv_mass=torch.tensor([
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,                # Object
                0.0, 0.0, 0.0, 0.0                      # Floor
            ]),
            batch_size=1
        )
        self.object_weld = FacetedWeld(
            self.state,
            [
                [0, 1, 3, 2],
                [0, 4, 1],
                [1, 4, 3],
                [3, 4, 2],
                [2, 4, 0],
            ],
            0.0
        )

        self.interior_point_constraint = LengthSet(
            self.state, [5], [0, 1, 2, 3, 4], 0.0
        )

        self.floor_weld = FacetedWeld(
            self.state,
            [
                [6, 7, 9, 8],
            ],
            1.0
        )

        self.contacts = [FloorContact(self.state, i, -1, 1.0) for i in range(5)]
        self.imu = PitchRollIMU(self.state, [4, 5], 0.001)

    def get_simulated_constraint_set(self):
        return [
            *self.object_weld.get_constraint_set(),
            *self.interior_point_constraint.get_constraint_set(),
            *self.contacts
        ]

    def get_measured_constraint_set(self):
        return [self.imu]

    def push_point(self, point: int, force: torch.Tensor):
        self.state.x_current[:, point] += force


if __name__ == "__main__":
    imu = IMU()
    inertia = Inertia([0, 1, 2, 3, 4])

    simulator = Simulator(
        simulated_constraints=imu.get_simulated_constraint_set(),
        measured_constraints=imu.get_measured_constraint_set(),
        effects=[inertia],
        num_iter=2,
        dt=0.001
    )

    renderer = Renderer(imu.state, [imu.object_weld, imu.floor_weld])

    imu.push_point(4, torch.tensor([0.01, 0.012, 0]))
    imu.push_point(1, torch.tensor([-0.01, 0, 0]))

    for _ in range(500):
        for _ in range(5):
            imu.state = simulator.step(
                imu.state,
                measurements={
                    "pitch": torch.tensor([0.0]),
                    "roll": torch.tensor([0.0])
                }
            )
        renderer.update(imu.state)

        imu_p = imu.state.x_current[:, imu.imu.imu_points[0]]
        imu_l = imu.state.x_current[:, imu.imu.imu_points[1]]

        renderer.add_arrows(
            [imu_p, imu_p, imu_l],
            [imu.imu.u, imu.imu.v, imu.imu.f]
        )
    renderer.close()
