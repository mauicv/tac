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


DT = 0.01
ALPHA = 0.0001

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
                0.0, 0.0, 0.0, 0.0                           # Floor
            ]),
            batch_size=1
        )
        object_weld = FacetedWeld(
            self.state,
            [
                [0, 1, 3, 2],
                [0, 4, 1],
                [1, 4, 3],
                [3, 4, 2],
                [2, 4, 0],
            ],
            ALPHA,
            DT
        )

        interior_point_constraint = LengthSet(
            self.state, [5], [0, 1, 2, 3, 4], ALPHA, DT
        )

        floor_weld = FacetedWeld(
            self.state,
            [
                [6, 7, 9, 8],
            ],
            1.0, DT
        )

        contacts = [FloorContact(self.state, i, -1, 1.0, DT) for i in range(5)]
        self.imu = PitchRollIMU(self.state, [4, 5], ALPHA, DT)

        self.simulator = Simulator(
            constraints=[
                *object_weld.get_constraint_set(),
                *interior_point_constraint.get_constraint_set(),
                *contacts,
                self.imu
            ],
            effects=[Inertia([0, 1, 2, 3, 4, 5])],
            num_iter=2,
            dt=0.001
        )

        self.renderer = Renderer(self.state, [object_weld, floor_weld])

    def push_point(self, point: int, force: torch.Tensor):
        self.state.x_current[:, point] += force

    def step(self):
        for _ in range(10):
            self.state = self.simulator.step(
                self.state,
                measurements={
                    "pitch": torch.tensor([0.1]),
                    "roll": torch.tensor([0.0])
                }
            )
        self.renderer.update(self.state)
        imu_p = self.state.x_current[:, self.imu.imu_points[0]]
        imu_l = self.state.x_current[:, self.imu.imu_points[1]]

        self.renderer.add_arrows(
            [imu_p, imu_p, imu_l],
            [self.imu.u, self.imu.v, self.imu.f]
        )

    def close(self):
        self.renderer.close()


if __name__ == "__main__":
    imu = IMU()
    imu.push_point(random.randint(0, 5), torch.tensor([ 0.010, 0.012, 0.00]))
    imu.push_point(random.randint(0, 5), torch.tensor([-0.010, 0.00, 0.00]))

    for _ in range(500):
        imu.step()

    imu.close()
