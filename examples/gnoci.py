import pyvista as pv
import numpy as np
from tac.core.state import State
from tac.core.sim import Simulator
from tac.constraints.length import Length
from tac.constraints.joint import Joint
from tac.constraints.contact import FloorContact
from tac.constraints.planar import Planar, VectorizedPlanar
from tac.constraints.weld import FacetedWeld
from tac.viz.renderer import Renderer
from tac.effects.inertial import Inertia, Gravity
import torch
import time


foot_stance_length = 0.2
foot_width = 0.1
foot_height = 0.3
knee_height = 0.5
hip_height = 0.8
knee_back_offset = 0.2
gnoci_width = .8
gnoci_hip_width = 0.6
gnoci_yoke_width = 0.4
axis_offset = 0.1

LENGTH_ALPHA = 0.0
JOINT_ALPHA = 0.0
DT=0.001


class Gnoci:
    def __init__(self):
        points = torch.tensor([
            [-gnoci_width/2, -foot_stance_length/2, 0.0],                       # 0: Left Foot forward
            [-gnoci_width/2, foot_stance_length/2, 0.0],                        # 1: Left Foot backward
            [-gnoci_width/2, 0.0, foot_height/2],                               # 2: Left Foot top
            [gnoci_width/2, -foot_stance_length/2, 0.0],                        # 3: Right Foot forward
            [gnoci_width/2, foot_stance_length/2, 0.0],                         # 4: Right Foot backward
            [gnoci_width/2, 0.0, foot_height/2],                                # 5: Right Foot top

            [-gnoci_width/2, -knee_back_offset, knee_height],                   # 6: Left knee
            [gnoci_width/2, -knee_back_offset, knee_height],                    # 7: Right knee
            [-gnoci_width/2-axis_offset, 0.0, foot_height/2],                   # 8: Left knee_foot joint axis point
            [gnoci_width/2+axis_offset, 0.0, foot_height/2],                    # 9: Right knee_foot joint axis point            
            [-gnoci_width/2, 0.0, hip_height],                                  # 10: Left hip
            [gnoci_width/2, 0.0, hip_height],                                   # 11: Right hip
            [-gnoci_width/2-axis_offset, -knee_back_offset, knee_height],       # 12: Left hip_knee joint axis point
            [gnoci_width/2+axis_offset, -knee_back_offset, knee_height],        # 13: Right hip_knee joint axis point
            [-gnoci_width/2, 0.0, hip_height+0.1],                              # 14: Left hip upper
            [gnoci_width/2, 0.0, hip_height+0.1],                               # 15: Right hip upper
            [-gnoci_width/2-axis_offset, 0.0, hip_height],                      # 16: Left hip_upper_hip joint axis point
            [gnoci_width/2+axis_offset, 0.0, hip_height],                       # 17: Right hip_upper_hip joint axis point
            [-gnoci_hip_width/2, 0.0, hip_height+0.1],                          # 18: Left inner_hip upper
            [gnoci_hip_width/2, 0.0, hip_height+0.1],                           # 19: Right inner_hip upper
            [-gnoci_width/2, axis_offset, hip_height+0.1],                      # 20: Left inner_hip joint axis point
            [gnoci_width/2, axis_offset, hip_height+0.1],                       # 21: Right inner_hip joint axis point
            [0.0, 0.0, hip_height+0.1],                                         # 22: Left yoke
            [-gnoci_hip_width/2, 0.0, hip_height+0.2],                          # 23: Left yoke joint axis point
            [gnoci_hip_width/2, 0.0, hip_height+0.2],                           # 24: Right yoke joint axis point
            [0.0, -0.1, hip_height+0.2],                                        # 25: body axis point
            [0.0, 0.1, hip_height+0.2],                                         # 26: back axis point

            [-gnoci_width/2 - foot_width, -foot_stance_length/2, 0.0],          # 27: Left Foot forward
            [-gnoci_width/2 - foot_width, foot_stance_length/2, 0.0],           # 28: Left Foot backward
            [gnoci_width/2 + foot_width, -foot_stance_length/2, 0.0],           # 29: Right Foot forward
            [gnoci_width/2 + foot_width, foot_stance_length/2, 0.0],            # 30: Right Foot backward

            [0.0, 0.0, hip_height+0.3],                                         # 31: imu_point

            [-5.0,-5.0, -0.0],                                                  # 32: Floor point 1
            [5.0, -5.0, -0.0],                                                  # 33: Floor point 2
            [-5.0, 5.0, -0.0],                                                  # 34: Floor point 3
            [5.0, 5.0, -0.0],                                                   # 35: Floor point 4
        ])

        inv_mass=torch.ones(points.shape[0])

        self.state = State(
            points,
            inv_mass=torch.ones(points.shape[0]),
        )

        planars = [
            Planar(self.state, [2, 8, 12, 6], 0.0),
            Planar(self.state, [5, 9, 13, 7], 0.0),

            Planar(self.state, [6, 12, 10, 16], 0.0),
            Planar(self.state, [7, 13, 11, 17], 0.0),
        ]

        joints = [
            Joint(self.state, [2, 8], 0, 6, JOINT_ALPHA),
            Joint(self.state, [2, 8], 27, 12, JOINT_ALPHA),
            Joint(self.state, [5, 9], 3, 7, JOINT_ALPHA),
            Joint(self.state, [5, 9], 29, 13, JOINT_ALPHA),
            Joint(self.state, [6, 12], 2, 10, JOINT_ALPHA),
            Joint(self.state, [6, 12], 8, 16, JOINT_ALPHA),
            Joint(self.state, [7, 13], 5, 11, JOINT_ALPHA),
            Joint(self.state, [7, 13], 9, 17, JOINT_ALPHA),
            Joint(self.state, [10, 16], 6, 14, JOINT_ALPHA),
            Joint(self.state, [10, 16], 12, 20, JOINT_ALPHA),
            Joint(self.state, [11, 17], 7, 15, JOINT_ALPHA),
            Joint(self.state, [11, 17], 13, 21, JOINT_ALPHA),
            Joint(self.state, [14, 20], 10, 18, JOINT_ALPHA),
            Joint(self.state, [14, 20], 16, 23, JOINT_ALPHA),
            Joint(self.state, [15, 21], 11, 19, JOINT_ALPHA),
            Joint(self.state, [15, 21], 17, 24, JOINT_ALPHA),
            Joint(self.state, [18, 23], 14, 22, JOINT_ALPHA),
            Joint(self.state, [18, 23], 20, 31, JOINT_ALPHA),
            Joint(self.state, [19, 24], 15, 22, JOINT_ALPHA),
            Joint(self.state, [19, 24], 21, 31, JOINT_ALPHA),
        ]


        lengths = [
            Length(self.state, 25, 31, LENGTH_ALPHA),
            Length(self.state, 26, 31, LENGTH_ALPHA),
            Length(self.state, 22, 31, LENGTH_ALPHA),
            Length(self.state, 23, 31, LENGTH_ALPHA),
            Length(self.state, 24, 31, LENGTH_ALPHA),
            Length(self.state, 0, 27, LENGTH_ALPHA),
            Length(self.state, 1, 28, LENGTH_ALPHA),
            Length(self.state, 3, 29, LENGTH_ALPHA),
            Length(self.state, 4, 30, LENGTH_ALPHA),
            Length(self.state, 27, 28, LENGTH_ALPHA),
            Length(self.state, 27, 8, LENGTH_ALPHA),
            Length(self.state, 28, 8, LENGTH_ALPHA),
            Length(self.state, 29, 30, LENGTH_ALPHA),
            Length(self.state, 29, 9, LENGTH_ALPHA),
            Length(self.state, 30, 9, LENGTH_ALPHA),
            Length(self.state, 0, 1, LENGTH_ALPHA),
            Length(self.state, 0, 2, LENGTH_ALPHA),
            Length(self.state, 1, 2, LENGTH_ALPHA),
            Length(self.state, 3, 4, LENGTH_ALPHA),
            Length(self.state, 3, 5, LENGTH_ALPHA),
            Length(self.state, 4, 5, LENGTH_ALPHA),
            Length(self.state, 2, 6, LENGTH_ALPHA),
            Length(self.state, 5, 7, LENGTH_ALPHA),
            Length(self.state, 2, 8, LENGTH_ALPHA),
            Length(self.state, 5, 9, LENGTH_ALPHA),
            Length(self.state, 6, 8, LENGTH_ALPHA),
            Length(self.state, 7, 9, LENGTH_ALPHA),
            Length(self.state, 6, 10, LENGTH_ALPHA),
            Length(self.state, 7, 11, LENGTH_ALPHA),
            Length(self.state, 6, 12, LENGTH_ALPHA),
            Length(self.state, 7, 13, LENGTH_ALPHA),
            Length(self.state, 10, 12, LENGTH_ALPHA),
            Length(self.state, 11, 13, LENGTH_ALPHA),
            Length(self.state, 10, 14, LENGTH_ALPHA),
            Length(self.state, 11, 15, LENGTH_ALPHA),
            Length(self.state, 14, 16, LENGTH_ALPHA),
            Length(self.state, 15, 17, LENGTH_ALPHA),
            Length(self.state, 10, 16, LENGTH_ALPHA),
            Length(self.state, 11, 17, LENGTH_ALPHA),
            Length(self.state, 14, 18, LENGTH_ALPHA),
            Length(self.state, 15, 19, LENGTH_ALPHA),
            Length(self.state, 14, 20, LENGTH_ALPHA),
            Length(self.state, 15, 21, LENGTH_ALPHA),
            Length(self.state, 18, 20, LENGTH_ALPHA),
            Length(self.state, 19, 21, LENGTH_ALPHA),
            Length(self.state, 18, 22, LENGTH_ALPHA),
            Length(self.state, 19, 22, LENGTH_ALPHA),
            Length(self.state, 18, 23, LENGTH_ALPHA),
            Length(self.state, 19, 24, LENGTH_ALPHA),
            Length(self.state, 22, 23, LENGTH_ALPHA),
            Length(self.state, 22, 24, LENGTH_ALPHA),
            Length(self.state, 25, 26, LENGTH_ALPHA),
            Length(self.state, 22, 25, LENGTH_ALPHA),
            Length(self.state, 22, 26, LENGTH_ALPHA),

            Length(self.state, 0, 8, LENGTH_ALPHA),
            Length(self.state, 1, 8, LENGTH_ALPHA),
            Length(self.state, 3, 9, LENGTH_ALPHA),
            Length(self.state, 4, 9, LENGTH_ALPHA),
            Length(self.state, 2, 12, LENGTH_ALPHA),
            Length(self.state, 5, 13, LENGTH_ALPHA),
            Length(self.state, 6, 16, LENGTH_ALPHA),
            Length(self.state, 7, 17, LENGTH_ALPHA),
            Length(self.state, 10, 20, LENGTH_ALPHA),
            Length(self.state, 11, 21, LENGTH_ALPHA),
            Length(self.state, 14, 23, LENGTH_ALPHA),
            Length(self.state, 15, 24, LENGTH_ALPHA),
            Length(self.state, 23, 25, LENGTH_ALPHA),
            Length(self.state, 23, 26, LENGTH_ALPHA),
            Length(self.state, 24, 25, LENGTH_ALPHA), 
            Length(self.state, 24, 26, LENGTH_ALPHA),

            Length(self.state, 8, 12, LENGTH_ALPHA), 
            Length(self.state, 9, 13, LENGTH_ALPHA),
            Length(self.state, 12, 16, LENGTH_ALPHA), 
            Length(self.state, 13, 17, LENGTH_ALPHA),
            Length(self.state, 16, 20, LENGTH_ALPHA), 
            Length(self.state, 17, 21, LENGTH_ALPHA),
            Length(self.state, 20, 23, LENGTH_ALPHA), 
            Length(self.state, 21, 24, LENGTH_ALPHA),

        ]

        floor_weld = FacetedWeld(
            self.state,
            [
                [32, 33, 35, 34],
            ],
            1.0
        )

        floor_contacts = [FloorContact(self.state, i, 0.0, 1.0) for i in range(33)]

        self.simulator = Simulator(
            state=self.state,
            constraints=[
                *lengths,
                *floor_contacts,
                *joints,
                *planars,
            ],
            effects=[
                Inertia([i for i in range(33)], 0.0),
                Gravity([i for i in range(33)], [0.0, 0.0, -9.81])
            ],
            num_iter=1,
            dt=DT,
            constraint_optimisation='Gauss-Seidel'
            # constraint_optimisation='Jacobi'
        )

        self.renderer = Renderer(
            self.state,
            lengths + joints + planars,
            camera_distance=2,
        )

    def step(self):
        for _ in range(10):
            self.state = self.simulator.step(self.state)
        self.renderer.update(self.state)

    def close(self):
        self.renderer.close()

    def get_residuals(self):
        residuals = []
        for i, constraint in enumerate(self.simulator.constraints):
            residuals.extend(constraint.get_residual(self.simulator.lambdas[i]).flatten().tolist())
        return residuals

    def get_residual_names(self):
        i = 0
        names = []
        for constraint in self.simulator.constraints:
            for _ in range(constraint.shape[0]):
                names.append(f'{constraint.__class__.__name__}_{i}')
                i += 1
        return names

if __name__ == "__main__":
    gnoci = Gnoci()
    times = []
    for i in range(1000):
        start = time.perf_counter()
        gnoci.step()
        end = time.perf_counter()
        times.append(end - start)

    print(f"Average Time taken: {np.mean(times)}")
    print("time ratio: ", (DT * 25)/np.mean(times))
    print(f"Time per step: {np.mean(times) / 25}")
    gnoci.close()

