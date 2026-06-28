import pyvista as pv
import numpy as np
from tac.core.state import State
from tac.constraints.length import Length
from tac.constraints.weld import FacetedWeld
from tac.viz.renderer import Renderer
import torch
import time


foot_width = 0.2
foot_height = 0.3
knee_height = 0.5
hip_height = 0.8

knee_back_offset = 0.2

gnoci_width = .8

gnoci_hip_width = 0.6
gnoci_yoke_width = 0.4

axis_offset = 0.1


class Gnoci:
    def __init__(self):
        points = torch.tensor([
            [-gnoci_width/2, -foot_width/2, 0.0],                               # 0: Left Foot forward
            [-gnoci_width/2, foot_width/2, 0.0],                                # 1: Left Foot backward
            [-gnoci_width/2, 0.0, foot_height/2],                               # 2: Left Foot top
            [gnoci_width/2, -foot_width/2, 0.0],                                # 3: Right Foot forward
            [gnoci_width/2, foot_width/2, 0.0],                                 # 4: Right Foot backward
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
            [0.0, 0.0, hip_height+0.2],                                         # 26: imu_point
            [-5.0,-5.0, -0.0],                                                  # 27: Floor point 1
            [5.0, -5.0, -0.0],                                                  # 28: Floor point 2
            [-5.0, 5.0, -0.0],                                                  # 29: Floor point 3
            [5.0, 5.0, -0.0],                                                   # 30: Floor point 4
        ])
        self.state = State(
            points,
            inv_mass=torch.ones(points.shape[0]),
        )

        self.lengths = [
            Length(self.state, 0, 1, 1.0),
            Length(self.state, 0, 2, 1.0),
            Length(self.state, 1, 2, 1.0),
            Length(self.state, 3, 4, 1.0),
            Length(self.state, 3, 5, 1.0),
            Length(self.state, 4, 5, 1.0),
            Length(self.state, 2, 6, 1.0),
            Length(self.state, 5, 7, 1.0),
            Length(self.state, 2, 8, 1.0),
            Length(self.state, 5, 9, 1.0),
            Length(self.state, 6, 8, 1.0),
            Length(self.state, 7, 9, 1.0),
            Length(self.state, 6, 10, 1.0),
            Length(self.state, 7, 11, 1.0),
            Length(self.state, 6, 12, 1.0),
            Length(self.state, 7, 13, 1.0),
            Length(self.state, 10, 12, 1.0),
            Length(self.state, 11, 13, 1.0),
            Length(self.state, 10, 14, 1.0),
            Length(self.state, 11, 15, 1.0),
            Length(self.state, 14, 16, 1.0),
            Length(self.state, 15, 17, 1.0),
            Length(self.state, 10, 16, 1.0),
            Length(self.state, 11, 17, 1.0),
            Length(self.state, 14, 18, 1.0),
            Length(self.state, 15, 19, 1.0),
            Length(self.state, 14, 20, 1.0),
            Length(self.state, 15, 21, 1.0),
            Length(self.state, 18, 20, 1.0),
            Length(self.state, 19, 21, 1.0),
            Length(self.state, 18, 22, 1.0),
            Length(self.state, 19, 22, 1.0),
            Length(self.state, 18, 23, 1.0),
            Length(self.state, 19, 24, 1.0),
            Length(self.state, 22, 23, 1.0),
            Length(self.state, 22, 24, 1.0),
            Length(self.state, 25, 26, 1.0),
            Length(self.state, 22, 25, 1.0),
            Length(self.state, 22, 26, 1.0),
        ]

        self.floor_weld = FacetedWeld(
            self.state,
            [
                [27, 28, 30, 29],
            ],
            1.0
        )


if __name__ == "__main__":
    gnoci = Gnoci()
    renderer = Renderer(
        gnoci.state,
        gnoci.lengths + [gnoci.floor_weld],
        camera_distance=5,
    )
    while True:
        # gnoci.state = gnoci.simulator.step(gnoci.state)
        renderer.update(gnoci.state)
    time.sleep(10)
    renderer.close()


