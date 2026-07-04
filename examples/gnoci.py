import pyvista as pv
from tac.core.state import State
from tac.core.sim import Simulator
from tac.constraints.builders import PlanarBuilder, LengthBuilder, JointBuilder
from tac.constraints.hard_floor import HardFloorConstraint
from tac.viz.renderer import Renderer
from tac.effects.inertial import Inertia, Gravity
import jax.numpy as jnp
import time
import jax
import numpy as np


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


def build_env():
    points = jnp.array([
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
    ])[None]

    inv_mass=jnp.ones(36)[None]

    state = State.create(
        x_current=points,
        inv_mass=inv_mass,
        control_size=10,
    )

    pb = PlanarBuilder()
    pb.add_constraint([2, 8, 12, 6], 0.0)
    pb.add_constraint([5, 9, 13, 7], 0.0)
    pb.add_constraint([6, 12, 10, 16], 0.0)
    pb.add_constraint([7, 13, 11, 17], 0.0)
    planars = pb.build(state)

    jb = JointBuilder()
    jb.add_constraint([2, 8, 0, 6], JOINT_ALPHA)
    jb.add_constraint([2, 8, 27, 12], JOINT_ALPHA)
    jb.add_constraint([5, 9, 3, 7], JOINT_ALPHA)
    jb.add_constraint([5, 9, 29, 13], JOINT_ALPHA)
    jb.add_constraint([6, 12, 2, 10], JOINT_ALPHA)
    jb.add_constraint([6, 12, 8, 16], JOINT_ALPHA)
    jb.add_constraint([7, 13, 5, 11], JOINT_ALPHA)
    jb.add_constraint([7, 13, 9, 17], JOINT_ALPHA)
    jb.add_constraint([10, 16, 6, 14], JOINT_ALPHA)
    jb.add_constraint([10, 16, 12, 20], JOINT_ALPHA)
    jb.add_constraint([11, 17, 7, 15], JOINT_ALPHA)
    jb.add_constraint([11, 17, 13, 21], JOINT_ALPHA)
    jb.add_constraint([14, 20, 10, 18], JOINT_ALPHA)
    jb.add_constraint([14, 20, 16, 23], JOINT_ALPHA)
    jb.add_constraint([15, 21, 11, 19], JOINT_ALPHA)
    jb.add_constraint([15, 21, 17, 24], JOINT_ALPHA)
    jb.add_constraint([18, 23, 14, 22], JOINT_ALPHA)
    jb.add_constraint([18, 23, 20, 31], JOINT_ALPHA)
    jb.add_constraint([19, 24, 15, 22], JOINT_ALPHA)
    jb.add_constraint([19, 24, 21, 31], JOINT_ALPHA)
    joints = jb.build(state)

    line_cells = [
        [25, 31],[26, 31],[22, 31],[23, 31], 
        [24, 31],[0, 27],[1, 28],[3, 29],
        [4, 30],[27, 28],[27, 8],[28, 8],
        [29, 30],[29, 9],[30, 9],[0, 1],
        [0, 2],[1, 2],[3, 4],[3, 5],
        [4, 5],[2, 6],[5, 7],[2, 8],
        [5, 9],[6, 8],[7, 9],[6, 10],
        [7, 11],[6, 12],[7, 13],[10, 12],
        [11, 13],[10, 14],[11, 15],[14, 16],
        [15, 17],[10, 16],[11, 17],[14, 18],
        [15, 19],[14, 20],[15, 21],[18, 20],
        [19, 21],[18, 22],[19, 22],[18, 23],
        [19, 24],[22, 23],[22, 24],[25, 26],
        [22, 25],[22, 26],[0, 8],[1, 8],
        [3, 9],[4, 9],[2, 12],[5, 13],
        [6, 16],[7, 17],[10, 20],[11, 21],
        [14, 23],[15, 24],[23, 25],[23, 26],
        [24, 25],[24, 26],[8, 12],[9, 13],
        [12, 16],[13, 17],[16, 20],[17, 21],
        [20, 23],[21, 24]
    ]

    lb = LengthBuilder()
    for cell in line_cells:
        lb.add_constraint(cell, LENGTH_ALPHA)
    lengths = lb.build(state)

    hard_floor_constraint = HardFloorConstraint.create(points=jnp.arange(33), floor_z=-0.0)

    simulator = Simulator(
        constraints=[
            *lengths,
            *joints,
            *planars,
            hard_floor_constraint,
        ],
        effects=[
            Inertia(jnp.arange(33), jnp.array([0.0])),
            Gravity(jnp.arange(33), jnp.array([0.0, 0.0, -9.81]))
        ],
        num_iter=1,
        dt=DT,
    )

    renderer = Renderer(
        state,
        lines=line_cells,
        faces=[[32,33,35,34]],
        camera_distance=3,
    )

    def scanstep(state, _):
        state = simulator.step(state)
        return state, None

    @jax.jit
    def step_fn(state):
        state, _ = jax.lax.scan(scanstep, state, None, length=10)
        return state

    return state, step_fn, renderer

if __name__ == "__main__":
    start = time.perf_counter()
    state, step_fn, renderer = build_env()
    left_yoke = [0.0, -0.007, hip_height+0.1]
    state = state.update_x(jnp.array([22]), jnp.array([left_yoke]))
    state = step_fn(state)
    end = time.perf_counter()
    print(f"Startup time taken: {end - start}")

    logs = []
    for i in range(1000):
        start = time.perf_counter()
        state = step_fn(state)
        end = time.perf_counter()
        rt = end - start
        st = DT*10
        logs.append(rt)
        renderer.update(state)

    print(f"Average Time taken: {np.mean(logs)}")
    print("time ratio: ", (DT * 10)/np.mean(logs))
    print(f"Time per step: {np.mean(logs) / 10}")
    renderer.close()