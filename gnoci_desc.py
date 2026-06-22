import math

import torch

from tac.core import Point, Weld, Hinge, IMU, Contact, Robot, Actuator, JointPosition
from tac.solver import Simulator
from tac.estimation import torso_velocity



root = [
    Point([0.1, 0, 0]),
    Point([0.1, 0, 0.1]),
    Point([-0.1, 0, 0]),
    Point([-0.1, 0, 0.1]),
]
imu = Point([0, 0, 0.1])
hip_weld = Weld(*root, imu)

left_yoke_points = [
    *root[0:2],
    Point([0.15, 0, 0]),
    Point([0.15, 0.1, 0]),
]
left_yoke_weld = Weld(*left_yoke_points)
left_hip_points = [
    *left_yoke_points[2:4],
    Point([0.2, 0, 00]),
    Point([0.25, 0.0, 0.0]),
]
left_hip_weld = Weld(*left_hip_points)
left_upper_leg_points = [
    *left_hip_points[2:4],
    Point([0.2, 0, -0.1]),
    Point([0.25, 0.0, -0.1]),
]
left_upper_leg_weld = Weld(*left_upper_leg_points)
left_lower_leg_points = [
    *left_upper_leg_points[2:4],
    Point([0.2, 0.0, -0.2]),
    Point([0.25, 0.0, -0.2]),
]
left_lower_leg_weld = Weld(*left_lower_leg_points)
left_foot_points = [
    *left_lower_leg_points[2:4],
    Point([0.225, 0.1, -0.3]),
    Point([0.225, -0.1, -0.3]),
]
left_foot_weld = Weld(*left_foot_points)


right_yoke_points = [
    *root[2:4],
    Point([-0.15, 0, 0]),
    Point([-0.15, 0.1, 0.0]),
]
right_yoke_weld = Weld(*right_yoke_points)
right_hip_points = [
    *right_yoke_points[2:4],
    Point([-0.2, 0, 0.0]),
    Point([-0.25, 0.0, 0.0]),
]
right_hip_weld = Weld(*right_hip_points)
right_upper_leg_points = [
    *right_hip_points[2:4],
    Point([-0.2, 0, -0.1]),
    Point([-0.25, 0.0, -0.1]),
]
right_upper_leg_weld = Weld(*right_upper_leg_points)
right_lower_leg_points = [
    *right_upper_leg_points[2:4],
    Point([-0.2, 0.0, -0.2]),
    Point([-0.25, 0.0, -0.2]),
]
right_lower_leg_weld = Weld(*right_lower_leg_points)
right_foot_points = [
    *right_lower_leg_points[2:4],
    Point([-0.225, 0.1, -0.3]),
    Point([-0.225, -0.1, -0.3]),
]
right_foot_weld = Weld(*right_foot_points)


left_root_yoke_hinge = Hinge(hip_weld, left_yoke_weld, reverse=True)
left_yoke_hip_hinge = Hinge(left_yoke_weld, left_hip_weld, reverse=True)
left_hip_upper_leg_hinge = Hinge(left_hip_weld, left_upper_leg_weld, reverse=True)
left_upper_leg_lower_leg_hinge = Hinge(left_upper_leg_weld, left_lower_leg_weld, reverse=True)
left_lower_leg_foot_hinge = Hinge(left_lower_leg_weld, left_foot_weld, reverse=True)

right_root_yoke_hinge = Hinge(hip_weld, right_yoke_weld, reverse=False)
right_yoke_hip_hinge = Hinge(right_yoke_weld, right_hip_weld, reverse=False)
right_hip_upper_leg_hinge = Hinge(right_hip_weld, right_upper_leg_weld, reverse=False)
right_upper_leg_lower_leg_hinge = Hinge(right_upper_leg_weld, right_lower_leg_weld, reverse=False)
right_lower_leg_foot_hinge = Hinge(right_lower_leg_weld, right_foot_weld, reverse=False)

actuators = [
    Actuator(left_root_yoke_hinge, gain=1.0),
    Actuator(left_yoke_hip_hinge, gain=1.0),
    Actuator(left_hip_upper_leg_hinge, gain=1.0),
    Actuator(left_upper_leg_lower_leg_hinge, gain=1.0),
    Actuator(left_lower_leg_foot_hinge, gain=1.0),
    Actuator(right_root_yoke_hinge, gain=1.0),
    Actuator(right_yoke_hip_hinge, gain=1.0),
    Actuator(right_hip_upper_leg_hinge, gain=1.0),
    Actuator(right_upper_leg_lower_leg_hinge, gain=1.0),
    Actuator(right_lower_leg_foot_hinge, gain=1.0),
]

measurement_constraints = [
    JointPosition(left_root_yoke_hinge),
    JointPosition(left_yoke_hip_hinge),
    JointPosition(left_hip_upper_leg_hinge),
    JointPosition(left_upper_leg_lower_leg_hinge),
    JointPosition(left_lower_leg_foot_hinge),
    JointPosition(right_root_yoke_hinge),
    JointPosition(right_yoke_hip_hinge),
    JointPosition(right_hip_upper_leg_hinge),
    JointPosition(right_upper_leg_lower_leg_hinge),
    JointPosition(right_lower_leg_foot_hinge),
    IMU(hip_weld, at=imu),
    Contact(left_foot_points[-2], plane_z=-0.23),
    Contact(left_foot_points[-1], plane_z=-0.23),
    Contact(right_foot_points[-2], plane_z=-0.23),
    Contact(right_foot_points[-1], plane_z=-0.23),
]

gnoci = Robot(
    hz=40,
    welds=[
        hip_weld,
        left_yoke_weld,
        left_hip_weld,
        left_upper_leg_weld,
        left_lower_leg_weld,
        left_foot_weld,
        right_yoke_weld,
        right_hip_weld,
        right_upper_leg_weld,
        right_lower_leg_weld,
        right_foot_weld,
    ],
    hinges=[
        left_root_yoke_hinge,
        left_yoke_hip_hinge,
        left_hip_upper_leg_hinge,
        left_upper_leg_lower_leg_hinge,
        left_lower_leg_foot_hinge,
        right_root_yoke_hinge,
        right_yoke_hip_hinge,
        right_hip_upper_leg_hinge,
        right_upper_leg_lower_leg_hinge,
        right_lower_leg_foot_hinge,
    ],
    measured_constraints=measurement_constraints,
    actuators=actuators,
)

B = 1
sim = Simulator(gnoci, gravity=[0, 0, -9.81], solver_iters=8)
state = gnoci.initial_state(batch=B)

# --- forward simulation rollout -------------------------------------------------
# Open-loop dynamics: apply per-joint actions, integrate, read out torso velocity.
horizon = 200                      # steps; at hz=40 this is 5 s
num_joints = len(actuators)        # 10 servos

trajectory = [state]               # keep states for the renderer (PLAN Note 11)
velocities = []

for t in range(horizon):
    # action = per-joint angle delta (Δθ servo-target increments).
    # placeholder open-loop wiggle so it moves; a learned policy plugs in here.
    action = 0.05 * math.sin(2 * math.pi * t / horizon) * torch.ones(B, num_joints)

    state = sim.step(state, action)          # Verlet integrate + XPBD constraint solve
    v_torso = torso_velocity(state)          # CoM / torso linear velocity readout

    trajectory.append(state)
    velocities.append(v_torso)

# --- estimation mode (alternative) ---------------------------------------------
# Replay a log, fusing measured sensors each step instead of trusting the prior:
#
# for action_t, measurement_t in log:           # measurement_t = encoder θ, IMU pitch/roll, contacts
#     state = sim.step(state, action_t)         # XPBD prior
#     state = sim.correct(state, measurement_t) # apply measured_constraints (hard / small-compliance)
#     v_torso = torso_velocity(state)           # compare to ground-truth torso velocity (milestone H)

# render the rollout (PLAN Note 11):  renderer.play(trajectory)