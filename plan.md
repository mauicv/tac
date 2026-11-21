# Observations ↔ state (measurement constraints)

Goal: for training, (A) compute the robot's sensor readings from a sim `State`, and
(B) estimate a sim `State` from real sensor readings. Then (C) fit the sim's physical
parameters to real data.

Idea: a sensor reading is just another **constraint**. Encoders set joint angles, the
IMU's gravity vector sets torso roll/pitch, and foot switches pin soles to the floor.
The length and joint constraints already encode the kinematic tree, so the solver
propagates the readings to every point. That gives forward kinematics without writing a
body tree or a `fk` function.

Assumption: joints and actuators are 1-to-1, so each joint's angle is read from its
actuator's 4 points `[axis0, axis1, child_arm, parent_arm]` (as in `add_actuator`).

## What is observable

| Observation | Used as | Determines |
|---|---|---|
| joint angles q (6) | measurement constraint + loss | leg shape |
| gravity vector in IMU frame | measurement constraint + loss | torso roll, pitch |
| foot contacts (per foot) | measurement constraint + loss | torso height, no-slip → x/y velocity |
| gyro ω in IMU frame | **loss only** | angular velocity (checks the prediction) |

Not observable: **x, y, yaw** (the physics is the same if the whole robot is moved or
turned on a flat floor). With constraints these just stay where the prediction put
them. Nothing pulls on them, so nothing needs a prior. Observations and the loss don't
depend on them by construction.

The gyro is left out of the estimator: it relates `x_previous` to `x_current`, which
doesn't fit a single-snapshot `C`. Add it as a correction later if yaw drift or
angular-velocity error turns out to matter.

## A. State → observations: `observe(model, state, dt) -> Observation`

Batched `(B, …)`, pure jnp, differentiable. Each reading uses the **same `C` as its
measurement constraint**, so the estimator and the loss can't disagree on conventions.

1. **Torso frame** `A (B, 3, 3)` (rows = box axes) from the torso corners. The corner
   order follows the sign pattern, so the axes are `c[4]−c[0]`, `c[2]−c[0]` and
   `c[1]−c[0]`. Normalise, then Gram–Schmidt. No SVD. Shared helper
   `torso_frame(points) -> A`.
2. **IMU axes** in world: precompute `U = A_home @ R_mount` (IMU axes in box coordinates,
   3×3 constants). Then `imu_axes = Uᵀ @ A`, where row k is IMU axis k in world.
   `R_mount` is the sensor's mounting rotation. Identity means the IMU frame is
   world-aligned at home.
3. **Gravity**: `g_imu[k] = imu_axes[k] · (0, 0, −1)`.
4. **Gyro**: `ΔR = R_prevᵀ R_curr` from IMU axes on `x_previous` / `x_current`,
   `ω_imu = vee(ΔR − ΔRᵀ) / (2·dt)`.
5. **Joint angles**: `q = −(JointBuilder.C(actuator points) − rest)`, the same sign
   convention as the actuator control (checked against MuJoCo). Joint velocity is the
   same computed on `x_previous`.
6. **Contacts**: per foot, soft `σ((floor_z + tol − min z of sole points) / τ)` for the
   loss; hard `> 0.5` to compare with the switches.

## B. Observations → state: the estimator is a sim step

```
predict:  Inertia → Gravity                         (Verlet carries velocity)
correct:  lengths → actuators(commanded u)
          → joints(measured q) → IMU(measured g)   (measurement constraints)
          → floor + contact(measured flags)         (contact last)
```

This is `Simulator(estimator_constraints, effects, num_iter, dt)`, the existing loop
with extra constraints. Velocity, which is what we want to estimate, comes out as
`x_current − x_previous` after the correction. The measurement constraints' compliances
`α` act as sensor noise. Stiff means the reading is trusted over the prediction, soft
means it is blended (roughly a complementary filter inside the physics).

**Measurement values live in `state.control`.** Append columns after the actuator
commands: `[u (n_act) | q_meas (n_joint) | g_meas (2) | contact (n_feet)]`. No `State`
change is needed, and the existing `C − rest + control` offset path carries them.

### Measurement constraints

- **Joint**: a second `JointBuilder` over the same 4 points as each actuator, with
  `control = q_meas column`. Same sign as the actuator (`C → rest − u` ⇒ `q → q_meas`).
  `α_q` from encoder noise.
- **IMU gravity** (`ImuGravityBuilder`): points = the 4 torso corners
  `[c0, c4, c2, c1]`. Two scalar constraints, `C_k = imu_axes[k] · (0,0,−1)` for
  k = x, y, with `rest = 0` and control column = `−g_meas[k]`, so `C → g_meas[k]`.
  The z component is redundant because the vector is unit length. `U` is a closure
  constant. Normalise the measured vector first. Fixes roll/pitch only, so yaw stays
  free.
- **Contact** (`ContactMeasurement`): a hard projection like `HardFloorConstraint`,
  gated by the foot's flag column. When the flag is on, the foot's lowest sole point is
  set to `floor_z` and its x/y is frozen (`x_previous.xy = x_current.xy`, no-slip). When
  the flag is off, only the normal floor applies. Height and leg odometry both come
  from this.

### Running it

- **Online / on a log**: start from a standing home state, run the estimator step every
  sensor tick. After a short burn-in (~0.2–0.5 s) the state is consistent with the
  readings, and the velocity is the estimate.
- Each step is warm-started from the previous one, so a few iterations are enough.
  A cold start from home with large joint angles needs more `num_iter`.

### Watch-outs

- **Corrections inject velocity**: a position correction Δx changes the Verlet
  velocity by Δx/dt. That is the filter gain. If it's too noisy, soften `α`, or also
  apply the correction to `x_previous` on the first (burn-in) step.
- **Conflicting hard measurements** (encoder vs IMU vs contact at `α = 0`) make the
  solver fight. Keep the measurement constraints slightly compliant.
- **Sensor rate vs dt**: the sim runs at `dt = 0.002`, sensors likely at 100–500 Hz.
  Hold the readings over the substeps, or apply measurement constraints only on
  sensor-tick substeps.

## C. Learning the sim parameters

```
s_t     = estimator over obs[t−H … t]                (burn-in H steps, stop_gradient at first)
s_{t+k} = sim.step^k(s_t, u[t … t+k])                (open loop, commanded controls only)
loss    = Σ_k ‖observe(s_{t+k}) − obs[t+k]‖²_W        (q, gravity, gyro, soft contact)
```

Gradients reach the learned parameters. x, y and yaw drop out of the loss. Linear
velocity is never supervised directly: it is whatever makes the future
gyro/gravity/q/contact readings line up.

### Parameters (~15–25 scalars)

| Parameter | Count | Notes |
|---|---|---|
| body mass | n_body − 1 (~6) | spread over each body's 8 corners → inertia from the box; **fix total mass** (weigh the robot) |
| actuator compliance `α_act` | 6 | servo stiffness |
| actuator damping / latency | 0–12 | not modelled yet, but real servos lag. Likely the biggest sim-to-real term |
| drag | 1 | `Inertia.friction` |
| `R_mount` | 3 | or calibrate statically (robot still in a few poses) |
| measurement `α`s | ~3 | estimator gains, can be tuned rather than learned |

**Identifiability**: scaling every inverse mass by `s` and every compliance by `1/s`
gives the same XPBD update (`Δx = w∇C·Δλ`, and `Δλ` scales by `1/s`). Gravity is an
acceleration, so it doesn't break the tie. Absolute mass is therefore unidentifiable
from kinematic sensors alone. Fix the total mass and learn mass ratios.

Auxiliary points (hinge axis, contacts, sites) currently have `inv_mass = 1` like the
corners. Give them a small fixed mass fraction so they don't dominate the body inertia.

### How much data

Rough estimate: **~20–50 episodes of 10–20 s (≈5–15 min of robot time)**, cut into
tens of thousands of overlapping K-step windows. Reasoning:

- ~20 global parameters vs. ~10 sensor scalars per tick at ~100+ Hz: raw volume is
  not the limit. A minute of data is >10⁵ scalars.
- The limit is **excitation**. Each parameter needs windows where it changes the
  readings:
  - mass ratios: single-support phases, pushes/tilts, foot lift-offs
  - compliance: joints under load vs unloaded
  - latency: fast command changes
  - drag: free-ish motion
  Ten episodes of standing still identify almost nothing.
- Keep rollouts short (K ≈ 10–25 sensor ticks, 0.1–0.25 s). Gradients through hard
  contact over long open-loop horizons get noisy, and model error compounds.

Measure this rather than trust the estimate (todo 4). Fit synthetic data generated with
known parameters plus sensor noise, and plot parameter error against the number of
episodes. A learned residual world model on top would need far more data (hours, not
minutes).

## Todo

### 1. Model additions
- [ ] Torso body name + its 4 frame corners `[c0, c4, c2, c1]`, home axes `A_home`
- [ ] IMU site, `R_mount` (default identity) → `U`
- [ ] Foot → sole point ids, `floor_z` on `Model`
- [ ] Control layout: `[u | q_meas | g_meas | contact]` column offsets on `Model`
- [ ] Mass parametrisation: per-body mass → corner `inv_mass`, small aux-point mass

### 2. `tac/model/observe.py`
- [ ] `Observation` flax struct: `q`, `qd`, `gyro`, `gravity`, `contact`
- [ ] `torso_frame(points)`, `imu_axes(...)` (shared with the IMU constraint)
- [ ] `observe(model, state, dt)`

### 3. `tac/model/estimator.py` + constraints
- [ ] `ImuGravityBuilder` (2 scalar `C`s per IMU, control-offset targets)
- [ ] Measured-joint `JointBuilder` set (one per actuator, own control columns)
- [ ] `ContactMeasurement` (flag-gated floor pin + no-slip)
- [ ] `build_estimator(model, alphas) -> Simulator`, `set_measurements(state, obs)`

### 4. Headless checks
- [ ] Static: set MuJoCo q + torso pose, estimator from home → points match MuJoCo
      up to x, y, yaw (tests the "FK for free" claim and how many iterations it needs)
- [ ] `observe` on sim states vs MuJoCo site orientation and an analytic ω
- [ ] Tracking: run sim → observe (+ noise) → estimator; estimated torso velocity vs
      true, error vs `α` and sensor rate
- [ ] Gradients finite through `observe ∘ sim ∘ estimator`
- [ ] Identifiability: fit perturbed params on synthetic episodes; error vs #episodes

## Open questions (real sensors)

- Gravity: fused gravity vector from the IMU, or raw accelerometer? (Raw accel = gravity −
  acceleration; predicting it needs 3 timesteps.)
- Contacts: the foot limit switches, one boolean per foot?
- Do the servos report joint velocity, or only position? Is their latency known?
- IMU mounting and axis conventions (sets `R_mount`).
- Sensor rates and timestamps (sync between IMU, encoders, switches).
