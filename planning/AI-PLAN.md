# tac — Plan

`tac` is a minimal, differentiable position-based dynamics (PBD / XPBD) physics engine
in PyTorch, built to serve as a **state estimator** for a bipedal robot ("gnoci"). Its
primary job is to estimate the robot's **CoM / torso linear velocity** from onboard
sensors only (IMU, joint encoders, foot-contact switches) — no external motion capture.

The same engine doubles as the differentiable backbone of a **world model**: the
XPBD prior can be corrected either by (a) a learned neural residual, or (b) the robot's
measured next state (sensor fusion / filtering).

> Name: `tac` comes from **Ta**keshi **K**ovacs (*Altered Carbon*) and also reads as a
> shortening of *tactile* (physical/contact). Kept for now; alternatives noted in the
> open questions.

---

## The core abstraction (settled framing)

Everything is built from **three kinds of thing**, nothing else:

1. **Point** — just a position `(x, y, z)`. No stored orientation, no frame, no
   quaternion. A point is a particle.
2. **Weld** — a set of points held mutually rigid by length constraints between them, so
   the set moves and turns as one body. A weld of **≥3 non-colinear points carries a full
   orientation** (a frame) *emergently, from the positions of its points*. The weld is the
   **rigid-body primitive**; orientation is never stored, only read off the point cloud.
3. **Constraint** — a relationship between points / welds, projected by the solver:
   `weld` (rigidity), `joint` (revolute coupling between two welds), `actuator` (drives a
   joint angle), `contact` (floor), `imu` (orient the torso from measured pitch/roll),
   `anchor` (pin a point), plus the bare `length` constraint that welds are built from.

This replaces the earlier "points-only with per-joint axis vectors" idea. The key
realisation: a lone point has no orientation, but a rigid triad does — so make the **weld**
the atom that carries orientation, and let orientation fall out of geometry. No
quaternions, no per-point frames, no rotational integration, and the FK axis-propagation
bookkeeping from the previous draft is **deleted** (see "Why welds" below).

### Revolute joint = two welds sharing an edge
A hinge between two welds is expressed by having them **share two points** (an *edge*).
Two rigid bodies sharing two points have exactly **one** relative DOF: rotation about the
line through those two points. So:

- The **two shared points define the hinge axis** — no axis vector is stored.
- The **shared points are the pivot** — the positional half of the joint is free (already
  enforced by both welds containing those points); no coincidence constraint needed.
- The **angle θ** is read from the off-axis points of the two welds about that edge.
- When the parent weld moves, the shared edge (and thus the axis) moves with it
  automatically, because it is made of the parent's points. **Axis propagation is free.**

`Joint(weld_parent, weld_child, theta0=0.0)` — axis, pivot and θ are all emergent from the
two welds' shared edge. **Both the arm and gnoci use shared edges.** (Escape hatch, rarely
needed, for an axis with no natural points to sit on: `Joint(..., pivot=p, axis=[...])`.)

### Why welds (the trade we accepted)
We compared three orientation models and chose welds:

- **Reduced / points-only** (previous draft): orientation implicit in arrangement, needs
  per-joint live axis vectors + a hand-written FK propagation rule; fragile to pure twist.
- **Per-point quaternion** (maximal coords): uniform pose atoms, but pushes complexity into
  the solver — angular corrections, inverse *inertia*, renormalisation — and contaminates
  the otherwise all-positional contact / velocity / IMU pipeline.
- **Welds** (chosen): orientation **emergent from positions**, so nothing to store,
  normalise, or keep in sync; the solver stays **positions-only** (the trivial XPBD
  position projection); twist is captured because 3 points span all rotational DOF. Price:
  **more points** (every oriented body is a triad+), and joint axes are now *geometry*
  rather than explicit parameters (this *reparametrises* axis calibration as "learn the
  rest positions of the axis points" — not lost, just moved). Points are cheap; we take it.

Quaternions only resurface for the deferred **airborne world-model rollout** (no
contact/IMU to anchor the torso), and only on that one body. Off the estimation path.

---

## Decisions locked in

| # | Topic | Decision |
|---|-------|----------|
| 1 | Dimensionality | **3D**, modelled as **points + welds + constraints** only. No explicit cubes/inertia tensors; rigidity, orientation, angles, contact are all constraints. |
| 2 | Topology | **End goal:** the `gnoci` MuJoCo model, built **by hand**. **Start:** a **2-DOF planar arm fixed to the ground** to validate the machinery (see "Build target"). |
| 3 | Action | **Per-joint angle delta** (Δθ → joint actuator target, scaled by a learnable **action gain**). |
| 4 | Gravity | Global acceleration vector (default `[0,0,-9.81]`), applied to point positions in Verlet. |
| 5 | Timestep | Fixed `dt` via an `hz` parameter (`dt = 1/hz`). No substepping for now. |
| 6 | Solver | **XPBD**, **positions-only** projection. Per-constraint compliance `alpha` (`0` = hard). Iteration count `solver_iters`. Constraint order matters (see Note 1). |
| 7 | Orientation | **Emergent from welds.** A weld of ≥3 non-colinear points carries a frame; no stored orientation, no quaternions, no per-point frames. |
| 8 | Joints | **Revolute = two welds sharing an edge** (2 shared points). Axis/pivot/θ emergent. `pivot+axis` is a documented fallback. |
| 9 | Actuators | A joint is a **servo**: a target angle `theta_target` (set by action `+= gain·Δ`, or by encoder in estimation) reached with learnable compliance `alpha` (servo softness). `theta = 0` at the build pose. |
| 10 | Contact | **2 contact points per foot** (forward/back), tied into the **foot weld** (triangle). **No-slip** in plane: while in contact zero the point's x,y velocity (`x_curr = x_prev`). **z asymmetric**: floor pushes up only — if below floor snap **both** `z_curr` and `z_prev` to floor height (no bounce); if above, leave free. |
| 11 | IMU | IMU attached to the torso weld **at a point**; the `imu` constraint **rotates the weld about that point** so its frame matches measured **pitch & roll** (accel+gyro, no magnetometer → **no yaw**; yaw free, possibly added later). |
| 12 | Modeled vs measured | **One Gauss–Seidel loop**, no tree. **Modeled** constraints are the dynamics prior; **measured** constraints are the *same kind of constraint* but with their target fed by a sensor (encoder angle, IMU tilt, contact). Both live in the **same constraint list and reassert every iteration** (hard = "frozen measurement", Decision 13). A measurement's effect **propagates by iteration, not by a downstream graph** — pinning a foot relocates the base because GS carries the pin up the rigid chain. So there is *no* propagation machinery and *no* tree (earlier draft's premise; dropped — see Note 1). What remains of "measured": target source (encoder→`theta_target`, IMU→`TiltImu`, touch→`Contact.active`), and order/priority. |
| 13 | Correction blend | **Hard enforcement** (measured constraints `alpha=0`, reasserted every GS pass = frozen measurement) for now; **small compliance** = soft fusion later (Note 2). |
| 14 | Encoder rate | Encoders assumed sampled at sim rate (resampled to `dt` if logged otherwise). |
| 15 | CoM / masses | Probably no explicit CoM initially; read velocity off the torso. **Per-point inverse mass `w` is a learnable parameter** (modulates how much each point moves under a constraint). |
| 16 | Neural model | Two candidate modes (don't over-invest): (a) NN predicts **sensor values** → fed through measured constraints; (b) NN predicts **deltas on `x_curr`** → re-solve. **Priority is velocity estimation**, not the world model. |
| 17 | Data | Logged robot trajectories (sensor + ground truth) **and** MuJoCo `gnoci` sim data. |
| 18 | Batch / devices | **Batched from day one** `(B, N, 3)`. **CPU + CUDA** (no MPS — Intel Mac locally, CUDA for training). |
| 19 | Tooling | Latest Python + PyTorch, `pyproject.toml` package, `pytest`. Newest versions that work on the local (older Intel) Mac. |
| 20 | Naming | `tac` (see top). Open to change. |

---

## Learnable parameters

Marked as `nn.Parameter` (vs fixed buffers); fit to real/sim data.

- **Weld / link rest lengths** — target distances in `length` constraints.
- **Per-constraint compliance `alpha`** — XPBD softness; for actuators this *is* the servo
  stiffness (1/strength); for measured constraints it ↔ inverse sensor confidence.
- **Per-point inverse mass `w`** — how much a point yields under a constraint (`x_i += w_i·Δλ·∇C_i`).
- **Action gain** — command Δ → `theta_target` increment.
- **Axis-point rest positions** — joint-axis calibration, since axes are geometry now.
- **(Optionally) contact thresholds.**

**Per-parameter learnability scale.** Each learnable parameter carries a **scale on how fast
it may move** during fitting — some quantities (axis-point rest positions, link lengths) are
near-fixed physical facts that should drift only *slowly*, while others (action gain,
compliances) can adapt freely. Implement as per-parameter learning-rate groups, or a
reparameterisation `value = base + scale · δ` with `δ` learned and `scale` small/frozen.
Keeps the geometry from being yanked around to fit transient data.

> "Actuator strength" = **action gain** (command→target) **+ compliance** (how stiffly the
> servo holds the target), *not* a raw scaling of child-point positions.

**Identifiability:** `inv_mass w` (per-point) and compliance `alpha` (per-constraint) both
scale how far a point moves under a constraint, so their effects partly overlap — the fit
can trade one against the other and become *under-determined* ("ill-posed": many parameter
combinations explain the data equally, so no unique best fit and the gradients fight). We
keep both (not perfectly degenerate — `w` is shared across all constraints on a point,
`alpha` across all points in a constraint), but expect to **regularise or pin one** if
fitting is unstable.

---

## Build target: 2-DOF arm first (then gnoci)

A revolute–revolute arm fixed to the ground (built in 3D; motion about world-z). Exercises
points + Verlet + `length` + `weld` + **shared-edge `joint`** + `actuator` + `anchor` + the
XPBD solver + velocity readout, with **no contact, no IMU, no floating base**.

```
base weld (anchored)                 ── anchor
│   shoulder servo θ₁  (axis ẑ = shared vertical edge)
└── link1 weld
    │  elbow servo θ₂  (axis ẑ = shared vertical edge)
    └── link2 weld (tip)
```

Each joint axis is materialised as a **shared vertical edge** (two stacked points along ẑ);
each link weld = that shared edge + an off-axis point for the frame. So the arm exercises
the real shared-edge machinery from day one. Action = Δθ on the two joints; test: the tip
traces the expected arc and `tip_velocity = Δx/dt` matches the analytic forward-kinematic
value.

First milestone; gnoci follows.

---

## The gnoci robot (from `gnoci_xml/gnoci.xml`)

Floating-base biped, root = `head_base` (torso, carries the IMU site). Each leg is a serial
chain of 5 hinge joints, mirrored left/right. MuJoCo joint axes are all `±z` in the local
(parent) frame.

```
head_base (torso weld, root)         ── IMU site "imu" (pitch, roll)
│
├── yoke           head__{l,r}_yoke           hinge → yaw   (horizontal)
│   └── hip        {l,r}_yoke__hip            hinge → roll  (lateral)
│       └── upper_leg  {l,r}_hip__upper_leg   hinge → pitch (leg swing)
│           └── lower_leg {..}__lower_leg     hinge → knee
│               └── foot   {..}__foot         hinge → ankle
│                   └── foot weld: forward + back contact points
```

Each body becomes a **weld**; consecutive bodies **share the edge** that is their hinge
axis. Where two consecutive axes intersect (the stacked hip hinges, all through ~the hip
point), the shared body is a clean triad whose two edges meet at the pivot — exactly the
"stub points" of the previous draft, now promoted to *being the axes*. Where axes are far
apart and parallel (e.g. pitch at the hip, knee at the knee), the body is a longer 4-point
weld with an edge at each end. All of this is **placed by hand** (like the arm); a builder
helper that would materialise the shared edge + frame points is **deferred** (Note 13). The
stored representation is purely points + welds + shared edges.

**Sensors** (the measurement model):
- `jointpos` ×10 — absolute joint angles (encoders).
- `touch` ×4 — foot contact (forward/back × left/right). 2 sensed switches per foot.
- `imu-gyro` + `imu-acc` on torso → consume **pitch & roll**; yaw not directly observed (Q-F).

(`scene.xml`: floor at `z = -0.23`, `timestep 0.005` → 200 Hz, gravity `-9.8`.)

---

## Core concepts

### State
Batched tensors, `B` = batch, `N` = points, `D = 3`:
- `x_current` `(B, N, 3)` — current positions.
- `x_previous` `(B, N, 3)` — previous positions (Verlet: velocity implicit, `x_cur − x_prev`).
- `inv_mass` `(B, N)` or `(N,)`, `0` = pinned. **Learnable.**
- per-joint `theta_target` `(B, J)` — the servo target (set by action or encoder).
- per-joint `theta_build` `(J,)` — rest-pose angle offset so `θ = 0` at build.
- topology metadata: weld membership (which points), each joint's `(weld_parent,
  weld_child, shared_edge_points)`. (No tree — propagation is emergent from GS; Note 1.)

**No per-point orientation, no per-joint axis, no quaternions.** The *actual* joint angle
is **derived from positions** each step (we store only the target); orientation of a body
is **derived from its weld's points**.

### Frame from a weld (helper)
For a weld with points `p₀,p₁,p₂,…`: `e1 = normalize(p1−p0)`,
`e2 = normalize((p2−p0) − ((p2−p0)·e1)e1)`, `e3 = e1×e2`. The columns `[e1 e2 e3]` are the
body frame. Used only to *read* orientation (IMU pitch/roll, joint-angle readback); never
stored. Well-conditioned only if the points span real area — see "sizing" below.

### Constraint set
A constraint exposes `project(state) -> state`, is differentiable, and carries a
**compliance `alpha`** (XPBD; `0` = hard), a **scope** (`modeled` | `measured`), and an
**order key**.

- **`length(a, b, L)`** — hold `|x_a − x_b| = L`. The atom welds are built from. `L` learnable.
- **`Weld([p1, p2, …])`** — make a point set mutually rigid via a `length` constraint
  between **every pair** (n points → n(n−1)/2 constraints: 3 for a triad, 10 for 5 points).
  Keep welds **≤ ~5 points** so the pairwise count stays small; revisit with a single
  shape-matching constraint only if a weld must be larger. A weld of ≥3 non-colinear points
  is a rigid body with an emergent frame.
- **`Joint(weld_parent, weld_child, theta0=0)`** — revolute hinge as a **shared edge**
  between the two welds (2 shared points = axis + pivot). Fallback `pivot=…, axis=…` for
  axes without natural points. Carries `theta_build` (rest offset).
- **`actuator`** (part of a joint) — a **servo**: drives the joint's angle toward
  `theta_target` with compliance `alpha`. The kinematic core is "rotate the downstream
  subtree about the shared edge" (see below); the XPBD form applies it compliantly so
  gravity/contact/other joints all reconcile in one solve. Action: `theta_target += gain·Δ`;
  estimation: `theta_target = encoder`.
- **`anchor(point, at)`** — pin a point to a world position (fixed root). Arm anchors its
  base; gnoci omits it (anchored by foot contacts).
- **`Contact(point, plane_z, active=None)`** *(implemented, step 8)* — asymmetric floor and
  the **only** constraint that edits `x_previous` (contact is velocity-level; Verlet velocity
  is `x_curr − x_prev`). No-slip pin in x,y while in contact (`x_curr.xy = x_prev.xy` → zero
  in-plane velocity); one-sided push-up in z (snap `z_curr` **and** `z_prev` to the floor →
  zero z-velocity, no bounce); free above the floor (the foot can lift). Hard kinematic
  projection (ignores `lam`/`alpha`); runs last and iterates against the foot weld so two
  pinned points anchor the stance foot → the floating base (Note 8). Activation: geometric
  `z < plane_z` (modeled), or the foot's **touch sensor** via `active` (measured). Snap is
  non-smooth (Note 5); smooth/compliant floor deferred. See `constraints/contact.py`.
- **`TiltImu(torso_weld, at=point, up_body)`** *(implemented, step 7)* — an IMU sits on the
  torso weld at a point. The accelerometer (gyro-fused) measures **which way is up, in the
  body frame** — a unit vector `up_body` whose 2 DOF *are* pitch & roll. The constraint reads
  the weld's emergent frame `R`, carries the measurement to world (`v = R·up_body`), and
  **rotates the whole subtree about the IMU point** by the shortest turn that stands `v` back
  onto world-up. The rotation axis is `u_w × v`, hence always **horizontal**, so the
  correction is **pure tilt and injects zero yaw** — no Euler decomposition, no gimbal. Same
  rotate-about-a-pivot primitive as the actuator (inverse mass 1, `alpha` = sensor softness);
  measured scope, so the whole robot re-orients rigidly (joint angles/lengths preserved).
  *Not* a "joint to the world": the world has no points to share an edge with. Yaw (about
  vertical) is unobservable from accel+gyro and left free; a future **`HeadingImu`** (from a
  magnetometer) will rotate about the *vertical* axis to set heading — the clean tilt-vs-
  heading seam (PLAN Note on the IMU split; see `constraints/imu.py`).

### How the actuator drives a joint
Kinematic core (the stiff limit / warm start / encoder-FK posing):
```python
P, Q = joint.shared_points            # the hinge edge
n = (Q - P) / norm(Q - P)             # axis, straight from the two shared points
for r in joint.downstream_points:     # child weld's OFF-axis points + all descendants
    r[:] = P + rodrigues(n, delta) @ (r - P)
# P, Q (and the parent) do not move; descendant axes are downstream points, so they
# rotate along automatically — propagation is free.
```
In the **servo / XPBD** form the action moves `theta_target`; the constraint
`C(x) = theta(x) − theta_target` is solved each iteration with compliance `alpha`
(`alpha=0` → reaches target exactly; `alpha>0` → lags = softness). The correction `∇C` for a
downstream point is the rotational tangent about the edge, so each XPBD pass *is* a partial
rotation about PQ — the same primitive, damped. Angle readback:
```python
u_par = perp(parent_offaxis - P, n);  u_chi = perp(child_offaxis - P, n)
theta = atan2(n·(u_par × u_chi), u_par·u_chi) - theta_build
```
Smooth (Rodrigues + atan2) except when an off-axis point lies on the axis — prevented by
sizing.

### Triad & edge sizing (conditioning)
Non-colinear gives a frame that *exists*; **small** gives one that's *noisy*. A read-off
orientation has resolution ≈ (position error) / (triad or edge size) — the geometry is a
lever arm turning position error into angle error, and the same 1/size scales the
*gradient*. So: make triads and shared edges **as large as geometry allows**; where a body
is thin (a short leg segment), add a deliberate **outrigger point** to give the triangle
area. The stacked hip hinges (~5 cm) are the worst case to watch. Edges double as both axis
and frame, so one sizing decision covers both.

XPBD update (per constraint, per iteration):
```
α̃ = alpha / dt²
Δλ = (−C − α̃·λ) / (Σ w_i |∇C_i|² + α̃)
x_i += w_i · Δλ · ∇C_i          # w_i = inv_mass_i
```

### Simulation step
```
state_t, action_t (joint Δθ)
        │  Verlet:  x_new = 2·x_cur − x_prev + a·dt²   (a = gravity + …)
        │  action (sim):       theta_target += gain·Δ
        │  measurement (est):  theta_target = encoder; TiltImu up_body = IMU;
        │                      Contact active = touch switch
        ▼
  ONE Gauss–Seidel loop (XPBD, solver_iters passes, ordered) over BOTH modeled and
  measured constraints — weld rigidity, joints/actuators (→ theta_target), anchor
  (fixed root), TiltImu, contact (active = geometric in sim, = touch in est). Measured
  ones are hard (reasserted each pass); their effect propagates by iteration, not a sweep.
        ▼
  state_{t+1}
        └──► (A) world-model: state = state_{t+1} + NN_residual(state, action)   [later]
        ▼
  Velocity readout:  v_torso = (x_torso_{t+1} − x_torso_t) / dt
```

---

## Proposed architecture / modules

```
src/tac/
  core/
    state.py        # positions, inv_mass, theta_target/build, weld+joint metadata
    integrate.py    # Verlet integration (gravity + global accel); positions only
    geometry.py     # weld/frame helpers (weld_frame). (builder helpers deferred, Note 13)
    rotate.py       # rodrigues(axis, δ); rotate-subtree-about-edge primitive
  constraints/
    base.py         # interface: project(state)->state, alpha, scope, order
    length.py       # length(a,b,L)  — the weld atom
    weld.py         # rigid cluster (pairwise lengths / optional shape-match)
    joint.py        # shared-edge revolute + servo actuator + angle readback
    anchor.py       # fixed-root pin
    contact.py      # asymmetric floor (simulated: no-slip xy, one-sided z)
    imu.py          # TiltImu: torso pitch/roll by gravity-vector alignment (yaw free)
    solver.py       # one GS loop, solver_iters passes, ordered (modeled + measured together)
  estimation/
    measurement.py  # set sensor-driven targets each step: encoder->theta_target,
                    #   IMU->TiltImu.up_body, touch->Contact.active (no-slip memory is x_prev)
    velocity.py     # torso/CoM velocity readout
  world_model/      # (later) residual.py, rollout.py
  robots/
    arm.py          # 2-DOF arm (shared-edge joints, world-z)
    gnoci.py        # gnoci point/weld/joint model, by hand
tests/
pyproject.toml
```

### API sketch
See **`gnoci_desc.py`** for the current working sketch of the user-facing API — the full
gnoci model expressed as points → welds → shared-edge hinges → actuators + measured
constraints (IMU, contact, encoders), plus a `Simulator` and a forward/estimation rollout
loop. It's a **forward-declared outline** (the `tac` package isn't implemented yet) and is
deliberately kept **loose**; treat it as the API target, not a spec. No higher-level
builder — gnoci is written out by hand (a `segment()`/`hinge()`/`mirror()` builder is
deferred, Note 13). The **2-DOF arm** will use the same primitives (shared-edge hinges,
`anchor` on the base, no IMU/contact).

---

## Notes (deferred design)

1. **Estimation = one GS loop, no tree (settled — replaces the earlier tree sweep).** An
   earlier draft made the estimation correction a directed **tree sweep** (forward root→leaf
   FK from IMU + encoder angles; backward leaf→root from contact to fix the base, repeated
   `outer_iters`). We dropped it: it's over-engineered, and almost all of it is just plain
   Gauss–Seidel. The model now:
   - **Modeled and measured constraints share one GS loop** and are projected every iteration.
     Measured ones are **hard** (`alpha=0`), so they reassert each pass = "fix this value,
     resolve the rest around it" (Decision 13). Encoder → `theta_target`; IMU → `TiltImu`
     `up_body`; touch → **`Contact.active`** (same `Contact` as sim, fed by the switch).
   - **Propagation is emergent, not engineered.** A pinned foot relocates the floating base
     because GS carries the pin **up the rigid chain** over iterations; IMU re-orients the
     legs because the joints reconcile. No downstream graph, no forward/backward direction —
     so **no tree**. (The only "downstream" left is each actuator/IMU's *own* point set it
     rigidly rotates — a local efficiency choice, hand-specified, not a propagation graph.)
   - The only true unknowns each step are the **floating-base DOF** (torso xyz + yaw); GS
     recovers them from the foot contacts + IMU orientation.
   - **Cost to watch (pure-PBD):** stiff constraints propagate ~one hop/iteration, so a foot
     relocating a base 5 joints up needs enough `solver_iters`; the rigid-subtree rotations
     (actuator/IMU) move big chunks per pass and help. If a deep chain is too slow, a
     directed **rigid-translation warm-start** of the base from the stance foot is the
     optimization to add — an optimization, not the architecture.
   - **Order/priority** still matters in a hard GS loop (keep the `order` key): measured last
     so it wins ties; compliance later (Note 2) turns ties into weighted fusion.
   - **Stance-foot contact (measured) = the same `Contact`, with `active = touch` — no new
     constraint, no held state.** `Contact`'s no-slip is *relative* (`x_curr.xy = x_prev.xy`)
     and its grounding snaps `z` to the floor, so the "where did the foot land" memory is just
     **`x_previous`**, which already propagates with the state through the rollout — nothing for
     a manager to hold. Applied hard each frame it keeps the foot stationary (zero velocity, no
     bounce), the torso is *placed* by GS from that pin + the encoder-posed rigid leg, and
     `v_torso = Δx/dt` across consecutive consistent states is clean. Sim vs est differ only in
     what drives `active`: geometric `z < floor` (sim) vs the touch switch (est). *(An absolute
     frozen-target `Anchor` would also work and avoids tiny float drift over a very long
     unbroken stance — kept as an optional optimization, not the default.)*

2. **Soft measurement fusion.** Currently hard (frozen measurement during the modeled
   solve). Later: per-sensor trust via XPBD compliance — measured constraint with
   `alpha > 0` ↔ inverse sensor confidence. Define interaction with the hard modeled solve.

3. **CoM & masses.** If the policy needs CoM, define it as the inverse-mass-weighted point
   average and learn the masses. Until then read velocity off the torso.

4. **Axis-rotation primitive.** Core op: "rotate a subtree's points about a world edge."
   Build/start axes are `n ∈ {x̂,ŷ,ẑ}` (Rodrigues simplifies); keep the helper general
   (`rotate(points, pivot, axis, δ)`).

5. **Contact z-snap differentiability.** Snapping `z_prev` is non-smooth; confirm gradients
   behave (or provide a smooth/straight-through variant) since it's on the velocity-estimate
   critical path.

6. **Neural model interface.** Keep `step` and `correct` separable so either
   "predict sensors → measured constraints" or "predict x deltas → re-solve" plugs in.

7. **Rotational integration (deferred).** Orientation is *solved/propagated, not
   integrated* — fine for estimation (IMU + contact always anchor the torso). The one case
   needing true angular-momentum integration is a **world-model rollout through an airborne
   phase** (both feet off, no encoders/IMU): the torso tumble is unobserved. Add a minimal
   rotational integrator — likely a **quaternion on that one body** — only if/when flight
   phases matter. Not on the estimation path.

8. **Double support = closed loop.** Both feet planted ⇒ legs + ground form a kinematic loop;
   no single pass satisfies both planted feet at once. The **one GS loop** (Note 1) handles it
   by iterating: the two feet's `Contact(active=touch)` + IMU + encoders are (mildly) over-
   constrained, and GS settles to the consistent compromise (weighted by compliance once
   soft, Note 2). Just needs enough `solver_iters` to converge — no special sweep.

9. **Weld point-count budget.** Welds multiply points (~3× over single-point segments).
   Points are cheap (positions + pairwise lengths), but keep welds ≤ ~5 points, and watch
   the extra point-sharing where many welds meet at a pivot (`hip_l` lives in torso + yoke
   + hip).

10. **Joint limits (deferred).** Real servos have angle ranges. Add **min/max θ** per joint
    as one-sided constraints (active only when θ exceeds the range), analogous to the
    one-sided floor. Needed, but later — not on the first estimation milestone.

11. **Renderer (near-term, before gnoci).** A 3D viewer to **sense-check geometry**
    (hand-typed point coordinates are where silent bugs hide — e.g. "is forward +y?") and,
    later, to **play back rollouts / trajectories**. Doubles as a structural assert: every
    `Hinge` must find **exactly 2 shared points**, every weld **≥3 non-colinear** — throw if
    not. Should draw: points, weld edges, each hinge's shared edge (= axis) with a ±
    direction arrow, contact points, the IMU site, and the floor plane (`z=-0.23`).
    **Tool choice is open** — matplotlib 3D is fine for a static pose but weak for animated
    rollouts and interaction. Since we'll want to scrub **time-series rollouts** (and batches)
    from a differentiable sim, prefer something built for that: **`rerun`** (rerun.io —
    temporal 3D logging, timeline scrubbing, batch-friendly, good PyTorch fit) or
    **`meshcat`** (browser 3D, common in robotics); reuse of MuJoCo's viewer is awkward since
    we model point clouds, not MJCF bodies. Decide when we build it; lean `rerun` for the
    rollout story. Keep the geometry it reads as plain data so the renderer doesn't prejudge
    the real `core/` types.

12. **Out-of-plane wobble under fast rotation — diagnosed & resolved.** Found building the
    2-DOF arm (steps 1-6): in-plane the tip tracks the analytic arc to ~1e-6, but a **~7e-4
    (sub-mm) out-of-plane (z) wobble** appeared during fast rotation — **independent of solver
    iteration count and float precision** (same in float32/float64, 25 vs 200 iters), so a
    **method artifact, not a convergence/precision bug**. Mechanism: Verlet linearly
    extrapolates the *rotating* rigid body off the constraint manifold (chord-vs-arc
    overshoot, in-plane); the solver repairs that in-plane length error along the weld's
    edges, but **every weld was a flat plate in the y=0 plane**, so the tip is held only by
    edges lying in that sheet — near z=0 the z-restoring gradient is only **second order** in
    the tilt, so z is barely constrained and `x_previous` (the implicit velocity) **sustains**
    it step to step. (Not a base/anchor issue: the base is hard-anchored and provably still to
    1e-5; the wobble is in the *free distal* points.)
    **Resolution (confirmed empirically).** Weld one **out-of-plane point into each link** (a
    `+y` offset giving each weld real 3D volume instead of a flat sheet, wired into the hinge
    downstream set so it rotates rigidly with its link). This gives the tip an edge with a
    genuine `y` lever, making the z-restoring force **first order**: peak z-wobble drops **~3
    orders of magnitude** (6.7e-4 → 6.5e-7 at offset 0.3, into the in-plane error floor), with
    in-plane accuracy and weld rigidity unaffected. Exposed as `build_arm(..., oop=offset)`
    (default `oop=0` = the original flat arm). **Implication for gnoci:** thin / near-planar
    segments need deliberate **out-of-plane outrigger points** — area in all three dimensions,
    not just in-plane (ties straight back to the "Triad & edge sizing" note). *Caveat:* a
    single `+y` point breaks the arm's mirror symmetry about y=0; a symmetric `±y` pair is the
    balanced variant if a residual y-bias ever shows up (costs link1 a 6th point, over the ≤5
    budget in Note 9). Other levers — no longer needed for the arm but still relevant once
    floating-base / contact dynamics extrapolate harder: **substepping** (chord overshoot
    falls ~quadratically — Decision 5), a brief **post-solve re-projection**, or **XPBD
    velocity damping** to bleed a sustained mode.

13. **Build-API helpers (deferred — not building them now).** Earlier drafts imagined
    construction helpers — a `hinge(parent, child, pivot, axis_dir)` that *materialises* the
    two shared-edge points + off-axis frame points, plus `segment()` / `mirror()` for the
    left/right legs. These are **construction-time conveniences only**: they do not add or
    change any runtime primitive — a joint is still a `Hinge` (reads θ off two welds sharing
    an edge) + an `Actuator` (servo), and gnoci's 3-DOF hip is still **three `Hinge`+`Actuator`
    pairs** through two small intermediate welds (yoke, hip), exactly as it would be by hand.
    Since gnoci is very likely the **only** model this library builds, the helper earns its
    keep only across a *second* robot, so we **place gnoci's points/welds/shared-edges by
    hand** (like the arm) and skip the builder. The one substantive thing old step 7 also
    covered — validating **stacked, non-parallel (orthogonal) hinges** (the arm only had
    parallel ẑ axes) — is not deferred; it folds into the by-hand gnoci build + its test
    (build step 9). Revisit the helper only if a second robot (or much hand-edited geometry
    pain) appears; the structural asserts it would host (exactly 2 shared points per hinge,
    ≥3 non-colinear per weld) live in the renderer for now (Note 11).

---

## Resolved

- **Foot:** 2 contact points per foot → **triangle foot weld** (ankle + fwd + back).
- **Root:** support both. Fixed root = `anchor` (arm); gnoci is floating-base (foot contacts).
- **gnoci build:** by hand in `robots/gnoci.py`. No MuJoCo parser for now.
- **Estimation solve:** **one GS loop** over modeled + measured constraints (measured = hard,
  sensor-driven, reasserted each pass); propagation is emergent, **no tree / no sweep** (Note 1).
  Stance-foot contact = the same `Contact` with `active = touch` (no-slip memory is `x_prev`; no
  manager, no held state).
- **Yaw:** IMU is accel+gyro only; assume yaw captured "well enough" by modelling/contact;
  add a magnetometer constraint later if velocity estimates drift. Pitch/roll from IMU.
- **Floor:** plane at `z = -0.23` (match the scene).
- **Angle sign:** follow MuJoCo's right-hand-rule convention about the axis (here the
  shared edge, oriented `P→Q`).
- **Units:** meters; angles in radians internally.
- **Arm joints:** **shared edges** (vertical, world-z), like gnoci — the arm is built in 3D.
  `pivot+axis` stays only as a rare escape hatch, not used by the arm.
- **Weld representation:** pairwise `length` constraints; keep welds **≤ ~5 points**
  (≤ 10 constraints). Shape-matching only if a weld must be larger.
- **Actuator strength:** **action gain + servo compliance** (not a scaling of point positions).
- **Axis calibration:** learn the **rest positions of a joint's two shared (axis) points**,
  governed by a **per-parameter learnability scale** so geometry drifts only slowly.
- **IMU:** attached to the torso weld **at a point**; the constraint **rotates the weld about
  that point** to match measured pitch & roll (yaw free). *Not* a joint-to-world (the world
  has no points). Hard-freeze vs small compliance during correction → folds into Note 2.
- **inv_mass + compliance:** **keep both** learnable; watch identifiability, regularise or
  pin one if fitting is unstable (see Learnable parameters).
- **Joint limits:** needed; add later as one-sided min/max θ constraints (Note 10).
- **End-to-end target (H):** the first gnoci milestone = replay a logged/sim trajectory, run
  estimation mode, and compare estimated torso velocity to ground truth. This defines
  "working".

---

## Open questions

_None outstanding — all resolved above. New questions go here as they arise._

---

## Suggested build order

**Phase 1 — engine on the 2-DOF arm** ✅ *complete*
- [x] **1.** `State` (positions only) + Verlet integration; test: free point moves at constant velocity, batched.
- [x] **2.** `length` + XPBD solver; test: 2-point link holds length, gradients flow.
- [x] **3.** `weld` (rigid cluster) + frame readback; test: a triad stays rigid and its read-off frame is stable under motion.
- [x] **4.** `anchor`; test: anchored point stays put under gravity.
- [x] **5.** `Joint` (**shared edge**) + `actuator` — rotate-subtree-about-edge primitive + servo to `theta_target` (single world-z edge); test: driving θ rotates the downstream weld to the target angle; encoder-set θ reproduces a pose.
- [x] **6.** Assemble the **2-DOF arm** (shared vertical edges); action = joint Δθ; test: tip traces expected arc, `tip_velocity = Δx/dt` matches analytic. **(milestone)** *(out-of-plane wobble found, diagnosed & resolved — Note 12.)*

**Phase 2 — orientation, contact, gnoci**
- [x] **7.** `TiltImu` pitch/roll on the torso weld (gravity-vector alignment, yaw-free by construction; `HeadingImu` deferred to a future magnetometer); test: imposed pitch/roll orients the body, weld stays rigid, a purely-yawed body is left untouched. ✅
- [x] **8.** `Contact` (asymmetric floor, no-slip xy + one-sided z, touch-sensor override); test: foot pins in x,y, can lift, falls and rests without bouncing, batched mixed contact. ✅
- [x] **9.** `robots/gnoci.py` by hand (idealized geometry; 25 pts, 11 welds, 10 shared-edge hinges, mirrored) + its **modeled** assembler / `step_sim` (like `Arm`); `scene_from_gnoci` + `examples/render_gnoci.py`. Tests: assembles & calibrates to 0, holds its static pose, **stacked orthogonal hip (yaw⊥roll⊥pitch) resolves to independent targets with legs rigid (twist included)**, full 5-joint leg reaches targets, batched. *(Metric calibration to MuJoCo deferred; standing-under-gravity contact stability is a later concern — Note 12 levers.)* ✅
- [ ] **10.** Torso velocity readout; test: known motion → correct velocity.
- [ ] **11.** Measurement correction — gnoci's **estimation** assembler + `step_estimate`, one GS loop: encoder→`theta_target`, IMU→`TiltImu`, touch→`Contact.active` (reuses steps 7+8; no manager, no held state — the no-slip memory is `x_prev`). Test: imposed measurements reproduce a known pose, **and both feet planted (double support) converges** (the closed loop, Note 8).
- [ ] **12.** End-to-end velocity estimation on sim data vs MuJoCo ground truth (Q-H).

**Phase 3 — learning**
- [ ] **13.** Learnable parameters (action gain, lengths, inv_mass, compliances, axis-point rests) + fitting to data.
- [ ] **14.** Neural residual + (optional) differentiable rollout.
- [ ] **15.** `torch.autograd.gradcheck` across the pipeline.
