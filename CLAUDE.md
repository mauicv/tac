# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`tac` is a minimal, differentiable **XPBD (position-based dynamics) physics engine** in
**JAX**. The aim is for it to grow into a **state estimator** for a bipedal robot ("gnoci")
— estimating torso/CoM velocity from onboard sensors (IMU, joint encoders, foot contacts)
— and to double as a differentiable physics prior for a learned world model.

## Core abstraction

Two things:

1. **Point** — just a position `(x, y, z)`. No stored orientation, no quaternion, no frame.
2. **Constraint** — a scalar relationship between points, projected by the XPBD solver. The
   atom is `length`; joints (revolute hinges), planarity, and contact build on it.

A **rigid body is not a primitive** — it's just a set of points wired together by pairwise
`length` constraints (see `examples/box.py`, which builds all 28 pairwise lengths across the
8 cube points). Orientation is never stored; it emerges from the point positions.

Consequences: no per-point quaternions, the solver is **positions-only**, state is **batched
from day one** as `(B, N, 3)`, and velocity is implicit (Verlet: `v = x_current − x_previous`).

## Current state of the code (JAX)

- `core/state.py` — `State(x_current, x_previous, inv_mass, control)`, a `flax.struct`
  (immutable). `x_current`/`x_previous` are `(B, N, 3)`, `inv_mass` `(B, N)`, `control`
  `(B, control_size)`. Build with `State.create(x_current, inv_mass, control_size=None)`
  (keeps ≥1 control column so the per-constraint gather never hits an empty axis). Helpers:
  `update_x`, `update_control`, `_get_x_w`.
- `core/sim.py` — `Simulator(constraints, effects, num_iter, dt)`. `step` = `effect_step`
  then `constraint_step`. `constraint_step` runs `num_iter` sweeps, threading a per-constraint
  Lagrange multiplier `lam` through each `constraint.apply(state, lam, dt) -> (state, lam)`.
- `constraints/autograd.py` — **`AutogradConstraint`**, the canonical XPBD constraint and the
  thing to match. A constraint is defined by a scalar measure `C(points)` and its autodiff
  gradient `grad_C = jax.grad(...)`. `apply` does the XPBD update
  `Δλ = −(C + α̃λ) / (Σ wᵢ|∇ᵢC|² + α̃)`, `Δxᵢ = wᵢ ∇ᵢC Δλ`. Notable fields: `rest`
  (`None` ⇒ captured from the build-time state; else an absolute target), `control`
  (per-constraint index into `state.control`, `−1` = uncontrolled, added to the residual),
  `angular_wrap` (wraps the residual into `(−π, π]` — for joint angles), `name`.
- `constraints/builders.py` — `ConstraintBuilder` + `LengthBuilder`, `FloorContactBuilder`,
  `JointBuilder`, `PlanarBuilder`. Each subclass just supplies a `C`. `add_constraint(points,
  alpha, control=-1)` greedily partitions constraints into **point-disjoint sets** (graph
  colouring) so each built `AutogradConstraint` batches a set that can be solved in parallel.
  `build(state)` returns a `list[AutogradConstraint]`.
- `constraints/hard_floor.py` — `HardFloorConstraint`, **not** XPBD: a hard projection that
  snaps penetrating points to `floor_z` and freezes their x/y (zeroes tangential velocity).
  Vectorised over a points array; matches the `apply(state, lam, dt) -> (state, lam)` interface
  so it drops into the same `Simulator` loop (it ignores `lam`).
- `effects/inertial.py` — `Inertia` (Verlet predict: advances `x_previous`, carries velocity,
  applies linear drag) and `Gravity` (adds `g·dt²` to `x_current` only). **Inertia is the one
  effect that advances `x_previous`, and must run before Gravity** — Gravity deliberately does
  not touch `x_previous` so momentum accumulates.
- `viz/renderer.py` — `pyvista`-based interactive `Renderer(state, lines, faces)`.
- `examples/` — `box.py`, `joint.py`, `length.py`, `gnoci.py` (open a pyvista window; need a
  display). `assets/box.gif` is a headless recording of the box rollout, embedded in the README.

## Commands

Checked-in virtualenv at `venv/` (Python 3.12). Prefer `venv/bin/python`.

```bash
# install editable with dev + viz extras
venv/bin/python -m pip install -e ".[dev,viz]"

# run an example (pyvista opens a window; needs a display)
venv/bin/python examples/box.py

# tests are CO-LOCATED under src/tac/**/tests/, NOT in a top-level tests/ dir.
# pyproject's `testpaths = ["tests"]` is stale, so point pytest at src:
venv/bin/python -m pytest src
```

Note: the co-located tests are currently **stale** — they construct `State` with the old
signature and fail at collection/first assert. Don't trust a red test suite as a signal about
your change; verify with a quick headless run + inline check instead (below).

`pyproject`'s `viz` extra pins `numpy<2`, but the JAX tree does not need it (that pin was for
the old x86-macOS torch wheel) and the working env runs numpy 2.x fine — treat the pin/comment
as stale if it gets in the way.

## Conventions

- **The `C` contract is the #1 shape gotcha.** `C: (B, n, k, 3) -> (B, n)` — it must reduce
  over *both* the `k` points in each constraint *and* the coordinate axis, returning **one
  scalar per constraint** (no trailing keepdim `1`). `apply` adds the `[..., None]` itself.
  See `LengthBuilder.C` for the canonical shape.
- **Application order:** effects first (`Inertia` → `Gravity`), then constraints; within the
  constraint list go length/rigidity → joints/actuators → **contact last** (hard/contact
  projections must come after anything that could pull a point back through the floor).
- **Batched from day one** — everything is `(B, …)` and `B > 1` is supported; document tensor
  shapes inline (e.g. `# (B, n, k, 3)`), since shape bugs are the main hazard in the batched math.
- Prefer an **optional constructor param** (e.g. `create(..., rest=None)`, `control=-1`) over a
  post-build `.replace()` when a value can be defaulted-from-state or overridden — `None`/`-1`
  means "derive it", otherwise use what's passed. [[constraint-rest-param]]
- **Do not write pytest files unless explicitly asked** ([[no-tests-unless-asked]]). Deliver the
  model/example and verify with a headless run + inline print/assert.
- Target devices are CPU + CUDA only (no MPS).
