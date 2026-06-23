# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`tac` is a minimal, differentiable **XPBD (position-based dynamics) physics engine** in
PyTorch. Its end goal is to be a **state estimator** for a bipedal robot ("gnoci") —
estimating torso/CoM velocity from onboard sensors (IMU, joint encoders, foot contacts)
— and to double as a differentiable physics prior for a learned world model.

**`planning/AI-PLAN.md` is the design authority.** It is a long, settled design document
covering the core abstraction (points / welds / constraints), the gnoci robot, the
estimation solve, learnable parameters, and a phased build order. Read it before making
non-trivial design decisions — it records *why* choices were made and what is deliberately
deferred. The code on disk is well behind the plan (see "Current state" below).

## Core abstraction (from the plan)

Everything is built from three things, nothing else:

1. **Point** — just a position `(x, y, z)`. No stored orientation, no quaternion, no frame.
2. **Weld** — a set of points held mutually rigid by a `length` constraint between *every
   pair*. A weld of ≥3 non-colinear points carries a full orientation *emergently* from its
   point positions. The weld is the rigid-body primitive; orientation is read off, never stored.
3. **Constraint** — a relationship between points, projected by the XPBD solver. The atom is
   `length`; welds, joints (shared-edge revolute hinges), anchors, contact, and IMU all build
   on it. A revolute joint = two welds sharing two points (the shared edge *is* the hinge axis).

Consequences worth keeping in mind: no per-point quaternions, the solver is **positions-only**,
state is **batched from day one** as `(B, N, 3)`, and velocity is implicit (Verlet:
`v = x_current − x_previous`).

## Current state of the code (mid-reimplementation)

The working tree is a clean-slate reimplementation that is **earlier-stage than the plan and
earlier than the previous commit** (the git history's "initial implementation" had many more
modules that were intentionally deleted). What exists and works today:

- `src/tac/core/state.py` — `State(points, inv_mass, batch_size)`; holds `x_current`,
  `x_previous` `(B, N, 3)` and `inv_mass` `(B, N)`.
- `src/tac/core/sim.py` — `inertial_step` (Verlet), `constraint_step`, and `Simulator`.
- `src/tac/constraints/length.py` — `Length(state, i, j, alpha)`; XPBD distance constraint
  with `apply(lam, dt)` and `get_length()`. This is the canonical XPBD update — match it.
- `src/tac/constraints/weld.py` — `SimpleWeld` (all pairwise lengths) and `FacetedWeld` (a
  list of faces, each a `SimpleWeld`); both expose `get_constraint_set()`.
- `src/tac/viz/renderer.py` + `convert.py` — `pyvista`-based live `Renderer` and conversion
  helpers (`state_2_points`, `weld_2_cells`).
- `examples/box.py`, `face.py`, `two_points.py` — runnable scenes.

**Known rough edges (refactor in flight, fix if you touch them):** `examples/box.py`,
`examples/face.py`, and `src/tac/viz/tests/test_2mesh.py` import `weld_2_mesh`, which no
longer exists in `convert.py`. `examples/two_points.py` and `examples/box.py` call
`Simulator(...)` with a stale signature (the current `Simulator.__init__` takes
`(constraints, num_iter, dt)`, not a leading `state`). `sim.py` still has a debug `print`.
Don't assume an example runs end-to-end without checking.

A previous structure (now deleted) had `src/tac/robots/`, `viz/scene.py`, and
`examples/render_*.py`; the memory notes `[[tac-example-pattern]]` and `[[no-tests-unless-asked]]`
describe *that* layout, so treat their file paths as the intended target shape, not the
current tree.

## Commands

This repo uses a checked-in virtualenv at `venv/` (Python 3.12). Prefer `venv/bin/python`.

```bash
# install editable with dev + viz extras (pytest, numpy<2, pyvista)
venv/bin/python -m pip install -e ".[dev,viz]"

# run the tests — tests are CO-LOCATED under src/tac/**/tests/, NOT in a top-level tests/ dir.
# pyproject's `testpaths = ["tests"]` is stale (that dir was deleted), so point pytest at src:
venv/bin/python -m pytest src

# a single file / single test
venv/bin/python -m pytest src/tac/core/tests/test_state.py
venv/bin/python -m pytest src/tac/constraints/tests/test_length.py::test_apply_length

# run an example (pyvista opens a window; needs a display)
venv/bin/python examples/box.py
```

`numpy<2` is load-bearing — see the comment in `pyproject.toml`: the last x86-macOS torch
wheel (2.2.2) is built against the numpy 1.x ABI and crashes under numpy 2.x. Keep the pin.

## Conventions

- **Do not write pytest files unless explicitly asked** ([[no-tests-unless-asked]]). Deliver
  the model/example and verify with a quick headless run + inline check instead. (Test files
  that already exist are fine to run and maintain.)
- New shapes follow the existing example pattern ([[tac-example-pattern]]); constraint
  application order is weld rigidity → actuators → anchors → contact.
- Tensor shapes are documented inline (e.g. `# shape, (b, 3)`) in the constraint code — keep
  that habit; shape bugs are the main hazard in the batched math.
- Target devices are CPU + CUDA only (no MPS).
