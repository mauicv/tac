# gnoci performance analysis

Why `gnoci.py` runs at ~1.5 s per `step()`, and what actually drives it.

## Measured baseline

Single-threaded, `Gnoci()` default config (`B=1`, 100 points):

| metric | value |
|---|---|
| wall time per `step()` | **1.55 s** |
| simulated constraints | 585 (`564 Length`, `20 Joint`, `1 FloorContact`) |
| `apply()` calls per `step()` | **5 850** |
| time per `apply()` | **264 µs** |
| time per `Length.apply()` in isolation | **300 µs** |

`step()` runs 5 inner sub-steps, each sub-step runs `num_iter=2` solver
iterations, each iteration loops over all 585 constraints in Python:

```
5 substeps × 2 iters × 585 constraints = 5 850 apply() calls / step
```

## Root cause: eager per-constraint dispatch on scalar-sized tensors

The cost is **not** real numerical work — the tensors are tiny (`(1, 3)` and
`(1, 1)`). It is **PyTorch's fixed per-operation dispatch overhead**, paid once
per ATen call, multiplied by a huge number of calls.

`Length.apply()` issues ~20 separate PyTorch ops (two fancy-index reads, a
norm, a clamp, ~10 arithmetic ops, two list-index gathers, two in-place
advanced-index writes), plus ~30 Python attribute/method lookups. None of that
work is amortized over an array, because each constraint only touches 2 points.

Per-op dispatch cost measured on this machine (B=1):

| op | cost |
|---|---|
| basic slice `x[:, i]` | 4.6 µs |
| **list index `inv_mass[:, [i]]`** | **16.1 µs** |
| **in-place adv-index add `x[:, i] += d`** | **13.1 µs** |
| `vector_norm` | 3.5 µs |
| `clamp` | 3.5 µs |
| tiny `a + a` | 2.3 µs |

So each `Length.apply()` is roughly:

```
2 × list-index (32 µs) + 2 × in-place adv-index write (26 µs)
+ 2 × slice reads (9 µs) + ~10 arithmetic/norm/clamp ops (~30 µs)
+ Python overhead  ≈  300 µs
```

Across a full step that is on the order of **120k–230k op dispatches**, each a
few µs → ~1.5 s. The number of constraints matters, but the real multiplier is
**ops-per-constraint × iterations × substeps**, all in an interpreted loop.

## Ranked contributing factors

1. **Python Gauss–Seidel loop over individual constraint objects (dominant).**
   585 objects × 10 passes/step, each doing ~20 eager ops. This is ~95% of the
   time. Everything below is secondary.

2. **Avoidable expensive ops inside the hot path.**
   - `self.state.inv_mass[:, [self.i]]` uses **list indexing** (a gather, 16 µs)
     to fetch a value that is **constant** — `inv_mass` never changes. Same for
     `self.L` and `a_tilde = alpha/(dt*dt)`; all recomputed/re-fetched every
     call.
   - In-place writes through advanced indexing (`x[:, self.i] += ...`) are
     13 µs each (an `index_put`), vs ~5 µs for a basic slice.

3. **B=1 wastes the batched design.** State is `(B, N, 3)` specifically to
   vectorize, but with `B=1` there is nothing to amortize dispatch over. Larger
   `B` is nearly free per-constraint (same dispatch count), so throughput per
   sample improves a lot — but it does nothing for single-rollout latency.

4. **Minor allocations per call.** `Gravity.apply` rebuilds
   `torch.tensor(self.gravity)` every effect application; `lam` slices
   (`lam_simulated[:, [ind]]`) are list-indexed each iteration.

## What is *not* the problem

- **`PerformanceMonitor` is negligible** — the `with self.perf_rec(...)` wrapper
  measured at **1.3 µs** per call (~0.5% of an apply). It does, however, append
  to an unbounded `events` list, so it leaks memory over long runs; fine for
  profiling, worth disabling for production rollouts.
- **The actual math is cheap** — norms, dots and the XPBD update are trivial at
  this size. There is no numerical hot spot to optimize.

## Remediation, by impact

1. **Vectorize each constraint *type* into batched tensor ops (biggest win).**
   Replace the per-object loop with arrays of indices. Store all length
   constraints as `i_idx (M,)`, `j_idx (M,)`, `rest (M,)`, then per iteration do
   one gather → one norm → one update → one **`index_add_`/`scatter_add_`**,
   processing all M constraints in a handful of dispatches instead of `M × 20`.
   Expected: ~100×+ fewer dispatches.
   - Caveat: true Gauss–Seidel is sequential (each constraint sees the previous
     one's writes). Batching makes it Jacobi-style within a group. To keep
     stability, **graph-color** the constraints into independent groups (no two
     in a group share a point) and do one vectorized pass per color — same
     convergence as Gauss–Seidel, vectorized.

2. **Hoist constants out of `apply()`.** Precompute and cache `a_tilde`, and
   store `wa`/`wb` as plain floats or pre-sliced tensors at construction time
   instead of re-gathering `inv_mass[:, [i]]` every call. Cheap, immediate win
   even before vectorizing.

3. **Avoid advanced-index in-place writes** in the scalar path; prefer basic
   slicing, or move to the `index_add_` batched form from (1).

4. **Gate the `PerformanceMonitor`** behind a flag so production runs don't wrap
   every apply or grow the events list.

5. **Use `B>1`** when running many rollouts — dispatch is shared, so cost per
   sample drops sharply (does not help one-shot latency).

`torch.compile` / TorchScript would cut per-op Python overhead but cannot fix
the fundamental "thousands of tiny ops" shape — (1) is the structural fix.
