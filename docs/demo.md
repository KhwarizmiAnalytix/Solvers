# ConvergenceDemo: native vs. third-party math and performance

Companion to [`Testing/Benchmark/ConvergenceDemo.cpp`](../Testing/Benchmark/ConvergenceDemo.cpp),
which solves the Rosenbrock problem with every available backend and prints
each one's status, iterations, `||r(x)||`/`||grad||`, `f(x)`, and wall-clock
time. This note analyzes those numbers: why native and third-party backends
differ in performance, and the mathematics behind the native Levenberg–Marquardt
feature set — geodesic acceleration, bold acceptance, and the separate RNC-LM
algorithm — with the measured iteration/time cost of each. Full option
reference and derivations live in [levenberg-marquardt.md](levenberg-marquardt.md)
and [rnc-lm.md](rnc-lm.md).

## Native vs. third-party: measured performance

The README's captured run (macOS/arm64, Release, Clang; see
[README "Example results"](../README.md#development)), `x0 = (-1.2, 1.0)`,
minimum at `(1, 1)`:

| Backend | status | iterations | time |
| --- | --- | --- | --- |
| `native/levenberg_marquardt` | converged | 32 | 99 µs |
| `native/gauss_newton` | converged | 10 | 5 µs |
| `ceres` | converged | n/a | 616 µs |
| `ipopt` | converged | 47 | 28.9 ms |
| `petsc_tao/BRGN` | converged | 18 | 52.9 ms |

| Backend / algorithm | status | iterations | time |
| --- | --- | --- | --- |
| `native/lbfgs` | converged | 86 | 37 µs |
| `ipopt` | converged | 47 | 21.9 ms |
| `nlopt/LD_LBFGS` | converged | n/a | 753 µs |
| `petsc_tao/LMVM` | converged | 37 | 628 µs |

These are unreplicated single-run timings for one problem and one machine —
illustrative of the architectural gap, not a portable benchmark.

The gap is architectural, not an accident of this one problem:

- **Native backends link directly into the calling process.** The LM
  normal-equations path (`damped_step_solver` in
  [`include/detail/eigen_support.h`](../include/detail/eigen_support.h))
  allocates nothing inside the iteration loop — all workspace is created once
  per solve and products use `noalias()`. This is enforced by `LmNoAllocCheck`
  compiling the kernel under Eigen's `EIGEN_RUNTIME_NO_MALLOC` runtime guard.
  That absence of per-iteration allocation, adapter translation, and
  general-purpose sparse machinery is why `native/gauss_newton` finishes in
  single-digit microseconds and `native/levenberg_marquardt` in double
  digits.
- **Ceres, Ipopt, NLopt, and PETSc/TAO pay for generality**: sparse autodiff
  graphs, trust-region/line-search machinery sized for problems with
  thousands of parameters and constraints, and (for Ipopt/PETSc) process and
  library boundaries built around large-scale interior-point or barrier
  methods. On a 2-parameter problem that machinery is pure overhead, which is
  why the gap spans three orders of magnitude (µs vs. ms) rather than a
  constant factor.
- **The trade native makes for that speed**: no general constraints, no
  sparsity exploitation beyond Eigen's dense path, and (per
  "Remaining implementation limits" in [levenberg-marquardt.md](levenberg-marquardt.md))
  no rank-revealing solve or factorization-status check. The performance
  advantage is specific to well-scaled, moderate-dimension, unconstrained or
  least-squares problems — not a general claim about the underlying
  algorithms.

Iteration counts are not comparable across the two tables: `||r(x)||` and
`||grad||` come from different problem formulations (least-squares vs.
general objective), and Ceres/NLopt never populate `solver_result::iterations`
(see [README, "Results and current limits"](../README.md)).

## Native LM feature mathematics and cost

Baseline native LM is first-order: each step is the damped Gauss-Newton
velocity `v` solving `A v = -g`, with `A = JᵀJ + λD`, `g = Jᵀr`, accepted only
when it reduces cost (`ρ > 0`, the ordinary gain ratio below). Geodesic
acceleration and bold acceptance are independent modifications of,
respectively, the proposed step and the acceptance rule.

### Geodesic acceleration — second-order step correction

From Transtrum & Sethna (arXiv:1201.5885, Eqs. 12/14/15, 19/20), a directional
second derivative of the residual along `v` is estimated by finite difference:

```
r_vv ≈ (2/h²)[r(θ + h v) − r(θ) − h J v],   A a = −Jᵀ r_vv,   s = v + a/2
```

instead of proposing `s = v` alone. The guard `||a|| ≤ α ||v||`
(`geodesic_acceleration_threshold`, default `0.75`) rejects the entire trial
— correction included — when the second-order term is too large relative to
the first; that is a signal the finite-difference step `h`
(`geodesic_acceleration_step`, default `0.05`) is too coarse for the local
curvature, not evidence to silently fall back to the uncorrected step.

**Why it helps**: on a curved residual manifold (Rosenbrock's valley is the
textbook case), the first-order velocity systematically overshoots the
curve, so plain LM keeps re-damping and re-stepping along the valley floor.
The second-order term follows the curve instead of its tangent.

**Cost**: one extra residual evaluation per trial for the directional
difference, and a rejection path that increases damping on failure rather
than reducing to plain LM.

#### Reproduction of the literature

Transtrum & Sethna's full text (arXiv:1201.5885v1, not just the abstract)
benchmarks geodesic acceleration and bold acceptance on 17 MINPACK-2/NIST
nonlinear-least-squares problems (α-pinene kinetics, coating thickness,
exponential/Gaussian fits, thermistor/enzyme/Rat42/Rat43/Bennett5 NIST
problems, a small neural-network fit) — **not Rosenbrock**; Rosenbrock does
not appear in their test suite at all, so the 32-iteration
`native/levenberg_marquardt` row above has no equation-for-equation
counterpart in the paper to check against. The paper's own reported
magnitudes, read directly from its text and figures (12–14), are:

| Feature | Paper's reported effect |
| --- | --- |
| Geodesic acceleration (LM+GA) | "speed ups as large as a factor of 70, with most improvements between a factor of 2 and 10"; comparable reductions in Jacobian evaluations on some problems |
| Bold acceptance alone | speedups "in some cases by a factor of 30," at the cost of a lower success rate on several problems |
| Bold acceptance + geodesic acceleration | Fig. 14 shows the acceleration correction keeps bold acceptance from "becoming lost" — i.e., GA is reported as a stabilizer for bold acceptance, not just an independent speedup; the paper gives no separate numeric table for the combination |

This repository does not yet carry a local ablation (`ConvergenceDemo` run
with `with_geodesic_acceleration(false)` vs. `true`, same problem and
tolerances) to put a measured number against those ranges. That is the gap
to close before claiming this implementation reproduces the paper's speedup
factors rather than just its equations — the equation-level mapping
(Eq. 14 for the corrected step, Eq. 15 for `||a|| ≤ α||v||`, Eqs. 20–22 for
bold acceptance) is checked by the eight regression tests listed in
[levenberg-marquardt.md "Validation"](levenberg-marquardt.md#validation),
but none of those tests measure iteration-count speedup against a baseline.

### Bold acceptance — relaxed acceptance for aligned trajectories

Ordinary acceptance requires positive actual and predicted reduction via the
gain ratio

```
ρ = (||r||² − ||r(θ+s)||²) / (−2 gᵀs − ||Js||²)
```

— a trial that briefly worsens the residual is always rejected, even moving
toward a better region. Bold acceptance (Eq. 22) instead compares the new
residual against the best cost accepted so far, weighted by the cosine
similarity `β` between the new and last-accepted velocity:

```
β = (v_new · v_old) / (||v_new|| ||v_old||)
(1 − β)^b ||r_new||² ≤ min(accepted ||r||²)
```

When the solver has been moving consistently in one direction (`β → 1`),
`(1 − β)^b` shrinks toward zero and the bar for accepting an uphill step
drops — the trajectory's own consistency is treated as evidence that a
temporarily worse point is still progress. This matters on problems where
narrow curved valleys or saddle-like flat regions make ordinary acceptance
reject every trial, re-damp repeatedly, and end in
`solver_status::stalled` once damping hits its ceiling, rather than
`converged`.

**Trade-off**: bold acceptance can return the last accepted iterate even
though it has higher cost than the best point actually visited — the minimum
cost is tracked only for the acceptance test, its parameters are not saved
(see "Remaining implementation limits" in
[levenberg-marquardt.md](levenberg-marquardt.md)). It is a targeted fix for
stalling runs, not a default-on improvement, which is why it defaults to
`false`.

Literature reproduction for bold acceptance is the same situation as for
geodesic acceleration above: the paper's reported "factor of 30" speedup
with reduced success rate (and Fig. 14's qualitative claim that GA keeps
bold acceptance from diverging) comes from the 17-problem MINPACK-2/NIST
suite, not from this repository's Rosenbrock demo, and there is no local
ablation run yet that measures this implementation's own speedup/success
trade-off to compare against that figure.

### RNC-LM — polynomial curve beyond a single second-order correction

Geodesic acceleration is one second-order correction to a straight-line step.
RNC-LM (Liu & Zhang, arXiv:2607.07623v2) generalizes this to a polynomial
curve through order 4:

```
θ_K(t) = θ + Σ t^q c_q / q!,     q = 1..order
G c_1 = −Jᵀr                                          (order 1: the LM velocity)
G c_n = −Σ_{k=0}^{n−2} C(n−2,k) J_kᵀ R_{n−k}           (n ≥ 2, moving-tangent terms)
```

with `G = JᵀJ + λD`. Each higher coefficient reuses the same damped normal
matrix and a single Cholesky factorization for all orders at a given damping
value — e.g. `G c₂ = −JᵀR₂`, `G c₃ = −JᵀR₃ − J₁ᵀR₂`, `G c₄ = −JᵀR₄ − 2J₁ᵀR₃ − J₂ᵀR₂`.
Acceptance is a curve-length/contraction search (Eqs. 32–40): predicted
reduction uses `m(t) = ||r + tJv||²/2` compared against the actual cost along
the curve, with quadratic interpolation in `t` clipped to a contraction
interval, and damping multiplied by 2 for `ρ < 0.25` or divided by 3 for
`ρ > 0.75`. There is no bold-acceptance analogue (monotone cost acceptance
only) and no acceleration-ratio rejection test — the curve-length search
plays that role instead.

**Why higher order helps**: moving-tangent terms (the `J_k` contributions)
capture how the Jacobian itself changes along the curve, which a single
second-order correction cannot. Measured on the library's stiff generalized
Rosenbrock benchmark (`SolversBenchmark rnc-lm`, stiffness `A = 1e6`, two
valley exponents, objective threshold `C < 1e-4`):

| Valley exponent | Order 1 | Order 2 | Order 3 | Order 4 |
| --- | ---: | ---: | ---: | ---: |
| 2 | 9689 | 456 | 33 | 23 |
| 3 | 6985 | 441 | 31 | 27 |

Each order increase is roughly an order-of-magnitude drop in outer
iterations on this stiff valley, where plain LM (even with geodesic
acceleration) needs thousands of steps because a single correction cannot
track the curvature.

#### Reproduction of the literature

Liu & Zhang's Table 1a (arXiv:2607.07623v2) benchmarks exactly this
problem — generalized Rosenbrock, stiffness `A = 1e6`, read at curve
parameter `t_min = 1` so `K = 1` reduces to plain LM — for dimensions
`n = 2` and `n = 3`:

| Method | n = 2 | n = 3 |
| --- | ---: | ---: |
| LM (K=1), paper | 8622 | 8069 |
| LM-GA, paper | 152 | 183 |
| RNC-LM K=2, paper | 152 | 183 |
| RNC-LM K=3, paper | 33 | 33 |
| RNC-LM K=4, paper | 2 | 21 |

Against this repository's own measured run of the same problem family
(table above, `order 1..4`, `valley exponent 2` and `3`):

| Order | Paper (n=2 / n=3) | This implementation (2 / 3) | Agreement |
| --- | --- | --- | --- |
| 1 (plain LM) | 8622 / 8069 | 9689 / 6985 | Close (same order of magnitude, ±13–15%) |
| 2 | 152 / 183 | 456 / 441 | **Diverges**: this implementation takes ~2.4–3× more iterations at order 2 |
| 3 | 33 / 33 | 33 / 31 | **Matches closely** at `n=2`; within 7% at `n=3` |
| 4 | 2 / 21 | 23 / 27 | **Diverges**: this implementation takes substantially more iterations at order 4, especially at `n=2` (23 vs. 2) |

Order 3 is the strongest reproduction point; orders 2 and 4 are not a
reproduction of the paper's numbers, consistent with what
[rnc-lm.md](rnc-lm.md) already states — the implemented curve-length search,
contraction bounds, and damping update are this repository's own choices
(Eqs. 37–40 mapped, not copied), and "no claim is made about ... reproduction
of the paper's Table 1: defaults and implementation details differ." The
divergence at order 4 is the most consequential one to investigate further,
since it is where the paper reports its largest advantage (`K=4` collapsing
to 2 iterations at `n=2`) and this implementation does not reproduce that
collapse.

The paper's other two benchmarks — a reaction-diffusion PINN (Table 2: at
`ν=5`, L-BFGS/LM/LM-GA all stay above 0.95 relative L² error while RNC-LM
`K=3` reaches 6.56×10⁻³) and a 985,160-configuration machine-learning
potential fit (`K=4` reaches a fixed training-RMSE target in 144 iterations
/ 11.39 h vs. plain LM's 387.28 h unfinished run at the 5000-iteration
budget — a 34× wall-clock speedup; `K=3` reaches the same target at 12.8×)
— are both large, domain-specific workloads with no counterpart in this
repository's test or benchmark suite. They are cited here as the literature's
own evidence for *why* higher order matters at scale, not as something this
implementation has attempted to reproduce.

**Cost**: not per-problem-size but per-iteration — each curve point needs
correspondingly higher directional derivatives of the residual (one model
sweep per parameter per derivative order with the bundled Taylor-AD adapter,
or a supplied analytic callback). Order 4 does strictly more work per
iteration than order 1; the win in the table is in outer-iteration count, not
automatically in wall-clock time for cheap residuals. RNC-LM is also a dense,
CPU-only implementation restricted to the least-squares form — no bounds, no
general objectives.

## Summary: cost/benefit by symptom

| Symptom | Mathematical cause | Native feature that addresses it | Literature reproduction |
| --- | --- | --- | --- |
| Fast baseline needed on a well-scaled least-squares problem | — | Native LM defaults (geodesic acceleration on, bold acceptance off) beat adapter overhead by orders of magnitude | n/a (architectural, not a literature claim) |
| Many iterations through a curved valley | First-order velocity overshoots curved residual manifold | Geodesic acceleration (on by default) | Equations mapped and regression-tested; no local ablation yet measures this repo's speedup against the paper's reported 2–70× |
| Repeated re-damping / `stalled` near a saddle or flat region | Ordinary acceptance rejects every uphill trial regardless of trajectory | Bold acceptance | Same gap as above, against the paper's reported ~30× with reduced success rate |
| Still hundreds/thousands of iterations on a stiff, highly curved problem | Single second-order correction insufficient; Jacobian itself varies along the curve | RNC-LM, increasing `order` | Order 3 matches the paper's Table 1a closely; orders 2 and 4 diverge (see comparison table above) |
| Bounds, general constraints, or very large sparse problems | Outside native LM's scope entirely | Ceres / Ipopt / PETSc-TAO, accepting the measured per-call overhead above | n/a |

Reproduce any of these numbers locally before they drive a design decision —
the README and the two linked docs carry the same caveat: unreplicated
single-run timings for one problem and one machine.
