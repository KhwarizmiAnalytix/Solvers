# Changelog

## Unreleased: design-review phases 0 to 6

Implements `docs/reviews/design-review-phased-plan-2026-10-04.md`. Behavior
changes that callers can observe are listed first.

### Behavior changes

- **Automatic dispatch only picks backends that are built.** A residual-only
  least-squares problem now runs native Levenberg-Marquardt with finite
  differences in a default build (it used to resolve to POUNDERS and return
  `backend_unavailable`). With PETSc built it still uses POUNDERS. A Ceres-native
  AD provider now routes to Ceres automatically when Ceres is built. Bounded
  problems pick a bound-capable built backend instead of failing on the native one.
- **`function_tolerance` compares `F = 0.5 * ||r||^2`** in Levenberg-Marquardt,
  Gauss-Newton and least-squares L-BFGS. It was `||r||` (LM, GN) and `||r||^2`
  (L-BFGS). With the default tolerance a run now stops at `||r|| ~ 2e-8` instead of
  at an essentially exact zero. Scale old tolerances by `0.5 * tol^2` to keep the
  old stopping point.
- **New status `solver_status::stalled`.** A native LM run that is rejected at its
  damping ceiling, a Gauss-Newton line search that fails, and an L-BFGS line search
  that gives up used to be reported as `max_iterations` (or threw). They now report
  `stalled`; `has_usable_iterate()` is true because the last accepted point is valid.
- **Ipopt and TAO outcomes are distinguished.** Hitting the iteration budget is
  `max_iterations` (it was `numerical_failure` for Ipopt). Infeasibility,
  line-search failure and user stops map to their own statuses, and the raw library
  code is kept in `backend_status`. Ceres keeps its raw termination type too.
- **Counters are `std::optional`.** `residual_evaluations`, `jacobian_evaluations`,
  `gradient_evaluations`, `accepted_steps`, `rejected_steps` (and the new
  `objective_evaluations`) are empty when a backend does not report them; zero now
  means measured zero. `gradient_norm` and `step_norm` are filled by the native
  kernels and Ceres.
- **No exceptions escape `api::solve` from the native kernels or user callbacks.**
  A throwing callback becomes `numerical_failure` with its message; a non-finite
  trial point is a rejected step; a non-finite initial point fails before iterating.
- **Every backend resolves derivative policy the same way.** Ipopt, TAO and native
  L-BFGS used to ignore `solve_options::derivatives` or choose their own order; they
  now honor it and report `effective_derivative_source`. With no derivative source,
  least squares reports `finite_difference` (it used to report `automatic`).
  Objective problems still require an explicit gradient unless
  `derivative_mode::finite_difference` is requested.
- **Finite-difference step is relative:** `h_j = step * max(1, |x_j|)`, one shared
  implementation (bounds-aware) used by the native kernels, the providers and the
  dispatcher.
- **LM uses LDLT on the normal equations** instead of `PartialPivLU`, and a
  breakdown of the factorization is a rejected step, not a failure.

### API additions

- `api::find_root(method, f, ...)` (`solvers/api/roots.h`): the seven root methods
  behind one entry point returning `root_result` (status, root, residual,
  iterations, evaluations, message). An unusable bracket is `invalid_problem`
  (checked before searching), a non-finite or throwing function is
  `numerical_failure`, and running out of budget keeps the best estimate as
  `max_iterations`. The `bool` forms in `root_finding_algorithms` are unchanged
  wrappers over the same iterations.
- `api::real_roots`, `real_roots_quadratic/cubic/quartic`: every real root,
  ascending and distinct, with a status. Selection is separate and explicit:
  `smallest_positive_root`, `largest_root`, `roots_in_interval`,
  `smallest_root_in_interval`, and the opt-in `at_least` for the old clamp.

- `solve_options::lm` (`api::lm_options`): variant, bold acceptance, geodesic
  acceleration, damping factors/floors/ceilings, and `linear_solver`
  (`normal_ldlt`, `augmented_qr`).
- Problem setters: `least_squares_problem::set_jacobian`, `set_curve_derivatives`,
  `derivative_provider()`; `optimization_problem::set_gradient`,
  `derivative_provider()`. A problem now has one derivative slot (a provider).
  `JacobianProvider` gained `jacobian_only`, `residual_passes_per_jacobian` and
  `curve_derivatives`.
- `AutoDiffJacobianProvider` builds its Ceres cost function once per provider
  instead of once per call (serialized by a mutex).

### Fixes found along the way

- `root_finding_algorithms::dekker` ignored `function_offset`; it now solves
  `f(x) = offset` like the other methods.
- The `fourth_degree_polynomial_solver` comment promised the minimum positive root
  while the code returns the largest real root (clamped to the threshold). The
  behavior is unchanged and the comment now says what it does.

### Deprecated or removed

- Deprecated for one release (still read, compile with a warning):
  `least_squares_problem::jacobian`, `jacobian_provider`, `provider_factory`,
  `rnc_derivatives`, `rnc_derivative_source`; `optimization_problem::gradient`,
  `gradient_provider`. `model_provider_factory` is now private. Setters no longer
  fill the deprecated fields, so code that *read* `problem.provider_factory` must use
  `problem.derivative_provider()->ceres_factory()`.
- Removed `algorithm::bfgs`: it silently ran L-BFGS. Use `algorithm::lbfgs`.
- Removed the duplicate `solverslib::auto_diff()` definition (re-exported from
  `solverslib::api`).
- The `solver_options_*` builders and the native kernel classes are now documented
  as an internal layer behind `api::solve`; direct use is deprecated for new code.

### Repository

- `default.profraw` is no longer tracked (`*.profraw` is ignored); the SVI plots
  moved to `docs/images/`.
