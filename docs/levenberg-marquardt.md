# Native Levenberg–Marquardt: mathematics and options

Reviewed 4 October 2026. Scope: the native solver, option builder, dense solve
helper, dispatch integration, and stopping/result behavior.

This document describes the existing LM/LM-GA implementation and its custom
combination of damping and acceptance policies. Passing its regression tests
is not evidence of reproducing a complete published algorithm. The separately
implemented [RNC-LM method](rnc-lm.md) targets Liu and Zhang’s September 2026
revision and is selected with `algorithm::riemann_normal_coordinate_lm`.

## Mathematical references

Transtrum and Sethna, [*Improvements to the Levenberg-Marquardt algorithm for
nonlinear least-squares minimization*](https://arxiv.org/abs/1201.5885) (2012),
Sections 2–4, Eqs. (12), (14), (15), (19), (20), (22), defines the two extensions:
**geodesic acceleration** and **bold acceptance**. This is an arXiv preprint;
the previous `J. Comput. Phys. (2012)` attribution was unsupported.

[Geodesic acceleration and the small-curvature approximation for nonlinear
least squares](https://arxiv.org/abs/1207.4999), Transtrum and Sethna (2012, arXiv preprint), supplies the
subsequent theoretical treatment. Its convergence results do not establish
convergence of this implementation with bold acceptance.

[Nielsen, *Damping Parameter in Marquardt's Method*, IMM-REP-1999-05](https://www2.imm.dtu.dk/documents/ftp/tr99/tr05_99.pdf)
is the damping reference. [Moré, *The Levenberg-Marquardt Algorithm:
Implementation and Theory* (1978)](https://doi.org/10.1007/BFb0067700)
is the reference for retaining maximum diagonal scaling across iterations.

Newer related work includes S. J. Brooks, [*Higher-Order Corrections to
Optimisers based on Newton’s Method*](https://arxiv.org/abs/2307.03820v2)
(2023; revised May 2024), which develops third- and fourth-order corrections
from geodesic acceleration. Those higher-order methods are not implemented
in this LM strategy and do not replace the original bold-acceptance definition.
Liu and Zhang’s [RNC-LM (2026, v2)](https://arxiv.org/abs/2607.07623v2) adds
moving-tangent corrections and a curve-length search; see the separate
implementation and mathematical mapping in [rnc-lm.md](rnc-lm.md).

## Equations implemented

Let residuals be `r(theta)`, Jacobian `J`, `g = J^T r`, and
`A = J^T J + lambda S`. The minimized objective is

\[
F(\theta)=\tfrac12\|r(\theta)\|^2,\qquad Av=-g.
\]

Geodesic acceleration uses

\[
r_{vv}\simeq\frac{2}{h^2}[r(\theta+hv)-r(\theta)-hJv],\quad
Aa=-J^T r_{vv},\quad s=v+\tfrac12a.
\]

Require `||a|| <= alpha ||v||`; otherwise reject the entire trial and increase
damping. The internal `step` has sign `-s`, `velocity` has sign `-v`, and `tmp`
has sign `-a`. Consequently `step += tmp/2` is the correct update.

Bold acceptance retains the first-order velocity of the last accepted trial:

\[
\beta=\frac{v_{new}^T v_{old}}{\|v_{new}\|\|v_{old}\|},\qquad
(1-\beta)^b\|r_{new}\|^2\leq\min_{accepted}\|r\|^2.
\]

A previous nonzero velocity is required. Nonfinite trials and trials failing
the acceleration bound cannot be accepted by this rule.

For the actual proposed displacement `s`, including both optional corrections,

\[
\rho=\frac{\|r\|^2-\|r(\theta+s)\|^2}
{-2g^Ts-\|Js\|^2}.
\]

Ordinary acceptance requires positive actual and predicted reduction. Bold
acceptance may permit negative reduction. Nielsen's accepted-step update is

\[
\lambda\leftarrow\lambda\max(1/3,1-(2\rho-1)^3).
\]

A nonpositive or nonfinite predicted reduction uses `rho=0` for damping.
There is no absolute value around the cube. An accepted uphill trial usually
increases damping under this rule. Rejection multiplies damping by `nu` and
doubles `nu`; acceptance resets `nu` to its configured initial value. Damping
is bounded to avoid runaway updates.

The directional finite difference suppresses remainders within an eight-epsilon
roundoff bound based on residual and coordinate scales. This implementation
safeguard prevents cancellation noise from rejecting every trial near a solution.
It is not an additional formula prescribed by the cited paper. In particular,
its coordinate-dependent threshold can suppress real curvature after a large
translation of parameters. The RNC-LM implementation obtains derivatives through
Taylor AD or an analytic callback and does not use this heuristic.

## Public options

Reach these options through `api::solve` with `solve_options::lm`
(`api::lm_options`, same names and defaults, with `variant` in place of `type`:
`lm_variant::levenberg_marquardt`, `quadratic_interpolation`, `nielsen`). They
are validated at the solve boundary: an invalid value returns `invalid_problem`
instead of throwing. Budgets and tolerances stay in `solve_options`; the
finite-difference step is owned by the shared finite-difference evaluator, not
by this struct. The `solver_options_lm` builder below is the internal layer
behind that mapping; direct use is deprecated for new code.

A step rejected while the damping already sits at its ceiling ends the run with
`solver_status::stalled` (the last accepted iterate is returned) rather than
spending the remaining iterations, and is no longer reported as `max_iterations`.

| Option | Default | Meaning |
| --- | --- | --- |
| `geodesic_acceleration` | true | Enable second-order step correction and safeguard |
| `geodesic_acceleration_threshold` | 0.75 | Bound on acceleration/velocity norm |
| `geodesic_acceleration_step` | 0.05 | Dimensionless directional finite-difference fraction `h` |
| `bold_acceptance` | false | Enable conservative Eq. (22) |
| `bold_acceptance_exponent` | 2 | Exponent `b` |
| `initial_damping` | 1e-4 | Initial damping |
| `initial_rejection_multiplier` | 2 | Initial Nielsen rejection multiplier |
| `damping_decrease_factor` | 9 | Marquardt accepted-step divisor |
| `damping_increase_factor` | 11 | Marquardt rejected-step multiplier; interpolation failure fallback |
| `damping_floor` | 1e-7 | Minimum damping for Marquardt and quadratic interpolation |
| `nielsen_damping_floor` | 1e-15 | Minimum Nielsen damping |
| `damping_ceiling` | 1e12 | Maximum quadratic-interpolation/Nielsen damping |
| `levenberg_marquardt_damping_ceiling` | 1e7 | Maximum Marquardt damping |
| `diagonal_scaling_floor` | 1e-12 | Lower bound for Marquardt diagonal scaling |
| `roundoff_noise_factor` | 8 | Multiplier for the geodesic finite-difference roundoff guard |
| `finite_difference_step` | 1e-5 | Relative central-difference step, `h_j = step * max(1, abs(x_j))` (legacy kernel constructors only; `api::solve` uses the shared evaluator default) |
| `type` | `NIELSEN` | Coupled scaling and damping policy |

Each option has a `with_...` builder method. The strategy enum contains only
`LEVENBERG_MARQUARDT`, `QUADRATIC_INTERPOLATION`, and `NIELSEN`.
`LEVENBERG_MARQUARDT` uses
`S_ii = max((J^T J)_ii, diagonal_scaling_floor)` and fixed damping factors,
with damping clamped to its configured floor and ceiling.
`QUADRATIC_INTERPOLATION` uses identity scaling and a scalar interpolation
trial, also with configured damping bounds. `NIELSEN` uses
`S_ii = max(1, all encountered (J^T J)_ii)` with the Nielsen update and its
own damping floor plus the shared ceiling. The
maximum-history scaling and floor are implementation choices, rather than
part of the Nielsen damping formula. No separate identity-scaled, fixed-factor
Levenberg policy is currently exposed.

`build()` validates finite positive algorithm parameters, damping factors above
one, nonnegative tolerances/iteration limit, and enum values. Built options are
independent snapshots. The `h=0.05` default is retained; the improvements paper
suggests `h=0.1`, which can be selected explicitly. Likewise the fixed factors
9/11 are retained and should not be called “delayed gratification.”

## Linear solvers

Each damped step solves `(J^T J + diag(D)) delta = J^T r`. Two methods are
available (`lm_options::linear_solver`, or `with_linear_solver(...)` on the
internal builder), both behind `damped_step_solver` in
`include/detail/eigen_support.h`:

| Method | Factorization | Use when |
| --- | --- | --- |
| `normal_ldlt` (default) | `Eigen::LDLT` of `J^T J + diag(D)` (symmetric positive definite, so no pivoting LU) | many more residuals than parameters; well-scaled `J` |
| `augmented_qr` | `Eigen::ColPivHouseholderQR` of `[J; diag(sqrt(D))]` against `[r; 0]` | `J` is ill-conditioned: it never forms `J^T J`, so the condition number is not squared |

The same pair is offered by Ceres (`DENSE_NORMAL_CHOLESKY` / `DENSE_QR`) and the
QR form is what MINPACK `lmder` and Eigen's unsupported `LevenbergMarquardt`
module use. A factorization that breaks down (non-finite input, or a rank
deficient QR) is treated as a rejected step so the damping increases; if it still
fails at the damping ceiling the run ends with `numerical_failure`.

The normal-equations path allocates nothing inside the iteration loop: all
workspace is created once per solve and products use `noalias()` into it. This is
verified by `LmNoAllocCheck`, which compiles the kernel with Eigen's runtime
malloc guard (`EIGEN_RUNTIME_NO_MALLOC`) and fails on any Eigen heap allocation
between the first evaluation and the return. Eigen's QR solve makes one small
temporary per solve call, so `augmented_qr` is not allocation-free.

### Function tolerance

`function_tolerance` is compared with the documented objective
`F = 0.5 * ||r||^2` (it used to be compared with `||r||` in LM and Gauss-Newton and
with `||r||^2` in least-squares L-BFGS). The default `epsilon` therefore now stops a
run once `||r|| < sqrt(2 * epsilon) ~= 2.1e-8`, rather than only at an essentially
exact zero residual. Callers who relied on the old reading should scale their
tolerance by `0.5 * tolerance_old^2`.

### Measured behavior

Iteration counts and final costs are identical for both methods on the library's
standard problems (`LmLinearSolversBenchmark`; analytic Jacobians, tolerances
`1e-24 / 1e-14 / 1e-14`):

| Problem | Linear solver | Status | Iterations | Final cost F |
| --- | --- | --- | --- | --- |
| LinearScalar | normal_ldlt / augmented_qr | converged | 3 / 3 | 2.750e-27 / 2.750e-27 |
| Rosenbrock2D | normal_ldlt / augmented_qr | converged | 32 / 32 | 1.312e-26 / 1.312e-26 |
| PowellSingular | normal_ldlt / augmented_qr | converged | 18 / 18 | 2.313e-20 / 2.313e-20 |
| ExponentialFit | normal_ldlt / augmented_qr | converged | 5 / 5 | 6.642e-32 / 6.642e-32 |

On the raw-SVI calibration fixture (`SviCalibrationBenchmark`, 5 parameters,
analytic Jacobian, gradient tolerance 1e-3) both converge in 8 iterations to the
same 3.37518 variance-bp RMSE as the other solvers; the median solve takes about
6.8 us with `normal_ldlt` and 7.7 us with `augmented_qr` on the development
machine, against 46 us for Ceres LM (16 iterations).

## Review findings and corrections

1. **High: wrong acceleration sign.** Subtracting `tmp/2` moved opposite the
   derived correction. Corrected and checked against a quadratic residual
   with an analytically known directional second derivative.
2. **High: invalid acceleration acceptance.** The previous guard used
   `2||a||` and accepted an uncorrected step on failure. It now uses `||a||`
   and rejects the trial; bold acceptance cannot override it.
3. **High: inconsistent gain ratio.** Actual reduction used residual norms,
   while the denominator represented squared cost and assumed an unmodified
   LM step. Both reductions now use squared quantities and the actual step.
4. **High: incorrect Nielsen update.** Removing the absolute value restores
   increasing damping for poor model agreement and accepted uphill moves.
5. **Medium: wrong bold direction.** History previously stored corrected
   displacements. It now stores LM velocities; rejection preserves history.
6. **Medium: interpolation bookkeeping.** A successful scaled trial now updates
   the step even with bold acceptance disabled. Interpolation uses squared
   residual differences and guards invalid interpolation factors.
7. **Medium: zero diagonal singularity.** Marquardt scaling now floors zero
   diagonals so inactive parameter columns do not create a singular damped
   system solely through their zero diagonal.
8. **Medium: stationary initial point.** The initial gradient is checked before
   proposing a trial. Zero gradients no longer exhaust iterations rejecting
   zero steps. Parameter convergence and logging handle zero parameter norms.
9. **Medium: invalid and mutable options.** Invalid numeric settings are rejected
   at build time; later builder mutations no longer change previously built
   options. The current API uses the renamed options listed above.

## Remaining implementation limits

The solver still forms normal equations and uses `PartialPivLU`, without a
rank-revealing solve, factorization status, or a linear-system residual check.
Finite trial checks help but do not establish accuracy for ill-conditioned
problems. An augmented QR/SVD solve remains a separate numerical improvement.

`function_tolerance` is an absolute residual-norm threshold, not a relative
objective-change tolerance. Gradient tolerance is an absolute norm of `J^T r`;
parameter tolerance tests `||s|| <= tol*(||theta_new||+tol)`. The defaults for
gradient and parameter tolerance are zero. The numerical Jacobian uses an
absolute bump and can lose accuracy across widely varying parameter scales.
Callback dimensions and finite residual/Jacobian inputs are not fully validated
at this native boundary; callback exceptions propagate.

Bold acceptance returns the last accepted iterate, which may have greater cost
than the best encountered iterate. The minimum cost is tracked for acceptance,
but its parameters are not saved. Budget exhaustion and damping exhaustion
both map to `not_converged`, and the historical iteration counter can
underreport the trial that hits the damping cap. Verbose logs show residual
norm as `f(x)`, rather than the minimized half-squared objective.

The higher-level API dispatch builds native options only from common iteration,
tolerance, and verbosity settings. These advanced options are available through
the native LM builder; they are not exposed through the unified API. The native
`log_file` option is not consumed directly by this solver. Changing API exposure,
callback validation, result diagnostics, and linear algebra is outside this
literature-alignment change.

## Validation

Eight focused regression tests cover correction sign and ratio, mandatory
acceleration rejection, aligned uphill acceptance, velocity-based acceptance,
Nielsen damping with poor model agreement, initial stationarity, an inactive
parameter column, and option validation/snapshot independence. The full Ceres-
enabled suite previously passed these regressions. RNC-LM validation is
documented separately in [rnc-lm.md](rnc-lm.md).
