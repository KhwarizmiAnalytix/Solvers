# Riemann-normal-coordinate Levenberg–Marquardt (RNC-LM)

Implemented against Jianing Liu and Dong H. Zhang,
[*Higher-Order Geometric Updates for Levenberg–Marquardt Method via Riemann
Normal Coordinates*, arXiv:2607.07623v2](https://arxiv.org/abs/2607.07623v2),
revised **28 September 2026**. This is a preprint. The implementation supports
orders 1–4 and exposes RNC-LM as a separate algorithm. The existing native LM
and its bold-acceptance options retain their separate behavior.

## Mathematical mapping

With `C = ||r||²/2`, `g = JᵀJ`, and `G = g + lambda D`, the equations are:

| Paper equation | Implemented operation |
| --- | --- |
| (2) | `G v = -Jᵀr`; `c₁ = v` |
| (3) | `theta_K(t) = theta + sum(t^q c_q/q!)` |
| (19), (21) | `G c_n = -sum(binomial(n-2,k) J_kᵀ R_(n-k))`, `k=0..n-2` |
| (32)–(34) | Predict reduction using `m(t)=||r+tJv||²/2`, compare with actual cost on the curve |
| (37), (38) | Quadratic interpolation in `t`, clipped to a contraction interval |
| (40) | Multiply damping by 2 for `rho<0.25`, divide by 3 for `rho>0.75`, otherwise keep it |

Here `R_k` and `J_k` are ordinary derivatives along the truncated curve
`theta_<n(t)`, evaluated at zero. Its already computed coefficients remain fixed
when differentiating the expansion point. One Cholesky factorization serves
all coefficient solves for a given damping value.

For example,

\[
Gc_2=-J^T R_2,\qquad
Gc_3=-J^T R_3-J_1^T R_2,\qquad
Gc_4=-J^T R_4-2J_1^T R_3-J_2^T R_2.
\]

The moving-tangent contributions distinguish RNC-LM from Brooks’s
[residual-linear higher-order recursion](https://arxiv.org/abs/2307.03820v2).

## Implementation choices and limits

The public interface is `api::solve` with
`api::algorithm::riemann_normal_coordinate_lm`. The native entry point is
`solve_rnc_lm`. Both use the same implementation.

This is a dense, CPU implementation. The Taylor adapter uses nested forward AD
and materializes the directional Jacobian derivatives. It evaluates the same
contractions as the paper’s Taylor-mode/VJP construction, but uses one model
sweep per parameter per derivative request. It does **not** reproduce the
paper’s reverse-mode implementation, GPU cost, arbitrary-order support, or
large neural-network performance results. A custom analytic derivative callback
can replace the bundled adapter.

Our concrete scaling is `D_ii=max(g_ii, diagonal_scaling_floor)`. The positive
floor handles inactive columns. Damping limits, initial damping, acceptance
threshold, and contraction bounds are configurable implementation settings;
the defaults below are not claimed to reproduce every paper experiment.
The damped normal matrix still incurs the conditioning limitations of normal
equations. Failed Cholesky factorizations and nonfinite coefficient systems
cause stronger damping. The existing LM finite-difference clipping heuristic
is not used.

Each curve starts at `t=1`. Trial rejection contracts `t` while retaining the
curve coefficients; exhausting the trial budget increases damping and rebuilds
the curve. Acceptance requires a finite positive trust ratio above the
configured threshold. RNC-LM uses monotone cost acceptance and has no bold
uphill-acceptance option or acceleration-ratio rejection test.

## Usage

```cpp
#include "api/dispatch.h"
#include "solvers/integrations/rnc_autodiff.h"

struct Rosenbrock {
    template<class T>
    bool operator()(const T* x, T* r) const {
        r[0] = T(1) - x[0];
        r[1] = T(10) * (x[1] - x[0] * x[0]);
        return true;
    }
};

auto problem = solverslib::rnc_least_squares(Rosenbrock{}, 2, 2);
solverslib::api::solver_options options;
options.algorithm = solverslib::api::algorithm::riemann_normal_coordinate_lm;
options.rnc_lm = solverslib::api::rnc_lm_options{};
options.rnc_lm->order = 4;
options.max_iterations = 300;
options.gradient_tolerance = 1e-10;
options.function_tolerance = 1e-10;
solverslib::vector_type start(2);
start << -1.2, 1.0;
auto result = solverslib::api::solve(problem, start, options);
```

`rnc_least_squares` also supplies an ordinary Jacobian for other algorithms.
It does not change automatic algorithm selection. No Ceres dependency is needed
for its Taylor adapter.

Models may return `bool` (false reports evaluation failure) or `void`. They must
be templated over the scalar type. Supported math includes arithmetic, integer
and constant real powers, `exp`, `log`, `sqrt`, `sin`, `cos`, `tan`, and `abs`.
Use unqualified calls with ADL, e.g. `using std::exp; exp(x[0])`.
Nonsmooth branch boundaries have no well-defined higher derivatives; callbacks
must be smooth in the region being explored. `sqrt` and noninteger powers
require their usual differentiable domains.

For an existing residual callback, call `problem.set_curve_derivatives(function, source)` with a function of signature

```cpp
void(const vector_type& base,
     const std::vector<vector_type>& coefficients,
     int order,
     rnc_curve_derivatives& out);
```

`coefficients[q-1]` contains the unnormalized derivative `c_q`, not `c_q/q!`.
Supply `out.residual[0..order]` and
`out.jacobian[0..max(0,order-2)]` as ordinary derivatives. At order 1, an empty
coefficient list requests the residual and Jacobian at the current point.
The residual callback and derivative callback must represent the same function.
The default source label is `supplied`; the Taylor factory sets
`automatic_differentiation`. The curve derivatives live in the problem's single
derivative slot (a `JacobianProvider` whose `curve_derivatives()` returns them),
so the same provider also serves as the ordinary Jacobian for other algorithms. Explicit incompatible derivative modes are rejected.
There is no numerical finite-difference fallback for missing curve derivatives.

## Options

`options.rnc_lm` contains:

| Field | Default | Valid values / purpose |
| --- | --- | --- |
| `order` | 3 | 1–4; order 1 is a straight LM curve |
| `max_curve_trials` | 4 | Positive; includes the full-length trial |
| `acceptance_threshold` | 1e-4 | Strictly between 0 and 1 |
| `contraction_min` | 0.3 | `0 < min < max < 1` |
| `contraction_max` | 0.5 | Upper contraction factor |
| `initial_damping` | 1e-4 | Within the damping limits |
| `damping_floor` | 1e-15 | Positive and below ceiling |
| `damping_ceiling` | 1e12 | Exhaustion reports numerical failure |
| `diagonal_scaling_floor` | 1e-12 | Positive floor for `D` |

Common iteration limits, tolerances, and verbosity are in `solver_options`.
Function tolerance measures `||r||`; gradient tolerance measures `||Jᵀr||`;
parameter tolerance measures the accepted displacement relative to the new
parameter norm, with an absolute tolerance-squared term near zero.

`iterations` counts attempted curves, including failures. `accepted_steps`
counts accepted curve points; `rejected_steps` counts rejected inner trial
points. `residual_evaluations` counts residual-only callback calls, while
`jacobian_evaluations` counts curve-derivative callback calls (including calls
that also calculate residuals and higher derivatives). These are callback
counts, not the adapter’s internal per-parameter AD sweeps. Returned diagnostics
refer to the last accepted point. Invalid options, unsupported capabilities,
callback failures, damping exhaustion, and budget exhaustion are distinguished.
Bounds, general scalar objectives, and evaluation budgets are unsupported.

## Validation and reproducible comparison

`TestRncModels.cpp` (one test per RNC model) checks analytic second-, third-, and fourth-order updates,
Taylor derivatives of polynomial and elementary functions, the moving-tangent
terms, polynomial curve interpolation, derivative reuse, rank deficiency,
stationarity, malformed inputs, failed callbacks, nonfinite trials, and the
initial-tangent trust-ratio model. The RNC tests, as 15 separate tests at the time, passed with AddressSanitizer
and UndefinedBehaviorSanitizer in a native-only build with Ceres, Ipopt, and
PETSc disabled. The full release CTest suite passed 187 tests, with one
unavailable-backend test skipped and no failures.

`SolversBenchmark rnc-lm` uses the generalized Rosenbrock residuals from Eq. (6), with
stiffness `A=1e6`, initial point `(1,1/n)`, and objective threshold `C<1e-4`.
Build with `SOLVERS_ENABLE_BENCHMARKS=ON`. Local results on 4 October 2026,
using the defaults above, were:

| Valley exponent | Order 1 | Order 2 | Order 3 | Order 4 |
| --- | ---: | ---: | ---: | ---: |
| 2 | 9689 | 456 | 33 | 23 |
| 3 | 6985 | 441 | 31 | 27 |

Values are outer iterations. All runs reached the objective threshold.
Order 1 uses the same curve-search policy as the higher orders. This is an
internal comparison, not reproduction of the paper’s Table 1: defaults and
implementation details differ. No claim is made about PINN or potential-energy
benchmarks, runtime speedups, or global convergence.
