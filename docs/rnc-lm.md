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

## Damping convention and paper alignment

The default is now **absolute damping**, `D = I`, with `lambda_0 = 1e-3`.
The previous default was `D_ii = max((JᵀJ)_ii, diagonal_scaling_floor)` and
`lambda_0 = 1e-4`. These choices give lambda different units; changing the
number alone does not make them equivalent.

This distinction explains the large iteration gap previously described in
[demo.md](demo.md). At the generalized Rosenbrock starting point, the diagonal
of `JᵀJ` is approximately `1e12`. The old default therefore added approximately
`1e8` to each diagonal entry and suppressed the initial velocity to about
`5e-9` per parameter. With `D = I` and `lambda_0 = 1e-3`, the initial velocity
is approximately `(-0.998, -0.998)` and the higher-order curve can move along
the valley immediately.

The paper defines `G = JᵀJ + lambda D` without specifying a numerical formula
for `D` in the available main text. Figure 1 labels the RNC trajectories with
`lambda = 1e-3`. Absolute damping is a reconstruction supported by that figure
and the exact full-step accepted-update counts below; it is not a claim that
we obtained the authors' unpublished configuration or code.

To retain the previous scaling explicitly:

```cpp
auto options = solverslib::solver_options_rnc_lm_builder()
    .with_damping_scaling(solverslib::rnc_lm_damping_scaling_enum::MARQUARDT)
    .with_initial_damping(1e-4)
    .build();
```

Both scaling modes reuse one Cholesky factorization for the velocity and all
higher coefficients. Neither the moving-tangent recursion nor the trust-ratio,
interpolation, or accepted-ratio damping updates needed changing.

## Usage

The header-only Taylor adapter supplies ordinary curve derivatives and an
ordinary Jacobian through the current callback interface. It has no Ceres
dependency. This complete example uses the quadratic valley from Eq. (6),
`A = 1e6`, initial point `(1, 1/2)`, and the paper's cost threshold `C < 1e-4`:

```cpp
#include <cmath>
#include <iostream>

#include "api/dispatch.h"
#include "solver_options/solver_options_rnc_lm.h"
#include "solvers/integrations/rnc_autodiff.h"

struct Valley
{
    int power;

    template <class T> void operator()(const T* x, T* r) const
    {
        using std::pow;
        r[0] = x[0];
        r[1] = T(1e6) * (x[1] - pow(x[0], power) / T(power));
    }
};

int main()
{
    using namespace solverslib;
    auto problem = rnc_least_squares(Valley{2}, 2, 2);
    auto options = solver_options_rnc_lm_builder()
        .with_order(4)
        .with_max_curve_trials(1)  // Table 1a: full steps, t_min = 1.
        .with_max_iterations(30000)
        .with_function_tolerance(std::sqrt(2e-4))
        .with_gradient_tolerance(0.0)
        .with_parameter_tolerance(0.0)
        .build();

    vector_type start(2);
    start << 1.0, 0.5;
    const auto result = api::solve(problem, start, *options);
    std::cout << "cost=" << result.objective
              << ", accepted=" << result.accepted_steps.value_or(0)
              << ", curves=" << result.iterations << '\n';
    return result.converged() ? 0 : 1;
}
```

For the cubic valley use `Valley{3}` and start at `(1, 1/3)`.
**The exponent is not the number of parameters:** both problems have two.
The default four curve trials allow three reductions; they correspond to the
paper's `t_min = 0.027` column, not its full-step column.

Models may return `bool` or `void`. A false return throws an evaluation error.
Use a scalar-templated model and unqualified math calls with ADL, such as
`using std::exp; exp(x[0])`. Supported operations include arithmetic, integer
and constant real powers, `exp`, `log`, `sqrt`, `sin`, `cos`, `tan`, and `abs`.
Higher derivatives require smooth models and valid differentiable domains.

### Supplied derivative callbacks

For an existing model, use `make_rnc_derivatives(model, num_parameters,
num_residuals)` or assign a callback directly to `problem.curve_derivatives`:

```cpp
void(const solverslib::vector_type& base,
     const std::vector<solverslib::vector_type>& coefficients,
     int order,
     solverslib::api::rnc_curve_derivatives& out);
```

`coefficients[q-1]` contains `c_q`, the ordinary curve derivative. The input
curve is `base + sum(c_q*t^q/q!)`. Fill `out.residual[0..order]` and
`out.jacobian[0..max(0, order-2)]` with **ordinary derivatives**, not Taylor
coefficients divided by factorials. At order 1 an empty coefficient list
requests the residual and Jacobian at the base point. The residual and
derivative callbacks must describe the same function.

At each higher order, differentiate along the entire supplied truncated curve,
including its already computed coefficients. A quadratic residual does not
imply `R_3 = R_4 = 0` along a curved parameter path. For
`r = [10(y-x²), 1-x]` and coefficients `v, c₂, c₃`, for example,
`R_4[0] = -80*v[0]*c₃[0] - 60*c₂[0]²`. The Taylor adapter accounts for these
terms and factorial conversions automatically. Existing Rosenbrock and SVI
callbacks now use it; SVI's fourth-order residual and Jacobian derivatives
are no longer replaced by zero.

The adapter uses nested forward Taylor differentiation, holds the curve
coefficients fixed when differentiating the base point, and materializes
`J_k` rather than computing the paper's reverse-mode VJP. It uses one model
sweep per parameter per derivative request. The result's derivative-source
label remains `supplied`, as required by the current callback API.
There is no finite-difference fallback for missing curve derivatives.

## Options and diagnostics

Configure these through `solver_options_rnc_lm_builder().with_...()`:

| Option | Default | Meaning |
| --- | --- | --- |
| `order` | 3 | 1–4; order 1 uses a straight LM curve |
| `max_curve_trials` | 4 | Includes the full-length trial; 1 disables contraction |
| `acceptance_threshold` | 1e-4 | Strictly between 0 and 1 |
| `contraction_min` | 0.3 | `0 < min < max < 1` |
| `contraction_max` | 0.5 | Upper contraction factor |
| `damping_scaling` | `IDENTITY` | Absolute damping; `MARQUARDT` selects the previous diagonal scaling |
| `initial_damping` | 1e-3 | Must lie within the damping limits; units depend on scaling |
| `damping_floor` | 1e-15 | Positive and below ceiling |
| `damping_ceiling` | 1e12 | Exhaustion reports numerical failure |
| `diagonal_scaling_floor` | 1e-12 | Positive floor used only by `MARQUARDT` scaling |

Common iteration limits, tolerances, and verbosity use the same builder.
Function tolerance compares `||r||`, gradient tolerance compares `||Jᵀr||`,
and parameter tolerance compares the accepted displacement with the new
parameter norm plus an absolute tolerance-squared term near zero. Disable
gradient and parameter tolerances for a comparison using only the paper's
cost threshold; set function tolerance to `sqrt(2 * cost_threshold)`.

`iterations` and `max_iterations` count attempted curves, including damping
retries. `accepted_steps` counts parameter updates and `rejected_steps` counts
rejected inner trial points. Keep all three quantities when comparing runs.
The paper's Table 1a numbers are reproduced by `accepted_steps`; equating
those numbers with our attempted-curve count overstates the remaining gap.
This counting correspondence is inferred from the numerical reproduction.

`residual_evaluations` counts residual-only callback calls, while
`jacobian_evaluations` counts curve-derivative callback calls, including calls
that also compute residuals and higher derivatives. These are callback counts,
not internal per-parameter Taylor sweeps. Returned norms refer to the last
accepted point. Callback exceptions propagate.

## Comparison with Table 1a

A manual numerical check on **13 October 2026**, using macOS/arm64, Clang,
Release, the Taylor adapter, `D = I`, `lambda_0 = 1e-3`, and
`C < 1e-4`, gave these full-step results. No test or benchmark cases were added.

| Valley exponent | Order | Paper | Accepted updates | Attempted curves |
| --- | ---: | ---: | ---: | ---: |
| 2 | 1 (LM) | 8622 | 8622 | 8679 |
| 2 | 2 | 152 | 152 | 218 |
| 2 | 3 | 33 | 33 | 76 |
| 2 | 4 | 2 | 2 | 2 |
| 3 | 1 (LM) | 8069 | 8069 | 8176 |
| 3 | 2 | 183 | 183 | 273 |
| 3 | 3 | 33 | 33 | 65 |
| 3 | 4 | 21 | 21 | 51 |

All eight accepted-update counts match the full-step column exactly.
With the default four trials (`t_min = 0.027`), the comparison is:

| Valley exponent | Order | Paper | Accepted updates | Attempted curves |
| --- | ---: | ---: | ---: | ---: |
| 2 | 2 | 11 | 9 | 9 |
| 2 | 3 | 4 | 5 | 5 |
| 2 | 4 | 2 | 2 | 2 |
| 3 | 2 | 393 | 393 | 416 |
| 3 | 3 | 7 | 6 | 6 |
| 3 | 4 | 6 | 6 | 6 |

These deeper-search results are similar, with small differences remaining in
orders 2 and 3. Exact agreement for every search depth is not established:
the authors' implementation and supplementary numerical settings were not
available for comparison. The available arXiv source contains the main paper
and figures, without the referenced supplementary methods or optimizer code.
The extreme conditioning also makes trajectories sensitive to floating-point
arithmetic; no settings were tuned separately for individual orders.

The historical `9689/456/33/23` and `6985/441/31/27` rows used diagonal damping
and four trials. They should not be compared with the paper's full-step column.
In particular, the fourth-order default-search discrepancy is now
`23 -> 2` for the quadratic valley and `27 -> 6` for the cubic valley.

## Reference code and remaining scope

The paper and primary numerical reference are
[Liu & Zhang, arXiv:2607.07623v2](https://arxiv.org/html/2607.07623v2).
Public GitHub repository searches for `RNC-LM`, `2607.07623`, and
`Riemann Levenberg`, plus searches for the author's listed email handle, did
not locate an author-published RNC-LM implementation as of 13 October 2026.
This is a search result, not proof that no repository exists.

[`jlumpe/geodesiclm`](https://github.com/jlumpe/geodesiclm) is a GitHub fork of
Transtrum's geodesic LM implementation. Its
[`geodesiclm.f90`](https://github.com/jlumpe/geodesiclm/blob/master/geodesicLM/geodesiclm.f90)
can serve as a reference for the LM/geodesic-acceleration baseline. It does
not implement Liu & Zhang's higher-order moving-tangent recursion or RNC curve
search. No external optimizer code was copied into this repository.

This remains a dense CPU implementation supporting orders 1–4, without bounds,
general scalar objectives, or a common evaluation budget. Normal equations
retain their conditioning limitations; failed Cholesky factorizations and
nonfinite coefficient systems increase damping. Curve acceptance is monotone;
there is no bold acceptance or acceleration-ratio safeguard. The PINN,
MGH10, fifth-order, and large potential-energy results have not been reproduced,
and no GPU or wall-clock speedup claim is made.
