# Backend automatic differentiation: design review and implementation plan

Date: 2026-09-27. Scope: current source, optional backend integration, and a proposed implementation. This document does not implement the changes or claim runtime validation.

**Recommendation:** use Ceres' public `DynamicAutoDiffCostFunction` for the existing runtime-sized least-squares API. Instantiate it in an optional integration header while the residual functor's concrete type is known, then erase the resulting executable evaluator behind a small backend-neutral interface. Retain supplied Jacobians and explicitly selected numerical differentiation. Keep Ceres optional.

“AD” here means automatic differentiation. Ceres Jets implement forward-mode AD, not reverse-mode/adjoint AAD. If adjoint differentiation is the actual requirement, that needs a separate provider and performance study. Ceres' [automatic derivatives documentation](https://ceres-solver.readthedocs.io/latest/automatic_derivatives.html) explains its Jet model.

## Current design and findings

The separation into `least_squares_problem`, `optimization_problem`, `solver_options`, dispatch, and backend adapters is a useful foundation. The least-squares convention is explicit: `F = 0.5 * ||r||^2`, `J(i,j) = dr_i/dx_j`. Keep this architecture; a new optimization abstraction is unnecessary to deliver AD.

| Priority | Finding and source | Consequence |
|---|---|---|
| P1 | [`problem.h`](../../include/api/problem.h), `least_squares_problem`, stores a functor in `std::any`, but [`dispatch.cpp`](../../src/api/dispatch.cpp), `run_ceres`, passes only the double residual and Jacobian callbacks. | The advertised templated-residual path does not compute derivatives. |
| P1 | [`ceres_solver.cpp`](../../src/solvers/ceres_solver.cpp), `solve`, installs a central-difference lambda with an absolute `1e-8` bump when the Jacobian callback is empty. | Current “internal Jacobian” means library-written finite differences, not Ceres AD or Ceres numeric differentiation. |
| P1 | [`dispatch.cpp`](../../src/api/dispatch.cpp), `inspect` and `select_algorithm`, considers only `jacobian.has_value()` and routes missing Jacobians to POUNDERS. | A usable AD provider would still be classified as derivative-free unless dispatch changes. An engaged optional holding an empty `std::function` is also misclassified. |
| P1 | [`TestSolverBackends.cpp`](../../Testing/Cxx/TestSolverBackends.cpp), `AutomaticDifferentiation` tests, checks storage/retrieval and residual values with `double`. | These tests do not instantiate Jets or verify a Jacobian. `CeresWithInternalJacobian` currently tests finite differences. |
| P2 | The Ceres adapter accepts one parameter block and one residual block for the entire model. | Runtime-sized dense AD is a good initial fit, but sparse/Schur choices cannot recover a dependency graph that the API does not express. |
| P2 | [`CMakeLists.txt`](../../CMakeLists.txt) links Ceres privately. [`SolversConfig.cmake.in`](../../Cmake/SolversConfig.cmake.in) reconstructs a static imported target and hardcodes all optional backends off. | Client-side AD template instantiation requires an explicit dependency target and a working installed-package configuration. |

Related correctness issues must be handled in the touched path:

- `ceres_solver::solve` applies bounds only when **both** vectors are nonempty, despite the public API accepting one-sided bounds.
- Its fallback lambda captures `this` and is retained in the solver object. Copying a used solver can leave the copy referring to the original instance.
- `LambdaCostFunctor::Evaluate` assumes valid output dimensions and returns true unconditionally; failures need an explicit evaluation contract.
- `run_ceres` maps `IsSolutionUsable()` to `converged`, loses the Ceres termination reason, and recomputes final residuals outside its exception handler.
- `max_function_evaluations` is not enforced by this Ceres path. Introducing AD adds multiple internal passes, so the counting convention must be defined.

## Why storing a templated functor is insufficient

The current callback types operate on `Eigen::VectorXd`/`MatrixXd`, hence on `double`. Ceres cannot differentiate arbitrary computation hidden behind that double-only interface. Converting a Jet to a double and reconstructing a Jet loses derivative information.

`std::any` retains an object but cannot invoke an unknown member template from a non-templated `.cpp`. `type_info` does not provide that facility either. `any_cast<F>` only helps if the consumer already knows `F`; registering a finite list of application functors would make the library non-extensible.

The required ordering is:

```text
Application functor F, still known at compile time
  -> instantiate Ceres AD wrapper for F
  -> expose ordinary double-buffer evaluation of residuals and Jacobian
  -> erase that executable interface
  -> dispatch and solve at runtime
```

All computations dependent on parameters must support the scalar type `T`. Use `T` intermediates and constants where necessary, and math overloads compatible with Jets, for example `using std::exp; exp(value)`. Data independent of parameters may remain double. External double-only routines need supplied derivatives or an explicitly numerical subcomponent; AD does not make nondifferentiable expressions smooth.

The vendored Ceres headers identify version 2.3.0. Its [`dynamic_autodiff_cost_function.h`](../../ThirdParty/ceres/include/ceres/dynamic_autodiff_cost_function.h) supports runtime parameter dimensions; [`autodiff_cost_function.h`](../../ThirdParty/ceres/include/ceres/autodiff_cost_function.h) supports fixed parameter-block sizes, with an optionally runtime residual count. Use these public APIs, not `ceres::internal` implementation details. The [official modeling reference](https://ceres-solver.readthedocs.io/latest/nnls_modeling.html#dynamicautodiffcostfunction) documents the dynamic interface.

## Proposed interface and ownership

Add an optional header, for example `solvers/integrations/ceres_autodiff.h`, with a named helper:

```cpp
// Proposed API; not available in the current implementation.
auto problem = api::make_ceres_autodiff_problem(
    2, 2, RosenbrockResiduals{});

api::solver_options options;
options.backend = api::backend::ceres;
options.derivatives = api::derivative_mode::automatic_differentiation;
auto result = api::solve(problem, initial_guess, options);
```

The helper creates the ordinary residual callback and the AD provider from the **same owned model**. Callers should not have to supply two independent residual implementations. When the provider is selected, use it as the authoritative residual source throughout that solve, including final diagnostics. A later replacement of the public residual callback must not silently pair different residuals and derivatives; document that replacing the model requires rebuilding/clearing its provider.

Add a narrow neutral provider contract in a new API/detail header:

- Immutable metadata: dimensions, derivative source, and supported execution backends.
- A factory that creates an evaluator owned by one solve.
- An evaluator operation conceptually shaped as `evaluate(x, residuals, jacobian_row_major_or_null)` with an explicit status.
- Null Jacobian means residual-only evaluation. Buffers have validated sizes. Jacobian layout is `m` rows by `n` columns, row-major.
- Distinguish successful evaluation, recoverable invalid trial, and fatal failure. Store fatal diagnostic/exception information in solve-local state; translate it at the solver boundary.

The problem can hold a `shared_ptr<const provider_factory>`; each solve owns the evaluator returned by that factory. The model can be owned through shared immutable state, but callbacks must be deterministic and reentrant if concurrent use is supported. Const qualification alone does not guarantee this. Work buffers and counters belong to the solve.

The templated Ceres provider owns a `DynamicAutoDiffCostFunction` and adapts the existing single-array functor signature:

```cpp
template<class Functor>
struct OneBlockFunctor {
    Functor functor;

    template<class T>
    bool operator()(T const* const* blocks, T* residuals) const {
        return functor(blocks[0], residuals);
    }
};

// Inside the optional template implementation, after range validation:
using Adapter = OneBlockFunctor<Functor>;
auto cost = std::make_unique<ceres::DynamicAutoDiffCostFunction<Adapter, 4>>(
    new Adapter{std::move(functor)});
cost->AddParameterBlock(static_cast<int>(n));
cost->SetNumResiduals(static_cast<int>(m));
```

The neutral evaluator invokes this object's `Evaluate`, passing a single parameter pointer and, when requested, a single Jacobian pointer. A thin Ceres `CostFunction` adapter forwards solver requests to the evaluator. This introduces a small virtual forwarding boundary while allowing residuals and derivatives to be computed together directly into Ceres' output buffers. Do not call the old residual callback separately before every AD evaluation.

Ceres `Problem` owns the outer adapter under its normal ownership policy; the adapter owns the evaluator; the evaluator owns the inner AD cost function. Never give two owners the same cost-function pointer. Catch application exceptions at the evaluation boundary, including any Ceres worker invocation, rather than relying only on an outer `try` around `Solve`.

Keep the provider interface independent of Ceres types. The optional integration header necessarily exposes a compile-time Ceres dependency to clients using it. Arbitrary client functors cannot all be instantiated inside the prebuilt Solvers library.

Do not extend the `std::any` mechanism. Deprecate its setter/getter in favor of the explicit helper, with a migration example. During a compatibility period, a stored legacy functor alone must not satisfy an explicit AD request; return an actionable unsupported-capability message. Rename internal `CostFunctionLambda_aad` to `jacobian_function` or equivalent, retaining a legacy alias if needed.

## Derivative policy and dispatch

Introduce a least-squares derivative policy, independent of the algorithm:

| Requested policy | Behavior |
|---|---|
| `automatic` | Prefer a callable supplied Jacobian; otherwise use a compatible AD provider; otherwise use the selected backend's documented numeric/derivative-free behavior. |
| `supplied` | Require an engaged **and callable** Jacobian. Missing callback is an error. |
| `automatic_differentiation` | Require an executable compatible AD provider. No silent numerical fallback. |
| `finite_difference` | Use the documented finite-difference provider even if a supplied Jacobian or AD provider exists. |

Derivative-free solving remains an algorithm choice, such as explicit POUNDERS. Reject explicit incompatible combinations, for example POUNDERS with AD-required policy, rather than accepting a request whose derivative requirement is unused.

For Ceres, resolve the policy to one of three concrete implementations:

1. Supplied Jacobian: retain the analytic/callback `CostFunction`, with validation.
2. AD: use the instantiated provider described above.
3. Numerical: replace the local fixed-bump lambda with Ceres `DynamicNumericDiffCostFunction<..., CENTRAL>` for models evaluable around the iterate, exposing a relative-step option. Numerical differentiation is a separate mode, not AD. The vendored [numeric-diff header](../../ThirdParty/ceres/include/ceres/dynamic_numeric_diff_cost_function.h) defines this path.

Ceres numeric differentiation does not receive parameter bounds from `Problem`. Do not promise bound-aware differences: for models whose domain ends at a bound, retain/add an explicit library finite-difference provider with feasible one-sided stencils, or require supplied/AD derivatives until that provider exists. Ceres' optimization bounds alone do not make numerical perturbations safe.

Resolve a single execution plan containing backend, algorithm, derivative source, and availability. Have both public selection helpers and `solve` use the same resolver, so they cannot report mutually inconsistent decisions. Add `has_autodiff_provider`/provider capabilities to traits, rather than conflating them with `has_supplied_jacobian`.

Initial dispatch rules:

- Explicit backend and derivative requests take precedence; validate their combination before invoking callbacks.
- Existing supplied-Jacobian, unconstrained small problems retain native LM by default.
- For an AD-only problem built with the Ceres helper, automatic selection chooses Ceres LM. Size thresholds must not divert it to a backend that cannot consume its provider.
- Preserve residual-only automatic POUNDERS selection for compatibility in this change. An explicit finite-difference policy instead needs a derivative-based algorithm/backend, such as native LM for supported unconstrained problems.
- Bounds require a bound-capable selected backend or an explicit unsupported-capability result. Validate explicit algorithm/backend compatibility; Ceres must not silently label its solve as native GN/BFGS.
- Explicit Ceres with no derivative requirement may use Ceres numerical differentiation. Explicit AD without a provider fails. An unavailable requested backend returns `backend_unavailable`, without numerical fallback.

Report the effective derivative source in `solver_result`, e.g. supplied, Ceres forward AD, Ceres central differences, or none. This makes behavior inspectable and testable.

## Implementation sequence and acceptance criteria

1. **Define the contract and resolver.** Update `problem.h`, `options.h`, `result.h`, and `dispatch.cpp`; add the provider interface. Correct empty optional-function detection. Define derivative precedence and compatibility rules above. Preserve ordinary callback construction. Tests must cover AD capability without a supplied Jacobian, explicit policy failures, backend unavailability, and automatic selection. No callback may run for an invalid request.

2. **Implement the Ceres AD integration.** Add `ceres_autodiff.h` and the one-block adapter; create the AD wrapper before type erasure. Add a provider-consuming path to `ceres_solver` while retaining the legacy callback constructor. Validate positive dimensions, `INT_MAX` conversions, buffer-size products, and provider/problem dimension agreement before calling Ceres. Verify residual-only and residual-plus-Jacobian evaluation directly before testing optimization.

3. **Unify Ceres evaluation paths.** Select supplied, AD, or numeric evaluation once per solve. Remove the retained self-capturing finite-difference closure and obsolete debug differentiation code. Validate callback output dimensions and finite values; propagate functor false returns. Apply lower and upper bounds independently, validate lengths/order, and define treatment of infeasible initial guesses. Start with rejection and a clear message rather than undocumented projection. Use the canonical selected evaluator for final diagnostics.

4. **Return truthful Ceres results.** Add an internal solve report carrying termination type, usability, iterations, and available evaluation statistics; retain the legacy bool wrapper if required. Map Ceres convergence, no-convergence limits, and failures separately. Preserve native termination codes and report actual algorithm/derivative source. Count residual-only calls, Jacobian requests, and underlying model invocations separately: dynamic AD may use several Jet passes for one Jacobian. Enforce a requested evaluation budget against a documented counter with a stored stop reason; otherwise reject nonzero budgets as unsupported during migration. Do not report budget exhaustion as numerical failure or unknown counts as measured zero.

5. **Make the optional dependency consumable.** Add an interface target `Solvers::CeresAutodiff` linking `Solvers::Solvers` and `Ceres::ceres`, enabled only in Ceres builds. Carry Ceres/Eigen/Abseil include and link requirements through targets, not hand-written include paths. Repair package metadata to reflect configured backend flags, static/shared artifacts, and required dependencies; prefer proper exported targets where practical. Test an installed external consumer. Keep the existing Bazel native target working without including the optional header; add a separate Ceres Bazel target only if that backend is explicitly supported there.

6. **Prove correctness and document migration.** Add focused AD tests, an example, README guidance, and backend-off compile checks. Rename `CeresWithInternalJacobian` to describe numerical differentiation. Replace storage-only AD tests with end-to-end capability tests while retaining residual-value checks as useful supporting tests. Mark the legacy `std::any` API as deprecated, with the explicit helper as replacement.

7. **Measure before broadening.** Benchmark analytic, Ceres AD, and numerical Jacobians at equal accuracy on representative dimensions. Record model calls, Jacobian requests, allocations, and wall time. Test a few compile-time strides, initially 4, without claiming one universal winner. Only then add fixed-dimension helpers or a residual-block graph API.

Steps 1–6 form the initial supported feature. Step 7 is a follow-up optimization effort. The broader scalar-objective/Hessian redesign is not a prerequisite.

## Validation matrix

| Area | Required evidence |
|---|---|
| Actual AD execution | Functor distinguishes double and Jet instantiations with counters; a Jacobian request exercises Jets. Solve with no supplied Jacobian and assert effective source is Ceres AD. |
| Mathematical Jacobian | Rosenbrock, Powell, exponential fitting, plus a rectangular model with more than four parameters. Compare every entry against independent analytic derivatives at several points; use directional finite differences as a secondary check. |
| Strided execution/layout | At least five active parameters exercise multiple stride-4 passes. Use a nonsymmetric rectangular Jacobian so transposes and column/row-major mistakes cannot pass accidentally. |
| Optional Jacobians | Both `jacobians == nullptr` and `jacobians[0] == nullptr` produce residuals without unnecessary AD work. |
| Policy | Supplied-plus-AD precedence, forced AD, forced numerical, no provider, empty engaged optional callback, explicit backend/algorithm incompatibility. |
| Optimization | Solve existing models, a noisy fit with nonzero optimum residual, and lower-only/upper-only/active-bound cases. Check mathematical outputs as well as termination status. |
| Failure handling | Functor false, application exception, NaN/Inf, malformed legacy callback output, invalid dimensions/bounds, and dimension metadata changed after provider creation. |
| Lifetime | Construct from a temporary functor; copy/move the problem; destroy the original; run repeated solves. Test concurrency only under the documented reentrant model contract, using sanitizers where supported. |
| Termination/counters | Iteration limit can leave a usable unconverged iterate. Verify budget accounting includes AD passes and final evaluations, or that unsupported budgets are rejected. |
| Build/package | Native-only build without Ceres headers; Ceres build; static/shared consumers; installed consumer that includes the integration header and instantiates a new application-defined functor. |

Raw-pointer functors must obey the declared buffer dimensions. Runtime dimension checks cannot make arbitrary out-of-bounds user writes safe; exercise representative misuse with sanitizers rather than claiming full memory validation.

## Other backends and later scalability

The reviewed native solvers accept Jacobian callbacks and currently estimate derivatives when omitted. The PETSc adapter registers supplied residual Jacobians through `TaoSetJacobianResidualRoutine`; the Ipopt adapter consumes gradient/Hessian callbacks and optionally uses a limited-memory Hessian approximation. Neither adapter currently contains a templated AD path. POUNDERS is derivative-free; Hessian approximation and finite differences should not be described as AD.

The neutral evaluator permits a later opt-in bridge from Ceres-generated Jacobians into native LM/GN or TAO without running the Ceres optimizer. That would still require the Ceres dependency and would initially materialize a dense Jacobian. Keep derivative-provider identity separate from optimizer identity, but defer exposing this bridge until Ceres integration is validated.

Dynamic AD computes derivatives in groups determined by its stride; a full monolithic Jacobian can require roughly `ceil(n / stride)` model passes. This is not a reverse-mode advantage for models with many parameters. A future block API should express parameter-block identities, residual-block dependencies, ownership, and bounds explicitly; then select static `AutoDiffCostFunction` where block dimensions are known. Sparse solvers and Schur methods become useful when that structure exists. A dense Jacobian provider also does not by itself provide matrix-free `Jv`/`J^T v` operations.

The recommended first delivery is therefore a truthful, tested dense Ceres AD path with explicit policy and clean packaging. Block sparsity, reverse-mode AD, and shared cross-backend derivative services can build on it without delaying that delivery.
