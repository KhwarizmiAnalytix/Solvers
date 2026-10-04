# Solvers

High-performance native numerical solvers for C++, with optional integrations for automatic differentiation, bound-constrained optimization, and large-scale workloads.

- **Native core** — root finding, least squares and L-BFGS with no external dependencies beyond Eigen.
- **Advanced backends** — optional Ipopt, PETSc/TAO, POUNDERS, and Ceres integrations where they add capabilities the native layer cannot cover.
- **Quant & scientific focus** — designed for calibration, bootstrapping, and pricing workloads.

---

## What It Solves

| Problem | Solver | Engine |
|---|---|---|
| Scalar root finding | Brent, Newton, Ridders, Bisection, Secant, Dekker, False Position | Native |
| Polynomial roots | Companion-matrix solver | Native |
| Nonlinear least squares | Levenberg-Marquardt, Gauss-Newton | Native |
| Unconstrained optimization | L-BFGS | Native |
| Bound-constrained scalar optimization | Interior-point method | Ipopt |
| Large-scale optimization | Newton-Krylov, trust-region | PETSc/TAO |
| Derivative-free least squares | POUNDERS | PETSc/TAO |

---

## Architecture

```text
                        Problem
                           │
              ┌────────────┴─────────────┐
              │                          │
       Least Squares               General Objective
              │                          │
       Derivative source?          Bounds / constraints?
         /          \               /         \
       yes           no           yes          no
        │             │            │            │
   Large scale?   PETSc built?    Ipopt     Large scale?
     /      \      /     \                    /      \
   yes      no   yes      no                yes       no
    │        │    │        │                 │         │
   TAO   Native  POUNDERS  Native LM       TAO    Native L-BFGS
         LM/GN             (finite diff.)
```

Selection is table-driven: each (backend, algorithm) pair is one row with the capabilities it supports, and automatic mode only picks rows whose backend is built into this binary. A request nothing can serve is reported as `unsupported_capability` naming the missing capability.

Scalar root-finding problems use the native root-solving layer directly.

The common API rejects capabilities a selected backend cannot enforce. Ipopt is
currently used for box bounds; general nonlinear constraints are reported as
`unsupported_capability` until constraint callback contracts are implemented.

---

## Quick Start

### Root Finding

```cpp
#include <solvers/root_finding_algorithms.h>

double root = 0.0;

auto options = solverslib::root_finding_options_builder()
    .with_tolerance_function(1e-12)
    .with_tolerance_parameter(1e-12)
    .with_max_iterations(100)
    .build();

auto f = [](double x) { return x * x - 2.0; };

bool converged = solverslib::root_finding_algorithms::brent(
    f, 0.0, 2.0, root, options);
```

For a status, counts and the best estimate instead of a `bool`, use the structured form:

```cpp
#include <solvers/api/roots.h>

using namespace solverslib::api;

root_options opts;
opts.max_iterations = 200;
const root_result result = find_root(root_method::brent, scalar_function(f), 0.0, 2.0, opts);
if (result.converged()) {
    // result.root, result.residual, result.iterations, result.evaluations
}

// Every real root of x^3 - 6x^2 + 11x - 6, then an explicit selection policy.
const auto roots = real_roots_cubic(-6.0, 11.0, -6.0);          // {1, 2, 3}
const auto first = smallest_positive_root(roots.roots);          // 1
```

### Least-Squares Calibration

Everything goes through `api::solve`. Describe the problem, call `solve`, and read one structured result.

```cpp
#include "solvers/api/solve.h"

using namespace solverslib;
using namespace solverslib::api;

least_squares_problem problem;
problem.num_parameters = num_parameters;
problem.num_residuals  = num_residuals;
problem.residuals = [](const vector_type& params, vector_type& r) {
    // r_i = model(params, market_i) - market_i
};
// Optional: without it the solver differentiates numerically.
problem.set_jacobian([](const vector_type& params, matrix_type& J) {
    // fill J with dr_i / dparam_j
});

solve_options options;
options.max_iterations = 100;

const solver_result result = solve(problem, initial_guess, options);
if (result.converged()) {
    // result.parameters holds the calibrated values
}
```

A problem carries one derivative slot (a provider): `set_jacobian(...)` for a callback, `derivatives(auto_diff())` for Ceres AD, `set_jacobian_provider(finite_difference(...))` for a custom stencil. Algorithm-specific tuning lives next to the common options, for example `options.lm` for the native Levenberg–Marquardt damping, geodesic and variant controls.

`solver_result::status` is one closed vocabulary across backends: `converged`, `max_iterations` (budget used up, iterate usable), `stalled` (no acceptable step; the last accepted point is valid), and the failure statuses. Work counters (`residual_evaluations`, `jacobian_evaluations`, `accepted_steps`, ...) are `std::optional`: an empty value means the backend does not report it, an engaged zero means it measured zero.

> The `solver_options_*` builders and the native kernel classes (`levenberg_marquardt_solver`, `gauss_newton_solver`, `lbfgs_solver`) are the internal layer behind `api::solve`. Calling them directly still works, but it is deprecated for new code: they will stop being public in a later release, and new options are only added to `api::solve_options`.

### Automatic Differentiation with Ceres

For problems where hand-coded Jacobians are expensive or error-prone, define a templated residual functor and attach it to a least-squares problem. Ceres uses Jet-based forward automatic differentiation to compute the Jacobian.

```cpp
#include <iostream>
#include "solvers/api/solve.h"
#include "solvers/integrations/autodiff_provider.h"

struct MyResiduals {
    template <typename T>
    bool operator()(const T* parameters, T* residuals) const {
        residuals[0] = parameters[0] * parameters[0] - T(4.0);
        return true;
    }
};

int main() {
    using namespace solverslib;
    using namespace solverslib::api;

    auto problem = least_squares(MyResiduals{}, 1, 1);
    problem.derivatives(api::auto_diff());

    solve_options options;
    options.backend = backend::ceres;
    options.derivatives = derivative_mode::automatic_differentiation;

    vector_type initial_guess(1);
    initial_guess[0] = 3.0;

    const solver_result result = solve(problem, initial_guess, options);
    if (!result.converged()) {
        std::cerr << result.message << '\n';
        return 1;
    }
    std::cout << "solution: " << result.parameters[0] << '\n';
}
```

The convenience factory stores the model and dimensions; `derivatives(auto_diff())` creates the Ceres provider. The Ceres backend must be enabled at configure time.

**Derivative policy** controls how Jacobians are computed:
- `automatic` (default): uses the attached provider's own source (supplied, AD or finite difference); with no provider, numerical differentiation
- `supplied`: requires a callback or analytic provider; `invalid_problem` if missing
- `automatic_differentiation`: requires an AD provider (Ceres); `unsupported_capability` if unavailable
- `finite_difference`: forces numerical differentiation even if other sources exist

Every backend resolves the policy the same way, and `solver_result::effective_derivative_source` reports what was actually used.

**Enable Ceres AD** with:
```bash
cmake -S . -B build -DSOLVERS_ENABLE_CERES=ON
```

This requires Ceres as a dependency; see [ceres-solver.org](http://ceres-solver.org).

---

## Why Solvers

- **Native first** — root finding, LM, Gauss-Newton and L-BFGS are implemented directly. No external optimizer needed for common calibration problems.
- **No redundant backends** — third-party libraries are added only when they introduce a genuinely new capability (constraints, large-scale, derivative-free).
- **Lightweight core** — a native-only build requires only C++17, CMake, and Eigen.
- **Common results** — the unified API reports convergence status, iterations, objective values, and diagnostics available from the selected backend.

---

## Quant Finance Use Cases

| Use Case | Recommended Solver |
|---|---|
| Yield-curve / hazard-rate bootstrap | Brent / Newton |
| Implied volatility | Brent / Newton |
| Vol-surface calibration | Native LM / Gauss-Newton |
| Interest-rate model calibration | Native LM / L-BFGS |
| Box-constrained parameter calibration | Ipopt |
| Large-scale PDE calibration | PETSc/TAO |
| Expensive model without derivatives | POUNDERS |

---

## Performance

Benchmark programs are in `Testing/Cxx/`:

- `BenchmarkRootFinders.cpp`
- `BenchmarkRootFindersVsLM.cpp`
- `BenchmarkSolvers.cpp`
- `BenchmarkSviCalibration.cpp` — calibrates the EURO STOXX 50 raw-SVI example
  from [Ferhati (2020)](https://doi.org/10.2139/ssrn.3543766) with native,
  Ceres, Ipopt, PETSc/TAO, and POUNDERS

Each benchmark reports wall-clock time, iterations, function evaluations, and final residual. Reproduce them locally with:

```bash
cmake --build build --target BenchmarkRootFinders
./build/Testing/Cxx/BenchmarkRootFinders

cmake --build build --target SviCalibrationBenchmark
./build/Testing/Cxx/SviCalibrationBenchmark
```

See the dedicated SVI calibration example in [svi_calib.md](svi_calib.md).

---

## Installation

### Native-only build (no external dependencies beyond Eigen)

```bash
git clone https://github.com/KhwarizmiAnalytix/Solvers.git
cd Solvers
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

### Optional: advanced backends

```bash
cmake -S . -B build \
    -DSOLVERS_ENABLE_CERES=ON \
    -DSOLVERS_ENABLE_IPOPT=ON \
    -DSOLVERS_ENABLE_PETSC=ON
```

- **Ipopt** requires BLAS/LAPACK and a sparse linear solver (e.g. MUMPS). The current adapter supports box bounds; general nonlinear constraints are reported as unsupported. See [coin-or/Ipopt](https://github.com/coin-or/Ipopt).
- **PETSc/TAO** includes TAO and POUNDERS. On macOS: `brew install petsc`. See [petsc/petsc](https://github.com/petsc/petsc).
- **Ceres** enables the automatic-differentiation integration shown above. See [ceres-solver.org](http://ceres-solver.org).

Advanced backends are not required for the native core.

---

## Project Status

### Available now

- Scalar root finding (Brent, Newton, Ridders, Bisection, Secant, Dekker, False Position)
- Polynomial solver
- Levenberg-Marquardt
- Gauss-Newton
- L-BFGS
- Unified problem/solve API with structured results
- Derivative providers for supplied, automatic-differentiation, and finite-difference paths
- Capability validation for bounds, nonlinear constraints, backend/algorithm selection, and invalid inputs
- Ceres automatic differentiation for templated least-squares residuals when `SOLVERS_ENABLE_CERES=ON`

### Optional backends

- Ceres, Ipopt, PETSc/TAO, and POUNDERS adapters are implemented but optional; they require their system dependencies and return `backend_unavailable` when not compiled in.

### In development

- Parameter and residual scaling
- Reproducible benchmark suite with CI regression tracking

The common API explicitly reports unsupported capabilities. Positive
`max_function_evaluations` budgets are rejected until every backend can account
for evaluations from line searches and finite-difference derivatives
consistently.

---

## Documentation

- `docs/` — architecture and backend design notes.

---

## License

See `LICENSE`.

Native LM mathematics, literature references, advanced options, and review findings:
[Levenberg–Marquardt](docs/levenberg-marquardt.md).

[Riemann-normal-coordinate LM (RNC-LM)](docs/rnc-lm.md) implements the
2026 higher-order curve method, with Taylor automatic differentiation and
curve orders 1–4 through the unified solve API.
