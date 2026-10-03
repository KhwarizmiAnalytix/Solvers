# Solvers

High-performance native numerical solvers for C++, with optional integrations for automatic differentiation, bound-constrained optimization, and large-scale workloads.

- **Native core** — root finding, least squares, BFGS/L-BFGS with no external dependencies beyond Eigen.
- **Advanced backends** — optional Ipopt, PETSc/TAO, POUNDERS, and Ceres integrations where they add capabilities the native layer cannot cover.
- **Quant & scientific focus** — designed for calibration, bootstrapping, and pricing workloads.

---

## What It Solves

| Problem | Solver | Engine |
|---|---|---|
| Scalar root finding | Brent, Newton, Ridders, Bisection, Secant, Dekker, False Position | Native |
| Polynomial roots | Companion-matrix solver | Native |
| Nonlinear least squares | Levenberg-Marquardt, Gauss-Newton | Native |
| Unconstrained optimization | BFGS, L-BFGS | Native |
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
       Jacobian available?         Bounds / constraints?
         /          \               /         \
       yes           no           yes          no
        │             │            │            │
   Large scale?    POUNDERS       Ipopt     Large scale?
     /      \                                /      \
   yes      no                             yes       no
    │        │                              │         │
   TAO    Native LM/GN                    TAO    Native BFGS/
                                                  L-BFGS
```

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

### Least-Squares Calibration

```cpp
#include <solvers/levenberg_marquardt_solver.h>
#include "detail/eigen_support.h"

using solverslib::vector_type;

auto residuals = [](vector_type const& params, vector_type& r) {
    // r_i = model(params, market_i) - market_i
};

auto jacobian = [](vector_type const& params, matrix_type& J) {
    // fill J with dr_i / dparam_j
};

solverslib::levenberg_marquardt_solver solver(
    num_parameters, num_residuals, residuals, jacobian);

vector_type params = initial_guess;
auto result = solver.solve(params, lm_options);

if (result.converged()) {
    // params now holds the calibrated values
}
```

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
- `automatic` (default): Prefers a supplied Jacobian, then an attached AD provider, then numerical or derivative-free execution
- `supplied`: Requires Jacobian callback; error if missing
- `automatic_differentiation`: Requires AD provider (Ceres); error if unavailable
- `finite_difference`: Forces numerical differentiation

**Enable Ceres AD** with:
```bash
cmake -S . -B build -DSOLVERS_ENABLE_CERES=ON
```

This requires Ceres as a dependency; see [ceres-solver.org](http://ceres-solver.org).

---

## Why Solvers

- **Native first** — root finding, LM, Gauss-Newton, BFGS, and L-BFGS are implemented directly. No external optimizer needed for common calibration problems.
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
- BFGS / L-BFGS
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
