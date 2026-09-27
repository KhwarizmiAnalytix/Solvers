# Solvers

High-performance native numerical solvers for C++, extended with specialized engines for constrained and large-scale optimization.

- **Native core** — root finding, least squares, BFGS/L-BFGS with no external dependencies beyond Eigen.
- **Advanced backends** — Ipopt, PETSc/TAO, POUNDERS only where they add capabilities the native layer cannot cover.
- **Quant & scientific focus** — designed for calibration, bootstrapping, and pricing workloads.

---

## What It Solves

| Problem | Solver | Engine |
|---|---|---|
| Scalar root finding | Brent, Newton, Ridders, Bisection, Secant, Dekker, False Position | Native |
| Polynomial roots | Companion-matrix solver | Native |
| Nonlinear least squares | Levenberg-Marquardt, Gauss-Newton | Native |
| Unconstrained optimization | BFGS, L-BFGS | Native |
| Constrained NLP | Interior-point method | Ipopt |
| Large-scale / matrix-free | Newton-Krylov, trust-region | PETSc/TAO |
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
       Jacobian available?         Constraints?
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

---

## Why Solvers

- **Native first** — root finding, LM, Gauss-Newton, BFGS, and L-BFGS are implemented directly. No external optimizer needed for common calibration problems.
- **No redundant backends** — third-party libraries are added only when they introduce a genuinely new capability (constraints, large-scale, derivative-free).
- **Lightweight core** — a native-only build requires only C++17, CMake, and Eigen.
- **Common results** — all solvers report convergence status, iterations, and residual norms through a consistent interface.

---

## Quant Finance Use Cases

| Use Case | Recommended Solver |
|---|---|
| Yield-curve / hazard-rate bootstrap | Brent / Newton |
| Implied volatility | Brent / Newton |
| Vol-surface calibration | Native LM / Gauss-Newton |
| Interest-rate model calibration | Native LM / L-BFGS |
| Constrained parameter calibration | Ipopt |
| Large-scale PDE calibration | PETSc/TAO |
| Expensive model without derivatives | POUNDERS |

---

## Performance

Benchmark programs are in `Testing/Cxx/`:

- `BenchmarkRootFinders.cpp`
- `BenchmarkRootFindersVsLM.cpp`
- `BenchmarkSolvers.cpp`

Each benchmark reports wall-clock time, iterations, function evaluations, and final residual. Reproduce them locally with:

```bash
cmake --build build --target BenchmarkRootFinders
./build/Testing/Cxx/BenchmarkRootFinders
```

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
    -DSOLVERS_ENABLE_IPOPT=ON \
    -DSOLVERS_ENABLE_PETSC=ON
```

- **Ipopt** requires BLAS/LAPACK and a sparse linear solver (e.g. MUMPS). See [coin-or/Ipopt](https://github.com/coin-or/Ipopt).
- **PETSc/TAO** includes TAO and POUNDERS. On macOS: `brew install petsc`. See [petsc/petsc](https://github.com/petsc/petsc).

Advanced backends are not required for the native core.

---

## Project Status

### Available now

- Scalar root finding (Brent, Newton, Ridders, Bisection, Secant, Dekker, False Position)
- Polynomial solver
- Levenberg-Marquardt
- Gauss-Newton
- BFGS / L-BFGS

### In development

- Unified `LeastSquaresProblem` / `OptimizationProblem` types
- Common `SolverResult` across all backends
- Ipopt adapter
- PETSc/TAO adapter (including POUNDERS)
- Parameter and residual scaling
- Reproducible benchmark suite with CI regression tracking

---

## Documentation

- `docs/` — architecture and backend design notes.

---

## License

See `LICENSE`.
