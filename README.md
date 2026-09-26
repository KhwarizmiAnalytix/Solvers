# Solvers

**High-performance numerical solvers for modern C++.**

`Solvers` provides optimized native implementations for the numerical problems most frequently encountered in quantitative finance, calibration, and scientific computing, while delegating specialized optimization problems to mature third-party engines only when they add genuinely new capabilities.

The design goal is simple:

> **Native where performance matters. Specialized third-party engines where the mathematics requires them. One consistent C++ API.**

---

## Overview

`Solvers` covers four main problem families:

- scalar root finding,
- nonlinear least squares,
- smooth unconstrained optimization,
- advanced constrained and large-scale optimization.

Common numerical problems are handled by optimized native implementations.

More specialized problems are routed to dedicated external engines:

- **Ipopt** for constrained nonlinear programming,
- **PETSc/TAO** for large-scale and matrix-free optimization,
- **POUNDERS** through PETSc/TAO for derivative-free nonlinear least squares.

No third-party library is introduced simply to duplicate an existing native solver.

---

## Problem-Oriented Architecture

`Solvers` is organized around the mathematical structure of the problem rather than around individual solver libraries.

```text
                           Problem
                              │
                 ┌────────────┴─────────────┐
                 │                          │
          Nonlinear Least Squares      General Objective
                 │                          │
          Jacobian available?          Constraints?
            /          \               /         \
          yes           no           yes          no
           │             │            │            │
      Large scale?    POUNDERS       IPOPT      Large scale?
        /      \                                  /      \
      yes      no                               yes       no
       │        │                                │         │
      TAO    Native LM/GN                       TAO   Native BFGS/
                                                       L-BFGS
```

Scalar root-finding problems use the native root-solving layer directly.

---

# Core Capabilities

| Problem | Algorithms / Engine | Implementation |
|---|---|---|
| Scalar root finding | Brent, Newton-Raphson, Ridders, Secant, Bisection, False Position, Dekker | Native |
| Polynomial roots | Polynomial solver | Native |
| Nonlinear least squares | Levenberg-Marquardt, Gauss-Newton | Native |
| Smooth unconstrained optimization | BFGS, L-BFGS | Native |
| Nonlinear constrained optimization | Interior-point nonlinear programming | Ipopt |
| Large-scale optimization | Newton-Krylov, trust-region and matrix-free methods | PETSc/TAO |
| Derivative-free nonlinear least squares | POUNDERS | PETSc/TAO |

---

# Native Core

The native solver layer is designed for low overhead and performance-sensitive applications.

## Root Finding

Available scalar root solvers include:

```text
Bisection
False Position
Ridders
Dekker
Brent
Newton-Raphson
Secant
```

Typical applications include:

- implied-volatility inversion,
- yield solving,
- curve bootstrapping,
- par-rate solving,
- strike inversion,
- scalar model calibration.

Example:

```cpp
#include <solvers/root_finding_algorithms.h>

double root = 0.0;

solvers::root_finding_options options;
options.tolerance_function = 1e-12;
options.tolerance_parameter = 1e-12;
options.max_iterations = 100;

const auto f = [](double x) {
    return x * x - 2.0;
};

const bool converged =
    solvers::brent(f, 0.0, 2.0, root, options);
```

---

## Nonlinear Least Squares

For problems of the form

$$
\min_x \frac{1}{2}\|r(x)\|^2,
$$

the native layer provides:

- Levenberg-Marquardt,
- Gauss-Newton.

These solvers are intended for calibration problems where the residual structure is known and can be exploited directly.

Typical examples include:

$$
\min_\theta
\sum_i
w_i
\left(
V_i^{model}(\theta)-V_i^{market}
\right)^2.
$$

Applications include:

- volatility-surface calibration,
- interest-rate model calibration,
- credit-model calibration,
- curve fitting,
- parameter estimation.

---

## Unconstrained Optimization

For smooth objectives

$$
\min_x f(x),
$$

the native layer provides:

- BFGS,
- L-BFGS.

L-BFGS is particularly suitable for larger parameter vectors where storing or factorizing a dense Hessian approximation would be undesirable.

---

# Advanced Optimization Backends

Third-party libraries are used only when they provide numerical capabilities not already covered efficiently by the native implementation.

## Ipopt

**Purpose:** general constrained nonlinear optimization.

Ipopt is used for problems of the form:

$$
\min_x f(x)
$$

subject to

$$
g_L \leq g(x) \leq g_U
$$

and

$$
x_L \leq x \leq x_U.
$$

Typical use cases include:

- nonlinear equality constraints,
- nonlinear inequality constraints,
- parameter bounds,
- structural calibration constraints,
- smoothness constraints,
- no-arbitrage constraints.

Repository:

```text
https://github.com/coin-or/Ipopt
```

Ipopt is an optional dependency.

---

## PETSc / TAO

**Purpose:** large-scale, sparse, matrix-free and Newton-Krylov optimization.

TAO provides optimization algorithms designed for large numerical problems, including:

- Newton-Krylov methods,
- trust-region Newton methods,
- matrix-free optimization,
- Hessian-vector-product based methods,
- sparse optimization,
- distributed-memory problems.

PETSc/TAO becomes useful when a problem is too large or too structured for a conventional dense local optimizer.

Repository:

```text
https://github.com/petsc/petsc
```

TAO is distributed as part of PETSc and does not require a separate installation.

---

## POUNDERS

**Purpose:** derivative-free nonlinear least squares.

POUNDERS is used when the objective has least-squares structure

$$
F(x)
=
\frac12
\sum_i r_i(x)^2
$$

but reliable Jacobians are unavailable.

This is particularly useful for:

- expensive PDE-based pricing models,
- noisy residual functions,
- legacy models without derivatives,
- nested numerical calculations,
- models where finite differences are prohibitively expensive.

POUNDERS is provided through PETSc/TAO.

---

# Solver Selection

The recommended solver depends on the mathematical structure of the problem.

## Root Finding

Use the native root solvers.

```text
Bracket available
    ├── Brent
    ├── Ridders
    └── Bisection

Derivative available
    └── Newton-Raphson

No derivative / no strict bracket requirement
    └── Secant
```

## Nonlinear Least Squares

```text
Jacobian available
    ├── normal-size problem
    │      └── Native LM / Gauss-Newton
    │
    └── large / matrix-free problem
           └── PETSc/TAO

Jacobian unavailable
    └── POUNDERS
```

## General Optimization

```text
Nonlinear constraints
    └── Ipopt

Unconstrained
    ├── normal-size problem
    │      └── Native BFGS / L-BFGS
    │
    └── large-scale / Hessian-vector formulation
           └── PETSc/TAO
```

---

# Quantitative Finance

`Solvers` is designed with calibration and numerical finance workloads in mind.

| Use case | Preferred solver |
|---|---|
| Yield-curve bootstrap | Brent / Newton |
| Implied volatility | Brent / Newton |
| Hazard-rate bootstrap | Brent |
| Volatility-surface calibration | Native LM / GN |
| Interest-rate model calibration | Native LM / L-BFGS |
| Constrained parameter calibration | Ipopt |
| Large-scale PDE calibration | PETSc/TAO |
| Expensive least-squares model without derivatives | POUNDERS |

A typical calibration problem is:

$$
\theta^\star
=
\arg\min_\theta
\sum_{i=1}^{N}
w_i
\left(
V_i^{model}(\theta)
-
V_i^{market}
\right)^2.
$$

The solver architecture allows this structure to be preserved instead of flattening every calibration problem into a generic scalar objective.

---

# Design Principles

## Native First

Common numerical algorithms remain native.

This keeps standard workflows lightweight and avoids introducing large third-party dependencies for ordinary calibration problems.

```text
Root finding       → Native
LM / GN            → Native
BFGS / L-BFGS      → Native
Polynomial roots   → Native
```

---

## Specialized Backends Only

External libraries are introduced only for capabilities that materially extend the native solver layer.

```text
Constrained nonlinear programming → Ipopt
Large-scale / matrix-free          → PETSc/TAO
Derivative-free least squares      → POUNDERS
```

The project deliberately avoids wrapping multiple libraries that solve the same problem without adding significant value.

---

## Problem-Oriented API

Application code should describe a numerical problem rather than depend directly on a third-party solver API.

The target interface is conceptually:

```cpp
LeastSquaresProblem problem{
    /* residual definition */
};

SolverOptions options;

auto result = solve(problem, initial_guess, options);
```

or:

```cpp
OptimizationProblem problem{
    /* objective and constraints */
};

auto result = solve(problem, initial_guess);
```

Backend-specific details remain isolated behind adapters.

---

## Consistent Results

All solvers should expose a common result model containing information such as:

```cpp
struct SolverResult {
    SolverStatus status;

    std::size_t iterations;
    std::size_t function_evaluations;
    std::size_t gradient_evaluations;
    std::size_t jacobian_evaluations;

    double initial_objective;
    double final_objective;

    Backend backend;
    Algorithm algorithm;
};
```

This makes native and third-party engines observable through the same interface.

---

# Dependencies

## Core

The native library requires:

```text
C++17
CMake
Eigen
```

Optional project dependencies may also be used for logging and testing.

## Advanced Backends

```text
Ipopt       constrained nonlinear programming
PETSc/TAO   large-scale optimization
POUNDERS    included through PETSc/TAO
```

These dependencies are optional.

A user requiring only native root finding, LM, Gauss-Newton, BFGS or L-BFGS should not need to install Ipopt or PETSc.

---

# Building

Clone the repository:

```bash
git clone https://github.com/KhwarizmiAnalytix/Solvers.git
cd Solvers
```

Configure:

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release
```

Build:

```bash
cmake --build build --parallel
```

Run tests:

```bash
ctest --test-dir build --output-on-failure
```

---

# Optional Backends

The intended configuration model is:

```bash
cmake -S . -B build \
    -DSOLVERS_ENABLE_IPOPT=ON \
    -DSOLVERS_ENABLE_PETSC=ON
```

A minimal native-only build should remain:

```bash
cmake -S . -B build
```

---

# Installing Optional Dependencies

## Ipopt

Repository:

```text
https://github.com/coin-or/Ipopt
```

Ipopt requires BLAS/LAPACK and a sparse linear solver such as MUMPS.

COIN-OR's `coinbrew` can be used to build the complete dependency stack.

---

## PETSc / TAO

Repository:

```text
https://github.com/petsc/petsc
```

On macOS:

```bash
brew install petsc
```

TAO and POUNDERS are included with PETSc.

---

# Testing

Numerical solvers require more than simple convergence tests.

The test suite should cover:

- normal convergence,
- poor initial guesses,
- endpoint roots,
- multiple roots,
- almost-zero derivatives,
- badly scaled variables,
- badly scaled residuals,
- nearly singular Jacobians,
- rank-deficient Jacobians,
- flat objectives,
- numerical overflow,
- `NaN` and `Inf` propagation,
- maximum-iteration termination.

Sanitizer and coverage builds are recommended as part of continuous integration.

---

# Benchmarking

Performance comparisons should always be like-for-like.

Recommended benchmark groups:

## Root Finding

```text
Native Brent
vs
Boost.Math reference
```

## Nonlinear Least Squares

```text
Native LM / GN
vs
reference nonlinear least-squares implementations
```

## Optimization

```text
Native L-BFGS
vs
established L-BFGS reference implementations
```

Each benchmark should report:

```text
wall-clock time
iterations
function evaluations
Jacobian evaluations
final residual
final objective
```

and should document:

```text
CPU
compiler
compiler flags
problem dimensions
initial guess
solver tolerances
```

Performance claims should be based on reproducible benchmark results rather than isolated timings.

---

# Project Structure

A problem-oriented layout is recommended:

```text
Solvers/
│
├── include/solvers/
│   ├── problem.hpp
│   ├── least_squares_problem.hpp
│   ├── optimization_problem.hpp
│   ├── constraints.hpp
│   ├── options.hpp
│   ├── result.hpp
│   ├── status.hpp
│   └── solve.hpp
│
├── src/
│   ├── dispatch.cpp
│   │
│   └── backends/
│       ├── native/
│       │   ├── roots.cpp
│       │   ├── lm.cpp
│       │   ├── gauss_newton.cpp
│       │   └── lbfgs.cpp
│       │
│       ├── ipopt/
│       │   └── ipopt_adapter.cpp
│       │
│       └── petsc/
│           ├── tao_adapter.cpp
│           └── pounders_adapter.cpp
│
├── tests/
└── benchmarks/
```

---

# Roadmap

The project direction is intentionally focused.

### Native

- [x] Scalar root finding
- [x] Polynomial solving
- [x] Levenberg-Marquardt
- [x] Gauss-Newton
- [x] BFGS / L-BFGS

### Architecture

- [ ] Unified `LeastSquaresProblem`
- [ ] Unified `OptimizationProblem`
- [ ] Common `SolverResult`
- [ ] Automatic problem-trait based dispatch
- [ ] Reusable solver workspaces
- [ ] Consistent absolute/relative tolerances

### Advanced Backends

- [ ] Ipopt adapter
- [ ] PETSc/TAO adapter
- [ ] POUNDERS adapter

### Numerical Infrastructure

- [ ] Parameter scaling
- [ ] Residual scaling
- [ ] Weighted nonlinear least squares
- [ ] Jacobian validation
- [ ] Matrix-free Jacobian-vector products
- [ ] Hessian-vector products

### Engineering

- [ ] Reproducible benchmark suite
- [ ] CMake install/export package
- [ ] External consumer build test
- [ ] Continuous sanitizer testing
- [ ] Performance regression tests

---

# Philosophy

`Solvers` is not intended to become a collection of wrappers around every optimization library available.

The project follows a simpler principle:

```text
Use optimized native algorithms for common numerical problems.

Use specialized external engines only when they introduce a
meaningfully different capability.

Expose everything through one coherent C++ interface.
```

The result is intended to remain lightweight for ordinary use while scaling to substantially harder numerical optimization problems when required.

---

# License

See the repository `LICENSE` file for licensing information.

---

# Repository

```text
https://github.com/KhwarizmiAnalytix/Solvers
```
