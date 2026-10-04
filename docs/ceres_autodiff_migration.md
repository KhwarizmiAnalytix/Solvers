# Ceres Automatic Differentiation Migration Guide

This document guides migration from legacy Jacobian APIs to modern automatic differentiation using Ceres.

## Overview

The Solvers library now provides **automatic differentiation (AD)** via Ceres, eliminating the need to hand-code Jacobians for many problems. The new `derivative_mode` policy ensures consistent behavior across all backends.

## Key Concepts

### Derivative Modes

Four policies control how derivatives are computed:

| Mode | Behavior | Use Case |
|------|----------|----------|
| `automatic` (default) | Try supplied → AD → numeric | Development; safe fallback chain |
| `supplied` | Require hand-coded Jacobian | Existing optimized code |
| `automatic_differentiation` | Require Ceres AD | New problems without manual Jacobian |
| `finite_difference` | Force numerical differentiation | Testing, debugging, nondifferentiable components |

### Cascade Policy

The default `automatic` mode implements a cascade:
1. **Supplied Jacobian** (fastest, if available)
2. **AD Provider** (fast, accurate)
3. **Numerical/Derivative-Free** (fallback)

This prevents silent degradation: explicit requests are validated before solving.

## Migration Examples

### Before: Hand-Coded Jacobian

```cpp
#include <solvers/api/solve.h>

struct MyModel {
    void residuals(const vector_type& x, vector_type& r) const {
        r[0] = x[0] * x[0] - 4.0;
        r[1] = x[1] * x[1] - 9.0;
    }

    void jacobian(const vector_type& x, matrix_type& J) const {
        J(0, 0) = 2.0 * x[0];
        J(0, 1) = 0.0;
        J(1, 0) = 0.0;
        J(1, 1) = 2.0 * x[1];
    }
};

int main() {
    MyModel model;
    api::least_squares_problem problem;
    problem.num_parameters = 2;
    problem.num_residuals = 2;
    problem.residuals = [&](const vector_type& x, vector_type& r) {
        model.residuals(x, r);
    };
    problem.set_jacobian([&](const vector_type& x, matrix_type& J) {
        model.jacobian(x, J);
    });

    vector_type x = {1.0, 1.0};
    api::solve_options opts;
    auto result = api::solve(problem, x, opts);

    return result.converged() ? 0 : 1;
}
```

### After: Automatic Differentiation

```cpp
#include <solvers/integrations/ceres_autodiff.h>

// Single templated functor works with double and Ceres Jets
struct MyResiduals {
    template <typename T>
    bool operator()(const T* const x, T* r) const {
        r[0] = x[0] * x[0] - T(4.0);
        r[1] = x[1] * x[1] - T(9.0);
        return true;
    }
};

int main() {
    // Create problem with AD provider
    auto problem = make_ceres_autodiff_problem(2, 2, MyResiduals{});

    vector_type x = {1.0, 1.0};
    api::solve_options opts;
    opts.backend = api::backend::ceres;
    opts.derivatives = api::derivative_mode::automatic_differentiation;

    auto result = api::solve(problem, x, opts);

    return result.converged() ? 0 : 1;
}
```

**Benefits of AD:**
- ✓ Single source of truth (one templated functor)
- ✓ No hand-coded Jacobian errors
- ✓ Automatic for complex expressions (exp, sin, etc.)
- ✓ Still uses Jacobian if you provide it (cascade)

## Derivative Policy Examples

### Explicit AD Requirement

Use when AD is critical and you want to fail fast if unavailable:

```cpp
api::solve_options opts;
opts.derivatives = api::derivative_mode::automatic_differentiation;
opts.backend = api::backend::ceres;
// Solves with AD Jacobian or returns unsupported_capability error
auto result = api::solve(problem, x, opts);
```

### Fallback to Numerical

Use to test AD quality against finite differences:

```cpp
api::solve_options opts_ad;
opts_ad.derivatives = api::derivative_mode::automatic_differentiation;
auto result_ad = api::solve(problem, x, opts_ad);

api::solve_options opts_fd;
opts_fd.derivatives = api::derivative_mode::finite_difference;
auto result_fd = api::solve(problem, x, opts_fd);

// Compare results
std::cout << "AD solution: " << result_ad.objective << std::endl;
std::cout << "FD solution: " << result_fd.objective << std::endl;
```

## Writing Templated Residuals

### Guidelines

1. **Use `T` for scalar parameters**
   ```cpp
   T x = params[0];  // T = double or Jet
   ```

2. **Use `T()` for constants**
   ```cpp
   T four = T(4.0);
   ```

3. **Use standard math functions**
   ```cpp
   using std::sin, std::cos, std::exp;  // ADL enables Jet overloads
   T val = sin(x) + exp(y);
   ```

4. **Return `true` for valid points, `false` for domain violations**
   ```cpp
   template <typename T>
   bool operator()(const T* const x, T* r) const {
       if (x[0] <= T(0.0)) return false;  // Domain error
       r[0] = sqrt(x[0]);
       return true;
   }
   ```

### Example: Exponential Fit

```cpp
struct ExponentialFitResiduals {
    std::vector<double> times;
    std::vector<double> observations;

    template <typename T>
    bool operator()(const T* const params, T* residuals) const {
        // params[0] = amplitude, params[1] = decay_rate
        T amp = params[0];
        T decay = params[1];

        for (size_t i = 0; i < times.size(); ++i) {
            T t = T(times[i]);
            T obs = T(observations[i]);
            T predicted = amp * exp(-decay * t);
            residuals[i] = predicted - obs;
        }
        return true;
    }
};

int main() {
    std::vector<double> times = {0, 1, 2, 3, 4};
    std::vector<double> obs = {10, 4, 1.6, 0.64, 0.256};

    ExponentialFitResiduals fit{times, obs};
    auto problem = make_ceres_autodiff_problem(2, 5, fit);

    vector_type x = {1.0, 0.1};
    api::solve_options opts;
    opts.backend = api::backend::ceres;

    auto result = api::solve(problem, x, opts);

    std::cout << "Amplitude: " << result.parameters[0] << std::endl;
    std::cout << "Decay rate: " << result.parameters[1] << std::endl;

    return 0;
}
```

## Stride and Performance

Ceres AD uses a **stride of 4** by default, meaning it computes Jacobians in groups:
- For n parameters, Ceres evaluates the residual ceil(n/4) times with Jet arithmetic
- This balances memory usage and computational efficiency

For optimal performance:
- Keep residual functors simple (avoid unnecessary branches)
- Use `T()` instead of implicit conversions
- Provide bounds when available (improves algorithm selection)

## Backwards Compatibility

### Legacy API (Deprecated)

The old `std::any` based API still works but is **not recommended**:

```cpp
// NOT recommended - type erasure prevents AD instantiation
problem.set_templated_residuals(MyResiduals{});
// Only works if Jacobian callback is also provided
```

**Why it's deprecated:**
- Cannot instantiate template member functions from non-templated code
- Requires separate Jacobian callback
- No provider factory pattern

### Recommended Replacement

```cpp
// Use provider factory pattern instead
auto problem = make_ceres_autodiff_problem(n, m, MyResiduals{});
// Provides AD automatically, no separate Jacobian needed
```

## Result Interpretation

The `solver_result` now tracks the effective derivative source:

```cpp
auto result = api::solve(problem, x, opts);

if (result.effective_derivative_source) {
    switch (*result.effective_derivative_source) {
    case api::derivative_mode::supplied:
        std::cout << "Used hand-coded Jacobian\n";
        break;
    case api::derivative_mode::automatic_differentiation:
        std::cout << "Used Ceres automatic differentiation\n";
        break;
    case api::derivative_mode::finite_difference:
        std::cout << "Used numerical differentiation\n";
        break;
    default:
        break;
    }
}
```

## Troubleshooting

### Build Error: "Ceres backend not compiled in"

**Problem:** `SOLVERS_ENABLE_CERES=OFF` in CMake

**Solution:**
```bash
cmake -S . -B build -DSOLVERS_ENABLE_CERES=ON
cmake --build build
```

### Error: "automatic_differentiation required but no AD provider available"

**Problem:** Using `derivative_mode::automatic_differentiation` without `make_ceres_autodiff_problem()`

**Solution:**
```cpp
// Create problem with AD
auto problem = make_ceres_autodiff_problem(n, m, functor);
// Don't just set jacobian and expect AD to work
```

### Jacobian computation mismatch

**Problem:** Hand-coded and AD Jacobians disagree

**Solution:**
```cpp
// Test AD vs finite differences
opts_ad.derivatives = api::derivative_mode::automatic_differentiation;
opts_fd.derivatives = api::derivative_mode::finite_difference;

auto result_ad = api::solve(problem, x, opts_ad);
auto result_fd = api::solve(problem, x, opts_fd);

// If results differ, check:
// 1. T() usage in templated functor
// 2. Function overloads (use std::sin not sin(double))
// 3. Domain violations (negative sqrt, etc.)
```

## See Also

- `include/solvers/integrations/ceres_autodiff.h` — Implementation
- `Testing/Cxx/TestCeresAutoDiff.cpp` — Test examples
- [Ceres AD documentation](http://ceres-solver.readthedocs.io/latest/automatic_derivatives.html)
