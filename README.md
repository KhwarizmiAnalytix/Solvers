# 🚀 Solvers: High-Performance Numerical Optimization & Root Finding

![C++17](https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B)
![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey.svg)

A comprehensive, production-ready C++ library for solving nonlinear equations, finding roots, and performing robust numerical optimization. Built for quantitative finance, scientific computing, and high-performance applications.

**Developed and maintained by [KhwarizmiAnalytix](https://github.com/KhwarizmiAnalytix)**

## Overview

The Solvers library provides battle-tested algorithms for:

- **Root Finding:** Newton-Raphson, Brent's method, bracketing methods, and more
- **Optimization:** Levenberg-Marquardt, L-BFGS, Gauss-Newton solvers
- **Least-Squares Fitting:** Curve fitting with adaptive trust-region strategies
- **Polynomial Solving:** Fast algorithms for polynomial roots
- **Multiple Backends:** Native implementations plus optional Ceres and NLopt integration

Whether you're calibrating financial models, fitting experimental data, or solving complex nonlinear systems, Solvers delivers the speed and reliability your application demands.

## ⚡ Key Features

| Feature | Description | Performance |
|---------|-------------|-------------|
| **Fast Root Finding** | Newton-Raphson, Brent, False Position, Bisection | 60-100x faster than general-purpose solvers |
| **Robust Optimization** | LM, L-BFGS, Gauss-Newton with automatic algorithm selection | 100% convergence on test suite |
| **Least-Squares Solver** | Optimized for overdetermined systems with trust-region methods | Adaptive Jacobian-based stepping |
| **Extensible Design** | Pluggable backends (Native, Ceres, NLopt) | Choose speed or robustness per problem |
| **Production Quality** | Comprehensive error handling, numerical stability | Used in quantitative finance workflows |
| **Multi-Backend Support** | Ceres, NLopt, and native implementations | Optional GPU acceleration with Ceres |

## 🎯 Why Choose Solvers?

### Speed
- Direct root finders optimized for scalar problems
- LM solver with adaptive stepping for faster convergence
- Competitive performance against specialized libraries

### Robustness
- Automatic algorithm selection based on problem characteristics
- Trust-region strategies prevent divergence
- Comprehensive error handling and convergence checks

### Flexibility
- Multiple solver implementations for different use cases
- Configurable precision, iteration limits, and tolerances
- Simple unified interface across different solver types

### Production-Ready
- Mature C++ codebase with extensive testing
- Well-documented algorithms and parameters
- Integrated error reporting and diagnostics

## 📚 Solver Algorithms

### Root Finding Algorithms

Find zeros of univariate functions with proven convergence:

| Algorithm | Type | Speed | Requirements | Best For |
|-----------|------|-------|--------------|----------|
| **Newton-Raphson** | Open | ⚡⚡⚡ | Derivative | Quick convergence when derivative available |
| **Brent's Method** | Bracketing | ⚡⚡ | Bracket | Robust, no derivative needed |
| **False Position** | Bracketing | ⚡⚡ | Bracket | Faster than bisection |
| **Bisection** | Bracketing | ⚡ | Bracket | Guaranteed convergence, simple |
| **Ridders' Method** | Hybrid | ⚡⚡ | Bracket | Excellent for smooth functions |
| **Secant Method** | Open | ⚡⚡ | Two guesses | No derivative, superlinear convergence |

### Optimization Algorithms

Minimize scalar or vector-valued functions with multiple backends:

| Algorithm | Type | Convergence | Use Case |
|-----------|------|-------------|----------|
| **Levenberg-Marquardt** | Trust-Region | Quadratic | **Recommended** for least-squares & curve fitting |
| **Gauss-Newton** | Gradient-Based | Linear | Fast for square systems (m ≈ n) |
| **L-BFGS** | Quasi-Newton | Superlinear | General unconstrained optimization |
| **Ceres Solver** | Backend | Quadratic | Advanced least-squares (optional) |
| **NLopt** | Backend | Problem-dependent | Non-smooth & constrained (optional) |

### Polynomial Solving

Specialized algorithms for polynomial root finding with numerical stability.

## 🏗️ Architecture

```
include/
├── solvers/                    # Core solver implementations
│   ├── root_finding_algorithms.h    # Root finders (Newton, Brent, etc.)
│   ├── levenberg_marquardt_solver.h # LM optimization
│   ├── lbfgs_solver.h              # L-BFGS optimization
│   ├── gauss_newton_solver.h       # Gauss-Newton solver
│   ├── polynomial_solver.h         # Polynomial root finding
│   ├── ceres_solver.h              # Ceres backend (optional)
│   └── nlopt_solver.h              # NLopt backend (optional)
├── solver_wrapper.h            # Unified solver interface
├── solver_options/             # Configuration for each solver
└── solver_output.h             # Results and diagnostics
```

## 🔧 Source Organization

The library is organized by mathematical domains:

- **Root Finding:** Scalar equation solvers
- **Optimization:** Multivariate function minimization
- **Polynomial:** Special-purpose polynomial solvers
- **Options:** Configuration objects for each solver type
- **Common:** Shared utilities and helpers

## 🚀 Quick Start

### Installation

#### Using CMake

```cmake
find_package(Solvers REQUIRED)
target_link_libraries(your_target PRIVATE Solvers::Solvers)
```

#### Building from Source

```bash
git clone https://github.com/KhwarizmiAnalytix/Solvers.git
cd Solvers
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

### Basic Usage Examples

#### 1. Finding a Root (Scalar Equation)

```cpp
#include "solvers/root_finding_algorithms.h"
#include "solver_options/root_finding_options.h"

using namespace solverslib;

// Define your function: f(x) = x^2 - 4 (root at x = 2)
auto residual = [](double x) { return x * x - 4.0; };
auto residual_with_derivative = [](double x, double& df_dx) {
    df_dx = 2.0 * x;
    return x * x - 4.0;
};

// Set up options
auto options = root_finding_options_builder()
    .with_tolerance_function(1e-14)
    .with_tolerance_parameter(1e-14)
    .with_max_iterations(100)
    .build();

// Solve using Newton-Raphson (requires derivative)
double root = 2.5;  // Initial guess
bool converged = root_finding_algorithms::newton_raphson(
    residual_with_derivative, root, root, options);

if (converged) {
    std::cout << "Root found: " << root << std::endl;
}
```

#### 2. Curve Fitting (Least-Squares Optimization)

```cpp
#include "solver_wrapper.h"
#include "solver_options/solver_options_lm.h"

// Define your model: y = a*exp(b*t)
auto residuals_func = [](const vector_type& params, vector_type& residuals) {
    double a = params(0);
    double b = params(1);
    for (size_t i = 0; i < data_points.size(); ++i) {
        double predicted = a * std::exp(b * times[i]);
        residuals(i) = data_points[i] - predicted;
    }
};

auto jacobian_func = [](const vector_type& params, matrix_type& jacobian) {
    double a = params(0);
    double b = params(1);
    for (size_t i = 0; i < data_points.size(); ++i) {
        jacobian(i, 0) = -std::exp(b * times[i]);  // ∂r/∂a
        jacobian(i, 1) = -a * times[i] * std::exp(b * times[i]);  // ∂r/∂b
    }
};

solver_wrapper wrapper(2, data_points.size(), residuals_func, jacobian_func);
auto options = std::make_shared<solver_options_lm>(
    500,      // max iterations
    1e-14,    // gradient tolerance
    1e-14,    // parameter tolerance
    1e-14     // residual tolerance
);

std::vector<double> parameters = {1.0, 0.1};  // Initial guess [a, b]
bool converged = wrapper.solve(parameters, options);

if (converged) {
    std::cout << "Fitted: a=" << parameters[0] << ", b=" << parameters[1] << std::endl;
}
```

#### 3. Choosing the Right Solver

```cpp
// For root finding: Use root_finding_algorithms
// - Fastest for scalar f(x) = 0
// - Newton-Raphson if derivative available
// - Brent's method if only function values

// For least-squares: Use Levenberg-Marquardt
// - Best all-around choice for fitting
// - min_x ||f(x)||^2
// - Handles overdetermined systems

// For general optimization: Use L-BFGS
// - General unconstrained min_x f(x)
// - Doesn't require Jacobian
// - Good for smooth functions
```

## 📊 Performance Characteristics

### Root Finding Performance

```
Problem Type    | Method           | Time    | Speed vs LM
----------------|------------------|---------|-------------
Quadratic       | Newton-Raphson   | 0.0000ms | 100x faster
Cubic           | Brent's Method   | 0.0001ms | 70x faster
Transcendental  | Newton-Raphson   | 0.0002ms | 50x faster
```

### Optimization Performance

```
Problem         | Solver  | Time     | Convergence
----------------|---------|----------|------------
Linear Scalar   | LM      | 0.0060ms | ✓ 100%
Rosenbrock 2D   | LM      | 0.0203ms | ✓ 100%
Exponential Fit | LM      | 0.0091ms | ✓ 100%
```

See [Benchmark Report](./BENCHMARKS.md) for detailed comparisons.

## 🛠️ Building with Options

```bash
# Enable benchmarks
cmake -DSOLVERS_ENABLE_BENCHMARKS=ON ..

# Optional: Enable Ceres backend
cmake -DSOLVERS_ENABLE_CERES=ON ..

# Optional: Enable NLopt backend
cmake -DSOLVERS_ENABLE_NLOPT=ON ..

# Build specific benchmark
cmake --build . --target RootFindersVsLMBenchmark
./Testing/Cxx/RootFindersVsLMBenchmark
```

## 📋 Dependencies

- **Required:** C++17 compiler, CMake 3.22+
- **Optional:** Ceres Solver (for Ceres backend)
- **Optional:** NLopt library (for NLopt backend)
- **Testing:** Google Test (for unit tests)

## 💡 Best Practices & Recommendations

### Choosing Your Algorithm

| Problem | Recommended | Why |
|---------|-------------|-----|
| Find root of f(x) = 0 | Newton-Raphson or Brent | 50-100x faster than general solvers |
| Fit curve y = f(x; θ) | Levenberg-Marquardt | Designed for least-squares, highly robust |
| General optimization min f(x) | L-BFGS | No Jacobian needed, reliable convergence |
| Polynomial roots | Polynomial Solver | Specialized algorithm, numerically stable |
| Real-time applications | Gauss-Newton (if m≈n) | Fastest option, but less stable than LM |

### Common Pitfalls to Avoid

✗ **Don't:** Use general solvers for simple root finding  
✓ **Do:** Use Newton-Raphson or Brent's method for f(x) = 0

✗ **Don't:** Use L-BFGS for least-squares curve fitting  
✓ **Do:** Use Levenberg-Marquardt for minimizing residuals

✗ **Don't:** Set unrealistic tolerances (1e-20 on double precision)  
✓ **Do:** Use 1e-14 to 1e-15 for IEEE 754 double precision

✗ **Don't:** Ignore initial guess quality for open methods  
✓ **Do:** Provide good starting point or use bracketing methods

### Tips for Success

1. **Provide Jacobians:** Derivatives significantly accelerate convergence
2. **Scale Your Problem:** Normalize variables to similar ranges
3. **Test Convergence:** Always check return value and convergence criteria
4. **Profile First:** Benchmark your specific problem before optimizing
5. **Use Appropriate Tolerance:** Balance accuracy vs computation time

## 🧪 Testing & Benchmarks

Run the test suite:

```bash
cd build
ctest
```

Run performance benchmarks:

```bash
cmake -DSOLVERS_ENABLE_BENCHMARKS=ON ..
cmake --build . 

./Testing/Cxx/RootFindersBenchmark      # Root finding algorithms
./Testing/Cxx/SolversBenchmark           # Optimization algorithms
./Testing/Cxx/RootFindersVsLMBenchmark  # Root finders vs LM comparison
```

## 📖 Documentation

- [Algorithm Details](./docs/algorithms.md)
- [API Reference](./docs/api.md)
- [Benchmark Results](./docs/benchmarks.md)
- [Configuration Guide](./docs/configuration.md)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

- Bug reports: [GitHub Issues](https://github.com/KhwarizmiAnalytix/Solvers/issues)
- Feature requests: [GitHub Discussions](https://github.com/KhwarizmiAnalytix/Solvers/discussions)
- Pull requests: [GitHub PRs](https://github.com/KhwarizmiAnalytix/Solvers/pulls)

## 📄 License

BSD-3-Clause License

Copyright (c) 2024 KhwarizmiAnalytix

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## 🙏 Acknowledgments

Built on decades of numerical computing research including:
- Levenberg (1944), Marquardt (1963) - Trust-region methods
- Nocedal, Wright - Numerical Optimization fundamentals
- Brent (1973) - Root finding algorithms
- Ceres Solver team - Optimization research and implementation

## 📮 Support

- **Questions?** [GitHub Discussions](https://github.com/KhwarizmiAnalytix/Solvers/discussions)
- **Found a bug?** [GitHub Issues](https://github.com/KhwarizmiAnalytix/Solvers/issues)
- **Want to contribute?** See [CONTRIBUTING.md](./CONTRIBUTING.md)

---

**Made with ❤️ by [KhwarizmiAnalytix](https://github.com/KhwarizmiAnalytix)**  
For quantitative finance, scientific computing, and high-performance applications.
