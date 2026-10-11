# Solvers

[![CI](https://github.com/KhwarizmiAnalytix/Solvers/actions/workflows/ci.yml/badge.svg)](https://github.com/KhwarizmiAnalytix/Solvers/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/KhwarizmiAnalytix/Solvers/branch/master/graph/badge.svg)](https://codecov.io/gh/KhwarizmiAnalytix/Solvers)

C++ numerical solvers for scalar roots, nonlinear least squares, and scalar-objective optimization. The native solvers use Eigen for dense linear algebra and the bundled Logging library for diagnostics. Ceres, Ipopt, NLopt, and PETSc/TAO are optional backends.

## Implemented solvers

| Problem | Implementation | Options / entry point |
| --- | --- | --- |
| Scalar roots | Bisection, false position, Ridders, Brent, Dekker, Newton-Raphson, secant | `detail::run_*` with `root_finding_options` |
| Polynomial roots | Quadratic, cubic, and quartic convenience functions | `polynomial_solver` |
| Nonlinear least squares | Native Levenberg–Marquardt | `solver_options_lm_builder` |
| Nonlinear least squares | Native Gauss–Newton with backtracking | `solver_options_gn_builder` |
| Nonlinear least squares | Native Riemann-normal-coordinate LM (RNC-LM), orders 1–4 | `solver_options_rnc_lm_builder` |
| Scalar-objective optimization | Native L-BFGS | `solver_options_bfgs_builder` |
| Least squares with optional box bounds | Ceres | `solver_options_ceres_builder` |
| Least squares or scalar objectives with optional box bounds | Ipopt | `solver_options_ipopt_builder` |
| Least squares or scalar objectives | NLopt: L-BFGS, MMA, SLSQP, CCSAQ, Nelder–Mead, Subplex | `solver_options_nlopt_builder` |
| Least squares or scalar objectives | PETSc/TAO: POUNDERS, BRGN, NLS, NTR, NTL, LMVM, BQNLS, BNLS | `solver_options_petsc_builder` |

All names above live in `solverslib`, except the unified problem and result types, which live in `solverslib::api`. Least-squares and optimization problems are passed to `api::solve(problem, initial_guess, options)`.

## Build

Clone with submodules, including the nested dependencies of Logging:

```bash
git clone --recurse-submodules https://github.com/KhwarizmiAnalytix/Solvers.git
cd Solvers
# For an existing checkout:
git submodule update --init --recursive
```

The top-level project requires CMake 3.20 or newer and declares C++17. The bundled Logging dependency defaults to C++20 for its own compilation, so the default build needs a compiler with C++20 support. Eigen and Logging are required bundled dependencies; GoogleTest is bundled and used when testing is enabled. The Python scripts are build tools, not Python bindings for the library.

### Native build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Direct CMake builds default to static libraries, enable tests when Solvers is the top-level project, and disable optional solver backends and benchmarks. With a multi-configuration generator such as Xcode or Visual Studio, add `--config Release` to the build command and `-C Release` to CTest.

The repository's setup helper, also used by CI, accepts dotted commands. Run it from `Scripts/`:

```bash
cd Scripts
python3 setup.py --help
python3 setup.py config.build.test.release.static
```

### Optional backends

Enable only the backends you need:

```bash
cmake -S . -B build-backends -DCMAKE_BUILD_TYPE=Release \
    -DSOLVERS_ENABLE_CERES=ON \
    -DSOLVERS_ENABLE_IPOPT=ON \
    -DSOLVERS_ENABLE_NLOPT=ON \
    -DSOLVERS_ENABLE_PETSC=ON
cmake --build build-backends --parallel
ctest --test-dir build-backends --output-on-failure
```

| CMake option | Dependency discovery |
| --- | --- |
| `SOLVERS_ENABLE_CERES` | Builds `ThirdParty/ceres` (requires CMake 3.22 or newer) unless the parent project already provides `Ceres::ceres` |
| `SOLVERS_ENABLE_IPOPT` | System library through pkg-config module `ipopt` |
| `SOLVERS_ENABLE_NLOPT` | System library through pkg-config module `nlopt` |
| `SOLVERS_ENABLE_PETSC` | System library through pkg-config module `PETSc`, plus MPI's C component |
| `SOLVERS_ENABLE_BENCHMARKS` | System Google Benchmark through pkg-config or `find_package(benchmark)` |

Other top-level flags are `SOLVERS_ENABLE_TESTING`, `BUILD_SHARED_LIBS`, `SOLVERS_ENABLE_SANITIZER` (address and undefined behavior), `SOLVERS_ENABLE_COVERAGE`, `SOLVERS_ENABLE_CLANGTIDY`, and `SOLVERS_ENABLE_IWYU`.

A request for a compiled-out optional backend returns `solver_status::backend_unavailable`. POUNDERS is a TAO algorithm, enabled by `SOLVERS_ENABLE_PETSC`; it has no separate build flag.

### CMake integration and installation

For a source checkout with initialized submodules:

```cmake
add_subdirectory(path/to/Solvers)
target_link_libraries(my_app PRIVATE Solvers::Solvers)
```

To install a native build:

```bash
cmake --install build --prefix "$PWD/install"
```

The exported package is discoverable with `-DCMAKE_PREFIX_PATH=/path/to/Solvers/install`:

```cmake
find_package(Solvers CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE Solvers::Solvers)
```

Installation includes solver headers, the library, Eigen headers, and Logging's package files. **Current packaging limitation:** the default static installation omits Logging's Loguru link dependency. Package discovery succeeds, but consumer executables fail to link with unresolved `loguru::` symbols. Use the source-based `add_subdirectory` integration above until that dependency is exported or linked separately.

A Ceres-enabled installed package also needs a separately installed Ceres dependency. For a static NLopt-enabled installation, the consumer currently needs to create `PkgConfig::NLOPT` before finding Solvers, because `SolversConfig.cmake` does not yet discover NLopt:

```cmake
find_package(PkgConfig REQUIRED)
pkg_check_modules(NLOPT REQUIRED IMPORTED_TARGET nlopt)
find_package(Solvers CONFIG REQUIRED)
```

PETSc-enabled consumers should enable the C language so MPI's C target can be discovered.

### Bazel

The Bazel graph builds the native library, unit tests, and Google Benchmark executables. All optional backend macros are set to zero in `BUILD.bazel`; use CMake for optional backends. The repository pins Bazel 8.0.0 in `.bazelversion`.

```bash
bazel build //:Solvers
bazel test //Testing/Cxx/...
bazel test //Testing/Benchmark/...
```

The helper provides separate test and benchmark phases:

```bash
cd Scripts
python3 setup_bazel.py build.test.release
python3 setup_bazel.py build.benchmark.release
```

Bazel packaging targets are `//:solvers_pkg_tar` and `//:install`:

```bash
bazel build //:solvers_pkg_tar
bazel run //:install -- --destdir=/path/to/staging
```

These stage solver headers and library artifacts; they do not produce the CMake package or bundle its dependency installation tree.

## Usage

### Scalar root finding

The current root interface returns a `detail::root_run` containing `root`, `iterations`, and an `outcome` of `converged`, `iteration_limit`, or `degenerate`:

```cpp
#include <iostream>
#include "solvers/root_finding_algorithms.h"

int main() {
    const auto options = solverslib::root_finding_options_builder()
        .with_tolerance_function(1e-12)
        .with_tolerance_parameter(1e-12)
        .with_max_iterations(100)
        .build();

    const auto result = solverslib::detail::run_brent(
        [](double x) { return x * x - 2.0; }, 0.0, 2.0, options);

    std::cout << result.root << '\n';
    return result.outcome == solverslib::detail::root_outcome::converged ? 0 : 1;
}
```

Bracketing methods require a valid sign-changing bracket. Newton-Raphson and Dekker take a callback of signature `double(double x, double& derivative)`. Invalid brackets and vanishing derivatives or secant denominators can throw.

`polynomial_solver` in `solvers/polynomial_solver.h` provides single-root convenience functions for monic quadratic, cubic, and quartic polynomials. These return one selected real root, not all roots. The quartic returns the largest real root clamped from below by its `threshold` argument.

### Least squares with native LM

Assign callbacks directly to the problem fields and select the solver through its options builder. `build()` returns a `std::shared_ptr<const solver_options_lm>`; pass `*options` to `solve`:

```cpp
#include <iostream>
#include "api/dispatch.h"
#include "solver_options/solver_options_lm.h"

int main() {
    using namespace solverslib;
    using namespace solverslib::api;

    least_squares_problem problem;
    problem.num_parameters = 1;
    problem.num_residuals = 1;
    problem.residuals = [](const vector_type& x, vector_type& r) {
        r.resize(1);
        r[0] = x[0] * x[0] - 4.0;
    };
    problem.jacobian = [](const vector_type& x, matrix_type& j) {
        j.resize(1, 1);
        j(0, 0) = 2.0 * x[0];
    };

    const auto options = solver_options_lm_builder()
        .with_max_iterations(100)
        .with_function_tolerance(1e-10)
        .with_gradient_tolerance(1e-10)
        .with_parameter_tolerance(1e-10)
        .build();

    vector_type start(1);
    start[0] = 3.0;
    const auto result = solve(problem, start, *options);
    std::cout << to_string(result.status) << ": " << result.parameters[0] << '\n';
    return result.converged() ? 0 : 1;
}
```

Omit `problem.jacobian` to use native LM's finite differences. Gauss–Newton uses the same problem structure with `solver_options_gn_builder`. Native LM supports Nielsen, Marquardt, and quadratic-interpolation damping, optional geodesic acceleration and bold acceptance, and `NORMAL_LDLT` or `AUGMENTED_QR` linear solves. Configure these through the builder's `with_...` methods.

### Scalar-objective optimization with native L-BFGS

```cpp
#include <iostream>
#include "api/dispatch.h"
#include "solver_options/solver_options_bfgs.h"

int main() {
    using namespace solverslib;
    using namespace solverslib::api;

    optimization_problem problem;
    problem.num_parameters = 1;
    problem.objective = [](const vector_type& x) {
        return (x[0] - 2.0) * (x[0] - 2.0);
    };
    problem.gradient = [](const vector_type& x, vector_type& g) {
        g.resize(1);
        g[0] = 2.0 * (x[0] - 2.0);
    };

    const auto options = solver_options_bfgs_builder()
        .with_max_iterations(100)
        .with_gradient_tolerance(1e-10)
        .build();

    vector_type start(1);
    start[0] = 3.0;
    const auto result = solve(problem, start, *options);
    std::cout << to_string(result.status) << ": " << result.parameters[0] << '\n';
    return result.converged() ? 0 : 1;
}
```

Native L-BFGS can approximate a missing gradient with finite differences. Its builder configures history size, line-search strategy, step limits, and finite-difference scaling.

## Solver selection and derivatives

Dispatch uses `options.solver()` and the concrete options type. There is no automatic backend-selection policy in the current `solve` interface. To switch solvers, build the corresponding options and pass them to the same problem overload.

| Solver | Derivative contract |
| --- | --- |
| Native LM / Gauss–Newton | Optional `problem.jacobian`; finite differences when absent |
| Native L-BFGS | Optional `problem.gradient`; finite differences when absent |
| RNC-LM | Requires `problem.curve_derivatives`; no numerical fallback |
| Ceres | Optional `problem.jacobian`; numerical differentiation when absent |
| Ipopt least squares | Requires `problem.jacobian`; dispatch constructs the scalar cost and gradient |
| NLopt least squares | Supplied Jacobian gives a scalar-cost gradient; omit it only for derivative-free algorithms |
| PETSc/TAO least squares | Residual callback and optional Jacobian; select an algorithm compatible with those callbacks |
| Ipopt / NLopt / PETSc scalar objectives | Dispatch requires `problem.gradient`; an optional Hessian is passed to Ipopt and PETSc |

The public problems currently store callbacks, not derivative-provider objects. There is no bundled `least_squares(...)` or `auto_diff()` factory in the current headers. Enabling Ceres builds its optimizer adapter; it does not automatically differentiate a templated residual model. An application may compute derivatives externally and supply them through the callbacks.

For PETSc residual problems, select a residual algorithm explicitly, for example `.with_tao_type(tao_algorithm_enum::BRGN)` or `.with_tao_type(tao_algorithm_enum::POUNDERS)`. The default TAO options select `LMVM`.

### RNC-LM curve derivatives

Set `problem.curve_derivatives` to a callback with this signature:

```cpp
void(const solverslib::vector_type& base,
     const std::vector<solverslib::vector_type>& coefficients,
     int order,
     solverslib::api::rnc_curve_derivatives& out);
```

It must fill `out.residual[0..order]` and `out.jacobian[0..max(0, order-2)]` with ordinary derivatives along the truncated curve. `coefficients[q-1]` is the unnormalized curve derivative `c_q`. An empty coefficient list at order 1 requests the residual and Jacobian at the base point.

Select the method with `solver_options_rnc_lm_builder().with_order(3).build()` and pass the resulting options to `solve`. Orders 1–4 are supported; the default is 3. The current public interface requires a caller-supplied curve callback. See `Testing/Cxx/TestRncLmSolver.cpp` for analytic callback examples.

## Results and current limits

`api::solver_result` includes the parameters, status, message, objective, iteration count, backend and algorithm fields, optional norms, optional work counters, and optional backend status code. `converged()` checks for `solver_status::converged`; `has_usable_iterate()` also accepts `max_iterations`, `stalled`, and `user_stopped`.

Missing diagnostics are represented by `std::nullopt`. Native LM, Gauss–Newton, and L-BFGS report their convergence reason; RNC-LM additionally fills evaluation and step counters. External adapters expose different subsets of diagnostics, and their algorithm field may remain `automatic`. Ceres currently maps its usable-solution boolean to the common status and does not preserve detailed termination or iteration diagnostics.

The current least-squares objective convention differs by solver: native LM and Gauss–Newton return `||r||²`; RNC-LM and the external adapters return `0.5 * ||r||²`. Scalar optimization returns the supplied objective value. Account for that factor when comparing results across backends.

Box bounds use `problem.bounds.lower` and `problem.bounds.upper`, each either empty or sized to `num_parameters`. Native LM and RNC-LM reject bounds. Native Gauss–Newton and L-BFGS currently do not enforce the bounds fields; use a compatible external algorithm for bounded problems. External bounds support depends on the selected algorithm.

General nonlinear constraints are not implemented: `optimization_problem.constraints` carries counts but no evaluation callbacks, and nonempty counts return `unsupported_capability`. There is no common evaluation-budget or parameter/residual-scaling option. NLopt maps `with_max_iterations(...)` to its function-evaluation limit.

## Benchmarks

Benchmarks use Google Benchmark and are separate from the GoogleTest suite. Enable `SOLVERS_ENABLE_BENCHMARKS` and make a system Google Benchmark installation discoverable:

```bash
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release \
    -DSOLVERS_ENABLE_BENCHMARKS=ON
cmake --build build-bench --parallel --target \
    BenchmarkNative BenchmarkExternal BenchmarkRootFinding

./build-bench/Testing/Benchmark/BenchmarkRootFinding
./build-bench/Testing/Benchmark/BenchmarkNative --benchmark_filter=SVI
./build-bench/Testing/Benchmark/BenchmarkNative --benchmark_list_tests=true
./build-bench/Testing/Benchmark/BenchmarkExternal
```

| Executable | Coverage |
| --- | --- |
| `BenchmarkRootFinding` | All seven scalar root methods and quadratic/cubic/quartic convenience solvers |
| `BenchmarkNative` | Rosenbrock and raw-SVI calibration with LM variants, Gauss–Newton, L-BFGS, and SVI RNC-LM orders 1–4 |
| `BenchmarkExternal` | Rosenbrock and raw-SVI calibration with Ceres, Ipopt, and PETSc BRGN/LMVM; unavailable backends are skipped |

Enable the corresponding backend flags in the benchmark build to run external cases. The current external benchmark executable does not register NLopt or POUNDERS cases. Google Benchmark options such as `--benchmark_filter`, `--benchmark_repetitions`, and `--benchmark_out` control selection, repetition, and export.

The setup helper can build and run all three executables with `python3 setup.py config.build.benchmark.release.static` from `Scripts/`.

## Source layout and documentation

- [`include/api/`](include/api/) — problem definitions, dispatch declarations, result and status types.
- [`include/solver_options/`](include/solver_options/) — solver-specific options and builders.
- [`include/solvers/`](include/solvers/) and [`src/solvers/`](src/solvers/) — native kernels, scalar roots, polynomial solvers, and optional adapters.
- [`src/api/dispatch.cpp`](src/api/dispatch.cpp) — current selection and callback requirements.
- [`Testing/Cxx/`](Testing/Cxx/) — four test executables: backend integration, roots, RNC-LM, and LM.
- [`Testing/Benchmark/`](Testing/Benchmark/) — benchmarks and the SVI fixture.
- [`docs/levenberg-marquardt.md`](docs/levenberg-marquardt.md) — native LM mathematics and damping policies.
- [`docs/rnc-lm.md`](docs/rnc-lm.md) — RNC-LM mathematics and implementation notes.
- [`svi_calib.md`](svi_calib.md) — SVI calibration background.

Some API examples and benchmark commands in the longer design notes predate the current callback/builder interface. Use the headers and the examples in this README for the current API.
