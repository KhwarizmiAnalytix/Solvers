# Solvers

[![CI](https://github.com/KhwarizmiAnalytix/Solvers/actions/workflows/ci.yml/badge.svg)](https://github.com/KhwarizmiAnalytix/Solvers/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/KhwarizmiAnalytix/Solvers/branch/master/graph/badge.svg)](https://codecov.io/gh/KhwarizmiAnalytix/Solvers)

**C++ numerical solvers for root finding, nonlinear least squares, and scalar-objective optimization.**

Solvers combines native Eigen-based algorithms with optional Ceres, Ipopt, NLopt, and PETSc/TAO backends. Least-squares and optimization problems share a callback-based interface and a structured result; solver-specific builders make the algorithm and its settings explicit.

- **Native algorithms:** Levenberg–Marquardt, Gauss–Newton with backtracking, L-BFGS, and Riemann-normal-coordinate LM (RNC-LM), orders 1–4.
- **Root finding:** seven scalar methods and single-root polynomial convenience functions.
- **Derivative control:** supply analytic callbacks or use finite differences where supported.
- **Build integration:** CMake targets, optional backend flags, GoogleTest tests, and Google Benchmark comparisons; Bazel supports the native library.

## Contents

- [Quick start](#quick-start)
- [CMake integration](#cmake-integration)
- [Usage](#usage)
- [Solver selection and derivatives](#solver-selection-and-derivatives)
- [Results and production considerations](#results-and-production-considerations)
- [Build configuration](#build-configuration)
- [Benchmarks](#benchmarks)
- [Development](#development)
- [Documentation and source layout](#documentation-and-source-layout)

## Quick start

### Requirements

| Component | Requirement |
| --- | --- |
| Compiler | C++20-capable compiler for the default dependency build; Solvers itself declares C++17 |
| CMake | 3.20 or newer for native builds; 3.22 or newer when building bundled Ceres |
| Git | Recursive submodules initialized, including Logging's nested dependencies |
| Required dependencies | Eigen and Logging, bundled under `ThirdParty/` |
| Tests | Bundled GoogleTest, enabled by default for a standalone CMake build |
| Optional tooling | Python 3.9 or newer for setup helpers; Python is not required for direct CMake builds |

The Python files provide build tooling; this repository does not provide Python bindings. Bundled Logging defaults to C++20 for its own compilation.

### Build and test

```bash
git clone --recurse-submodules https://github.com/KhwarizmiAnalytix/Solvers.git
cd Solvers

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

For an existing checkout, initialize dependencies with `git submodule update --init --recursive` before configuring. The default build uses static libraries, enables tests, and disables optional backends and benchmarks.

With a multi-configuration generator such as Visual Studio or Xcode, select the configuration explicitly:

```bash
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

## CMake integration

### Integrate from source

Use an initialized source checkout and link the exported build target:

```cmake
cmake_minimum_required(VERSION 3.20)
project(MyApplication LANGUAGES C CXX)

add_subdirectory(path/to/Solvers)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE Solvers::Solvers)
```

Solver headers are available through the target's include directories. Tests default to disabled when Solvers is included as a subproject. Set backend flags before `add_subdirectory` to enable the adapters your application needs.

### Install a package

```bash
cmake --install build --prefix "$PWD/install"
```

Configure a consumer with `-DCMAKE_PREFIX_PATH=/path/to/Solvers/install`, then use:

```cmake
find_package(Solvers CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE Solvers::Solvers)
```

Installation includes solver headers, the library, Eigen headers, and Logging's package files.

Additional requirements for installed packages with optional backends:

- **Ceres:** provide a separately installed, compatible Ceres package; Solvers does not install its bundled Ceres dependency.
- **NLopt:** the package configuration does not discover NLopt. For a static installation, create `PkgConfig::NLOPT` before finding Solvers:

  ```cmake
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(NLOPT REQUIRED IMPORTED_TARGET nlopt)
  find_package(Solvers CONFIG REQUIRED)
  ```

- **PETSc/TAO:** enable the C language in the consumer so MPI's C target can be discovered.

## Usage

Problem and result types live in `solverslib::api`; option builders and numerical types live in `solverslib`. Each example below is a complete `main.cpp` and can be linked using the source integration above.

### Nonlinear least squares

Define the residual vector and, optionally, its Jacobian. This example solves `x² - 4 = 0` from an initial guess of `3` using native Levenberg–Marquardt:

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
    if (!result.converged()) {
        std::cerr << "Solve failed: " << result.message << '\n';
        return 1;
    }
    std::cout << result.parameters[0] << '\n';
    return 0;
}
```

The builder returns `std::shared_ptr<const solver_options_lm>`; pass `*options` to `solve`. Omit `problem.jacobian` to use finite differences. The residual callback must fill a vector of length `num_residuals`; the Jacobian must have `num_residuals` rows and `num_parameters` columns.

Native LM provides Nielsen, Marquardt, and quadratic-interpolation damping, optional geodesic acceleration and bold acceptance, and `NORMAL_LDLT` or `AUGMENTED_QR` linear solves. Configure these through [`solver_options_lm_builder`](include/solver_options/solver_options_lm.h).

### Scalar-objective optimization

Define a scalar objective and an optional gradient. Native L-BFGS approximates a missing gradient with finite differences:

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
    if (!result.converged()) {
        std::cerr << "Solve failed: " << result.message << '\n';
        return 1;
    }
    std::cout << result.parameters[0] << '\n';
    return 0;
}
```

The gradient callback must fill a vector of length `num_parameters`. The [L-BFGS builder](include/solver_options/solver_options_bfgs.h) configures history size, line-search strategy, step limits, and finite-difference scaling.

### Scalar root finding

The root interface returns `detail::root_run` with `root`, `iterations`, and an `outcome` of `converged`, `iteration_limit`, or `degenerate`:

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

The available methods are bisection, false position, Ridders, Brent, Dekker, Newton–Raphson, and secant. Bracketing methods require a valid sign-changing bracket. Newton–Raphson and Dekker take a callback with signature `double(double x, double& derivative)`. Invalid brackets and vanishing derivatives or secant denominators can throw.

[`polynomial_solver`](include/solvers/polynomial_solver.h) provides convenience functions for monic quadratic, cubic, and quartic polynomials. Each returns one selected real root. The quartic returns the largest real root clamped from below by its `threshold` argument; a clamped value may no longer satisfy the polynomial.

## Solver selection and derivatives

Select a solver by constructing its options and passing them to `api::solve(problem, initial_guess, *options)`. Dispatch uses `options.solver()` and the concrete options type; it does not automatically choose a backend.

| Solver / backend | Problem type | Options builder | Derivative contract |
| --- | --- | --- | --- |
| Native LM | Least squares | `solver_options_lm_builder` | Optional Jacobian; finite differences when absent |
| Native Gauss–Newton | Least squares | `solver_options_gn_builder` | Optional Jacobian; finite differences when absent |
| Native RNC-LM | Least squares | `solver_options_rnc_lm_builder` | Requires curve derivatives; no numerical fallback |
| Native L-BFGS | Scalar objective | `solver_options_bfgs_builder` | Optional gradient; finite differences when absent |
| Ceres | Least squares | `solver_options_ceres_builder` | Optional Jacobian; numerical differentiation when absent |
| Ipopt | Least squares or scalar objective | `solver_options_ipopt_builder` | Requires Jacobian for least squares or gradient for scalar objectives; optional Hessian for scalar objectives |
| NLopt | Least squares or scalar objective | `solver_options_nlopt_builder` | Least-squares Jacobian supplies the scalar-cost gradient; omit it only for derivative-free algorithms. Dispatch requires a gradient for scalar objectives |
| PETSc/TAO | Least squares or scalar objective | `solver_options_petsc_builder` | Residual callback and optional Jacobian for least squares; dispatch requires a gradient for scalar objectives, with an optional Hessian |

NLopt supports L-BFGS, MMA, SLSQP, CCSAQ, Nelder–Mead, and Subplex. TAO options include POUNDERS, BRGN, NLS, NTR, NTL, LMVM, BQNLS, and BNLS. Select a TAO algorithm compatible with the problem's callbacks: for residual problems, explicitly choose `tao_algorithm_enum::BRGN` or `tao_algorithm_enum::POUNDERS`. The default TAO algorithm is `LMVM`.

The public problems store callbacks. Enabling Ceres builds its optimizer adapter; it does not automatically differentiate a templated residual model. Compute derivatives in your application and supply them through the callbacks when needed.

### RNC-LM curve derivatives

Set `problem.curve_derivatives` to a callback with this signature:

```cpp
void(const solverslib::vector_type& base,
     const std::vector<solverslib::vector_type>& coefficients,
     int order,
     solverslib::api::rnc_curve_derivatives& out);
```

Fill `out.residual[0..order]` and `out.jacobian[0..max(0, order-2)]` with ordinary derivatives along the truncated curve. `coefficients[q-1]` is the unnormalized curve derivative `c_q`. An empty coefficient list at order 1 requests the residual and Jacobian at the base point.

Select the method with `solver_options_rnc_lm_builder().with_order(3).build()`. Orders 1–4 are supported; the default is 3. RNC-LM defaults to absolute damping `D = I` with `lambda_0 = 1e-3`; `with_damping_scaling(rnc_lm_damping_scaling_enum::MARQUARDT)` selects diagonal-scaled damping. The [Taylor adapter](include/solvers/integrations/rnc_autodiff.h) supplies these callbacks from a scalar-templated model through `rnc_least_squares(model, num_parameters, num_residuals)`. It includes factorial conversions and higher derivatives along the complete curve, without a Ceres dependency. See [RNC-LM usage and paper comparison](docs/rnc-lm.md) for a complete example.

## Results and production considerations

### Termination and diagnostics

[`api::solver_result`](include/api/result.h) contains the returned parameters, status, message, objective, iteration count, backend and algorithm identifiers, optional norms and work counters, and an optional backend status code. `converged()` checks whether `status == solver_status::converged`.

| Status | Interpretation |
| --- | --- |
| `converged` | The solver's stopping criterion was satisfied |
| `max_iterations` | The iteration or evaluation limit was reached |
| `stalled` | No acceptable step was found |
| `invalid_problem` | Problem dimensions, callbacks, or options were rejected |
| `numerical_failure` | Numerical evaluation or a linear solve failed |
| `infeasible` | The supplied point or constraint set is infeasible |
| `unsupported_capability` | The selected solver cannot honor a requested capability |
| `backend_unavailable` | The requested optional backend was not compiled in |
| `user_stopped` | The run was stopped by a caller callback |

A limit or stalled run may leave useful parameters, but does not establish convergence. Inspect the status and message, then evaluate the returned point against your application's acceptance criteria. Set tolerances and budgets explicitly for the scale of your problem.

Optional diagnostics use `std::nullopt` when unavailable. Native LM, Gauss–Newton, and L-BFGS report the convergence reason; RNC-LM also reports evaluation and step counters. External adapters expose different subsets of diagnostics and may leave the algorithm identifier as `automatic`. Ceres maps its usable-solution boolean to the common status without preserving detailed termination diagnostics. Ceres and NLopt do not populate `iterations`, so their default value of zero does not represent a measured iteration count.

Native Gauss–Newton's reported gradient norm belongs to the iterate before the final accepted step. Recompute the gradient at the returned parameters if your acceptance policy requires its final norm.

### Objective conventions

The returned least-squares objective differs by implementation:

| Implementation | `result.objective` |
| --- | --- |
| Native LM and Gauss–Newton | `||r(x)||²` |
| RNC-LM and external least-squares adapters | `0.5 * ||r(x)||²` |
| Scalar-objective optimization | The supplied `objective(x)` |

Normalize this factor, or recompute a common objective at the returned parameters, when comparing backends.

### Bounds and constraints

Box bounds use `problem.bounds.lower` and `problem.bounds.upper`. Each vector must be empty or have length `num_parameters`.

| Solver | Bounds behavior |
| --- | --- |
| Native LM and RNC-LM | Reject bounds |
| Native Gauss–Newton and L-BFGS | Do not enforce the bounds fields |
| External backends | Support depends on the selected algorithm |

Use a compatible external algorithm when bounds are required. General nonlinear constraints are not implemented: `optimization_problem.constraints` stores counts without evaluation callbacks, and nonempty counts return `unsupported_capability`.

There is no common evaluation-budget or parameter/residual-scaling option. NLopt maps `with_max_iterations(...)` to a function-evaluation limit.

## Build configuration

### Optional backends

Backends are disabled by default and enabled independently. For example, to build Ceres support:

```bash
cmake -S . -B build-ceres -DCMAKE_BUILD_TYPE=Release \
    -DSOLVERS_ENABLE_CERES=ON
cmake --build build-ceres --parallel
ctest --test-dir build-ceres --output-on-failure
```

| Option | Default | Dependency / behavior |
| --- | --- | --- |
| `SOLVERS_ENABLE_CERES` | `OFF` | Builds `ThirdParty/ceres` unless the parent supplies `Ceres::ceres`; bundled Ceres requires CMake 3.22+ |
| `SOLVERS_ENABLE_IPOPT` | `OFF` | System library discovered through pkg-config module `ipopt` |
| `SOLVERS_ENABLE_NLOPT` | `OFF` | System library discovered through pkg-config module `nlopt` |
| `SOLVERS_ENABLE_PETSC` | `OFF` | System library discovered through pkg-config module `PETSc`, plus MPI's C component |
| `SOLVERS_ENABLE_TESTING` | Top-level: `ON`; subproject: `OFF` | Builds the GoogleTest suite |
| `SOLVERS_ENABLE_BENCHMARKS` | `OFF` | Requires Google Benchmark through pkg-config or `find_package(benchmark)` |
| `BUILD_SHARED_LIBS` | `OFF` | Requests shared libraries; building bundled Ceres forces static libraries |
| `SOLVERS_ENABLE_SANITIZER` | `OFF` | Address and undefined behavior sanitizers with GCC/Clang |
| `SOLVERS_ENABLE_COVERAGE` | `OFF` | Coverage instrumentation with GCC/Clang |
| `SOLVERS_ENABLE_CLANGTIDY` | `OFF` | clang-tidy analysis for Solvers |
| `SOLVERS_ENABLE_IWYU` | `OFF` | Include-what-you-use analysis for Solvers |

Use `PKG_CONFIG_PATH` for system backend installations outside pkg-config's search path. A request for a disabled backend returns `solver_status::backend_unavailable`. POUNDERS is enabled through PETSc/TAO; it has no separate build flag.

### Bazel

The repository pins Bazel 8.0.0 in `.bazelversion`. Its build graph supports the native library, unit tests, and Google Benchmark executables. Optional backend macros are set to zero in `BUILD.bazel`; enable those backends through CMake. `ConvergenceDemo` is currently a CMake-only target.

```bash
bazel build //:Solvers
bazel test //Testing/Cxx/...
bazel test //Testing/Benchmark/...
```

To package headers and library artifacts:

```bash
bazel build //:solvers_pkg_tar
bazel run //:install -- --destdir=/path/to/staging
```

These targets do not generate the CMake package or bundle its dependency installation tree.

## Benchmarks

The benchmark suite covers scalar roots, Rosenbrock optimization, and raw-SVI calibration. Google Benchmark executables provide repeated timing measurements; `ConvergenceDemo` prints diagnostics from one solve per configured case.

```bash
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release \
    -DSOLVERS_ENABLE_BENCHMARKS=ON
cmake --build build-bench --parallel --target \
    BenchmarkNative BenchmarkExternal BenchmarkRootFinding ConvergenceDemo

./build-bench/Testing/Benchmark/BenchmarkRootFinding
./build-bench/Testing/Benchmark/BenchmarkNative --benchmark_filter=SVI
./build-bench/Testing/Benchmark/BenchmarkExternal
./build-bench/Testing/Benchmark/ConvergenceDemo
```

| Executable | Coverage |
| --- | --- |
| `BenchmarkRootFinding` | Seven scalar root methods and quadratic/cubic/quartic convenience solvers |
| `BenchmarkNative` | Rosenbrock and SVI with LM variants, Gauss–Newton, L-BFGS, and SVI RNC-LM orders 1–4 |
| `BenchmarkExternal` | Rosenbrock and SVI with Ceres, Ipopt, and PETSc BRGN/LMVM; unavailable backends are skipped |
| `ConvergenceDemo` | Rosenbrock diagnostics for native LM, Gauss–Newton, L-BFGS, and enabled Ceres, Ipopt, NLopt, and PETSc BRGN/LMVM cases |

Enable the corresponding backend flags in `build-bench` to include external cases. Google Benchmark is required when configuring this directory, including when building only `ConvergenceDemo`. The external timing executable does not register NLopt or POUNDERS cases.

Export repeated measurements for a selected case:

```bash
./build-bench/Testing/Benchmark/BenchmarkNative \
    --benchmark_filter=SVI \
    --benchmark_repetitions=5 \
    --benchmark_report_aggregates_only=true \
    --benchmark_out=benchmark-results.json \
    --benchmark_out_format=json
```

Record the commit, compiler, build type, backend versions, and hardware alongside exported results. With multi-configuration generators, select `Release` when building and run the executable from the corresponding configuration directory.

### Rosenbrock convergence comparison

To enable all optional cases in the diagnostics demo, make the system dependencies discoverable and run:

```bash
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release \
    -DSOLVERS_ENABLE_BENCHMARKS=ON -DSOLVERS_ENABLE_CERES=ON \
    -DSOLVERS_ENABLE_IPOPT=ON -DSOLVERS_ENABLE_NLOPT=ON -DSOLVERS_ENABLE_PETSC=ON
cmake --build build-bench --parallel --target ConvergenceDemo
./build-bench/Testing/Benchmark/ConvergenceDemo
```

<details>
<summary>Example results from a single macOS/arm64 Release run</summary>

The measurements below used Clang, bundled Ceres, and Homebrew Ipopt/NLopt/PETSc, with `x0 = (-1.2, 1.0)`. The minimum is `(1, 1)` with objective zero. These unreplicated timings illustrate diagnostics for this problem and machine; use repeated benchmarks to compare performance. The least-squares and general-objective forms are solved independently below, each through its own backends, rather than merged into one table — a backend's `||r(x)||` and `||grad||` come from different problem formulations and are not directly comparable.

**Least-squares form**, `r(x) = [10(x2-x1²), 1-x1]`:

| Backend | status | iterations | `\|\|r(x)\|\|` | `f(x)` | time |
| --- | --- | --- | --- | --- | --- |
| `native/levenberg_marquardt` | converged | 32 | 1.62e-13 | 2.62e-26 | 99 µs |
| `native/gauss_newton` | converged | 10 | 3.66e-15 | 1.34e-29 | 5 µs |
| `ceres` | converged | n/a¹ | n/a² | 0.00e+00 | 616 µs |
| `ipopt` | converged | 47 | n/a² | 1.55e-19 | 28.9 ms |
| `petsc_tao/BRGN` | converged | 18 | n/a² | 2.34e-16 | 52.9 ms |

**General-objective form**, `min f(x)`, `f(x) = 0.5((x2-x1²)² + (1-x1)²)`:

| Backend / algorithm | status | iterations | `\|\|grad\|\|` | `f(x)` | time |
| --- | --- | --- | --- | --- | --- |
| `native/lbfgs` | converged | 86 | 9.48e-08 | 8.50e-15 | 37 µs |
| `ipopt` | converged | 47 | n/a² | 1.55e-19 | 21.9 ms |
| `nlopt/LD_LBFGS` | converged | n/a¹ | n/a² | 9.65e-19 | 753 µs |
| `petsc_tao/LMVM` | converged | 37 | n/a² | 4.56e-23 | 628 µs |

¹ Ceres and NLopt do not populate `solver_result::iterations`; Ipopt and PETSc/TAO do.

² External adapters recompute the objective at the returned parameters without exposing the library's internal residual or gradient norm.

Objective conventions and stopping tolerances differ across the cases; see [results and production considerations](#results-and-production-considerations).

</details>

## Development

Run the unit suite with `ctest --test-dir build --output-on-failure`. To check a Debug build with address and undefined behavior sanitizers:

```bash
cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug \
    -DSOLVERS_ENABLE_TESTING=ON -DSOLVERS_ENABLE_SANITIZER=ON
cmake --build build-sanitize --parallel
ctest --test-dir build-sanitize --output-on-failure
```

The [CI configuration](.github/workflows/ci.yml) defines native build jobs for Ubuntu, macOS, and Windows, plus selected optional backends and analysis configurations. Test the backend configuration your application uses when changing solver or adapter behavior.

The setup helpers accept dotted commands and must run from `Scripts/`:

```bash
cd Scripts
python3 setup.py --help
python3 setup.py config.build.test.release.static
python3 setup.py config.build.benchmark.release.static

python3 setup_bazel.py build.test.release
python3 setup_bazel.py build.benchmark.release
```

The CMake helper's benchmark phase runs the three Google Benchmark executables; run `ConvergenceDemo` directly for convergence diagnostics.

## Documentation and source layout

| Path | Contents |
| --- | --- |
| [`include/api/`](include/api/) | Problem definitions, dispatch declarations, result and status types |
| [`include/solver_options/`](include/solver_options/) | Solver options and builders |
| [`include/solvers/`](include/solvers/) and [`src/solvers/`](src/solvers/) | Native kernels, root methods, polynomial solvers, and optional adapters |
| [`src/api/dispatch.cpp`](src/api/dispatch.cpp) | Solver dispatch and callback requirements |
| [`Testing/Cxx/`](Testing/Cxx/) | Backend integration, roots, RNC-LM, and LM tests |
| [`Testing/Benchmark/`](Testing/Benchmark/) | Timing benchmarks, convergence demo, and SVI fixture |
| [`docs/levenberg-marquardt.md`](docs/levenberg-marquardt.md) | Native LM mathematics and damping policies |
| [`docs/rnc-lm.md`](docs/rnc-lm.md) | RNC-LM mathematics and implementation notes |
| [`docs/lm-literature-problem-list.md`](docs/lm-literature-problem-list.md) | Literature problem inventory, current coverage gaps, and prioritized additions |
| [`docs/svi_calib.md`](docs/svi_calib.md) | SVI calibration background |
| [`CHANGELOG.md`](CHANGELOG.md) | Change history |

Some examples in the design and mathematical notes predate the current callback/builder interface. Use the public headers, unit tests, and complete examples in this README for the current API.

### Reproducing CI with the setup helper

CI and local builds use the documented dotted actions and named options from
`python Scripts/setup.py --help`:

```sh
python Scripts/setup.py config.build.test.release.ninja.clang --build-dir build
python Scripts/setup.py install.release --build-dir build --install-prefix install
python Scripts/setup.py config.build.test --build-dir build-debug --compiler clang-cl --build-type Debug --testing on --benchmark off
```

Only declared options are accepted. Tool commands, raw build-system arguments,
and a `--` argument separator are rejected. Paths, backend selections, GPU
architectures, test filters, and installation settings have named options in
the usage text. `--source-dir` selects a dependency or installed-package consumer
project. `verifycache --build-dir build --target TARGET` checks the launcher and
clean-rebuild cache hits. `verifytoolchain` checks the compiler selected by Ninja.
Run Windows Ninja builds in a Visual Studio developer environment.

Bazel CI uses the same action style and the options listed by
`python Scripts/setup_bazel.py --help`, for example:

```sh
python Scripts/setup_bazel.py build.test --target //Testing/... --compiler default --test-output errors
```
