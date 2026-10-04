// Verifies that the native Levenberg-Marquardt iteration loop performs no
// linear-algebra heap allocation. Eigen's runtime malloc guard (EIGEN_RUNTIME_NO_MALLOC) is a
// per-program setting, so this executable compiles the kernel sources itself
// rather than linking the Solvers library (see Testing/Cxx/CMakeLists.txt).
#include <gtest/gtest.h>

#include <cmath>
#include <functional>

#include "detail/native_evaluation.h"
#include "solver_options/solver_options_lm.h"
#include "solvers/api/detail/evaluators.h"
#include "solvers/levenberg_marquardt_solver.h"

namespace solverslib
{
namespace
{
// Rosenbrock in residual form with an analytic Jacobian.
void residuals(const vector_type& x, vector_type& r)
{
    r[0] = 10.0 * (x[1] - x[0] * x[0]);
    r[1] = 1.0 - x[0];
}

void jacobian(const vector_type& x, matrix_type& j)
{
    j(0, 0) = -20.0 * x[0];
    j(0, 1) = 10.0;
    j(1, 0) = -1.0;
    j(1, 1) = 0.0;
}

// A larger, mildly nonlinear fit so the loop runs several accepted and rejected steps.
constexpr int kParameters = 6;
constexpr int kResiduals  = 40;

void wide_residuals(const vector_type& x, vector_type& r)
{
    for (int i = 0; i < kResiduals; ++i)
    {
        const double t     = static_cast<double>(i) / kResiduals;
        double       model = 0.0;
        for (int j = 0; j < kParameters; ++j)
        {
            model += x[j] * std::cos((j + 1) * t);
        }
        r[i] = model + 0.1 * std::sin(3.0 * model) - std::exp(-t);
    }
}

void wide_jacobian(const vector_type& x, matrix_type& jac)
{
    for (int i = 0; i < kResiduals; ++i)
    {
        const double t     = static_cast<double>(i) / kResiduals;
        double       model = 0.0;
        for (int j = 0; j < kParameters; ++j)
        {
            model += x[j] * std::cos((j + 1) * t);
        }
        const double slope = 1.0 + 0.3 * std::cos(3.0 * model);
        for (int j = 0; j < kParameters; ++j)
        {
            jac(i, j) = slope * std::cos((j + 1) * t);
        }
    }
}

// Runs one solve with Eigen allocation forbidden inside the iteration loop.
// The kernel allocates its workspace up front, so the guard is armed from the
// first residual evaluation (the first callback the kernel makes) until the
// kernel returns; the final result vector copies happen outside.
template <class Residuals, class Jacobian>
native_result guarded_solve(size_t n,
    size_t                         m,
    Residuals                      res,
    Jacobian                       jac,
    vector_type&                   x,
    const solver_options_lm&       options,
    int&                           calls_after_arming)
{
    bool armed             = false;
    auto wrapped_residuals = [&](const vector_type& p, vector_type& r)
    {
        if (!armed)
        {
            // First evaluation: the workspace has been allocated by now.
            armed = true;
            set_allocation_allowed(false);
        }
        ++calls_after_arming;
        res(p, r);
    };
    levenberg_marquardt_solver solver(n, m, wrapped_residuals, jac);
    native_result              result = solver.solve(x, options);
    set_allocation_allowed(true);
    return result;
}

// The harness itself: with the guard armed, an Eigen allocation must throw.
TEST(NoAllocHarness, GuardRejectsAnAllocation)
{
    set_allocation_allowed(false);
    bool threw = false;
    try
    {
        vector_type v(64);  // heap allocation
        v.setZero();
    }
    catch (const std::runtime_error&)
    {
        threw = true;
    }
    set_allocation_allowed(true);
    EXPECT_TRUE(threw) << "the malloc guard is not active; the checks below would prove nothing";
}

class LmNoAlloc : public ::testing::TestWithParam<levenberg_marquardt_solver_enum>
{
};

TEST_P(LmNoAlloc, IterationLoopDoesNotAllocate)
{
    auto options = solver_options_lm_builder()
                       .with_type(GetParam())
                       .with_max_iterations(200)
                       .with_geodesic_acceleration(true)
                       .build();
    vector_type x(kParameters);
    x.setConstant(0.3);
    int        calls = 0;
    const auto result =
        guarded_solve(kParameters, kResiduals, wide_residuals, wide_jacobian, x, *options, calls);
    EXPECT_GT(calls, 3) << "the loop must actually iterate";
    EXPECT_GT(result.iterations, 2u);
    // An allocation inside the guarded region surfaces as a failure status.
    EXPECT_NE(result.status, native_convergence::numerical_failure) << result.message;
}

TEST_P(LmNoAlloc, RosenbrockWithoutGeodesicAcceleration)
{
    auto options = solver_options_lm_builder()
                       .with_type(GetParam())
                       .with_max_iterations(200)
                       .with_geodesic_acceleration(false)
                       .build();
    vector_type x(2);
    x << -1.2, 1.0;
    int        calls  = 0;
    const auto result = guarded_solve(2, 2, residuals, jacobian, x, *options, calls);
    EXPECT_TRUE(result.converged()) << result.message;
    EXPECT_GT(calls, 3);
}

TEST_P(LmNoAlloc, FiniteDifferenceJacobianDoesNotAllocate)
{
    auto options =
        solver_options_lm_builder().with_type(GetParam()).with_max_iterations(200).build();
    vector_type x(kParameters);
    x.setConstant(0.3);
    // No analytic Jacobian: the shared finite-difference evaluator is used. Its
    // stencil scratch is sized on first use, so the guard is armed once the first
    // Jacobian (one residual call plus a 2n central stencil) has completed.
    int  calls   = 0;
    auto wrapped = [&](const vector_type& p, vector_type& r)
    {
        ++calls;
        if (calls == 1 + 2 * kParameters + 1)
        {
            set_allocation_allowed(false);
        }
        wide_residuals(p, r);
    };
    levenberg_marquardt_solver solver(kParameters, kResiduals, wrapped);
    const auto                 result = solver.solve(x, *options);
    set_allocation_allowed(true);
    EXPECT_GT(calls, 10);
    EXPECT_GT(result.iterations, 2u);
    EXPECT_NE(result.status, native_convergence::numerical_failure) << result.message;
}

INSTANTIATE_TEST_SUITE_P(Variants,
    LmNoAlloc,
    ::testing::Values(levenberg_marquardt_solver_enum::LEVENBERG_MARQUARDT,
        levenberg_marquardt_solver_enum::QUADRATIC_INTERPOLATION,
        levenberg_marquardt_solver_enum::NIELSEN));
}  // namespace
}  // namespace solverslib
