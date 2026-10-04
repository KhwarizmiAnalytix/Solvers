#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "detail/support.h"
#include "optimization_test_problems.h"
#include "solvers/api/solve.h"
#include "solvers/ceres_solver.h"
#include "solvers/ipopt_solver.h"
#include "solvers/petsc_tao_solver.h"

namespace solverslib::api
{
namespace
{
const vector_type kStart = (vector_type(2) << -1.2, 1.0).finished();

least_squares_problem rosenbrock(bool flip_jacobian_sign = false)
{
    const auto            tp = testing::make_rosenbrock_problem();
    least_squares_problem problem;
    problem.num_parameters = tp.num_parameters;
    problem.num_residuals  = tp.num_residuals;
    problem.residuals      = tp.residuals;
    problem.set_jacobian(
        [j = tp.jacobian, flip_jacobian_sign](const vector_type& x, matrix_type& out)
        {
            j(x, out);
            if (flip_jacobian_sign)
            {
                out = -out;  // every step now points uphill
            }
        });
    return problem;
}

optimization_problem rosenbrock_objective()
{
    optimization_problem problem;
    problem.num_parameters = 2;
    problem.objective      = [](const vector_type& x)
    {
        const double a = 1.0 - x[0];
        const double b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    };
    problem.set_gradient(
        [](const vector_type& x, vector_type& g)
        {
            const double b = x[1] - x[0] * x[0];
            g.resize(2);
            g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * b;
            g[1] = 200.0 * b;
        });
    return problem;
}

// -- status vocabulary: converged / budget / stalled / failure, per backend ----------
TEST(Results, NativeLeastSquaresStatuses)
{
    for (const algorithm alg : {algorithm::levenberg_marquardt, algorithm::gauss_newton})
    {
        SCOPED_TRACE(to_string(alg));
        solve_options options;
        options.backend   = backend::native;
        options.algorithm = alg;

        EXPECT_EQ(solve(rosenbrock(), kStart, options).status, solver_status::converged);

        // Budget: one iteration is not enough; the iterate is still usable.
        options.max_iterations = 1;
        const auto budget      = solve(rosenbrock(), kStart, options);
        EXPECT_EQ(budget.status, solver_status::max_iterations);
        EXPECT_TRUE(budget.has_usable_iterate());

        // Stalled: with the Jacobian's sign flipped no step ever decreases the cost.
        options.max_iterations = 200;
        const auto stalled     = solve(rosenbrock(true), kStart, options);
        EXPECT_EQ(stalled.status, solver_status::stalled) << stalled.message;
        EXPECT_FALSE(stalled.converged());
        EXPECT_TRUE(stalled.has_usable_iterate());
        // Nothing was accepted, so the iterate is still the starting point.
        EXPECT_NEAR(stalled.parameters[0], kStart[0], 1e-12);

        // Failure: a non-finite residual at the start.
        least_squares_problem bad = rosenbrock();
        bad.residuals             = [](const vector_type&, vector_type& r)
        { r = vector_type::Constant(2, std::numeric_limits<double>::quiet_NaN()); };
        const auto failure = solve(bad, kStart, options);
        EXPECT_EQ(failure.status, solver_status::numerical_failure);
        EXPECT_FALSE(failure.has_usable_iterate());
    }
}

TEST(Results, NativeLbfgsStatuses)
{
    solve_options options;
    options.backend        = backend::native;
    options.algorithm      = algorithm::lbfgs;
    options.max_iterations = 500;

    EXPECT_EQ(solve(rosenbrock_objective(), kStart, options).status, solver_status::converged);

    options.max_iterations = 2;
    EXPECT_EQ(solve(rosenbrock_objective(), kStart, options).status, solver_status::max_iterations);

    options.max_iterations   = 500;
    optimization_problem bad = rosenbrock_objective();
    bad.objective = [](const vector_type&) { return std::numeric_limits<double>::quiet_NaN(); };
    const auto failure = solve(bad, kStart, options);
    EXPECT_FALSE(failure.converged());
    EXPECT_FALSE(failure.has_usable_iterate() && failure.status == solver_status::converged);
}

TEST(Results, CeresStatusesAndRawCode)
{
    if (!ceres_solver::is_supported())
    {
        GTEST_SKIP() << "Ceres not built";
    }
    solve_options options;
    options.backend = backend::ceres;

    const auto converged = solve(rosenbrock(), kStart, options);
    EXPECT_EQ(converged.status, solver_status::converged);
    EXPECT_TRUE(converged.backend_status.has_value());

    options.max_iterations = 1;
    const auto budget      = solve(rosenbrock(), kStart, options);
    EXPECT_EQ(budget.status, solver_status::max_iterations);
    EXPECT_TRUE(budget.has_usable_iterate());
    EXPECT_TRUE(budget.backend_status.has_value());
}

TEST(Results, IpoptStatuses)
{
    if (!ipopt_solver::is_supported())
    {
        GTEST_SKIP() << "Ipopt not built";
    }
    solve_options options;
    options.backend = backend::ipopt;

    const auto converged = solve(rosenbrock_objective(), kStart, options);
    EXPECT_EQ(converged.status, solver_status::converged) << converged.message;
    EXPECT_TRUE(converged.backend_status.has_value());

    // Hitting the iteration budget is not a numerical failure.
    options.max_iterations = 1;
    const auto budget      = solve(rosenbrock_objective(), kStart, options);
    EXPECT_EQ(budget.status, solver_status::max_iterations) << budget.message;
    EXPECT_TRUE(budget.has_usable_iterate());

    optimization_problem bad = rosenbrock_objective();
    bad.objective = [](const vector_type&) { return std::numeric_limits<double>::quiet_NaN(); };
    options.max_iterations = 100;
    const auto failure     = solve(bad, kStart, options);
    EXPECT_EQ(failure.status, solver_status::numerical_failure) << failure.message;
}

TEST(Results, PetscTaoStatuses)
{
    if (!petsc_tao_solver::is_supported())
    {
        GTEST_SKIP() << "PETSc/TAO not built";
    }
    solve_options options;
    options.backend = backend::petsc_tao;

    const auto converged = solve(rosenbrock_objective(), kStart, options);
    EXPECT_EQ(converged.status, solver_status::converged) << converged.message;
    EXPECT_TRUE(converged.backend_status.has_value());

    options.max_iterations = 1;
    const auto budget      = solve(rosenbrock_objective(), kStart, options);
    EXPECT_EQ(budget.status, solver_status::max_iterations) << budget.message;
    EXPECT_TRUE(budget.has_usable_iterate());
}

// -- every field is engaged or deliberately empty, per route ---------------------------
struct field_table
{
    bool residual, jacobian, objective, gradient, accepted, rejected, gradient_norm, step_norm;
};

void expect_fields(const solver_result& r, const field_table& expected)
{
    EXPECT_EQ(r.residual_evaluations.has_value(), expected.residual) << "residual_evaluations";
    EXPECT_EQ(r.jacobian_evaluations.has_value(), expected.jacobian) << "jacobian_evaluations";
    EXPECT_EQ(r.objective_evaluations.has_value(), expected.objective) << "objective_evaluations";
    EXPECT_EQ(r.gradient_evaluations.has_value(), expected.gradient) << "gradient_evaluations";
    EXPECT_EQ(r.accepted_steps.has_value(), expected.accepted) << "accepted_steps";
    EXPECT_EQ(r.rejected_steps.has_value(), expected.rejected) << "rejected_steps";
    EXPECT_EQ(r.gradient_norm.has_value(), expected.gradient_norm) << "gradient_norm";
    EXPECT_EQ(r.step_norm.has_value(), expected.step_norm) << "step_norm";
}

TEST(Results, FieldsAreEngagedOrDeliberatelyEmptyPerRoute)
{
    // residual jac obj grad accepted rejected |g| |step|
    const field_table native_ls       = {true, true, false, false, true, true, true, true};
    const field_table native_lbfgs_ls = {true, true, false, false, true, false, true, true};
    const field_table native_obj      = {false, false, true, true, true, false, true, true};
    const field_table ceres           = {true, true, false, false, true, true, true, true};
    const field_table external_obj    = {false, false, true, true, false, false, false, false};

    for (const algorithm alg : {algorithm::levenberg_marquardt, algorithm::gauss_newton})
    {
        solve_options options;
        options.backend   = backend::native;
        options.algorithm = alg;
        SCOPED_TRACE(to_string(alg));
        expect_fields(solve(rosenbrock(), kStart, options), native_ls);
    }
    {
        solve_options options;
        options.backend        = backend::native;
        options.algorithm      = algorithm::lbfgs;
        options.max_iterations = 500;
        SCOPED_TRACE("native lbfgs, least squares");
        expect_fields(solve(rosenbrock(), kStart, options), native_lbfgs_ls);
        SCOPED_TRACE("native lbfgs, objective");
        expect_fields(solve(rosenbrock_objective(), kStart, options), native_obj);
    }
    if (ceres_solver::is_supported())
    {
        solve_options options;
        options.backend = backend::ceres;
        SCOPED_TRACE("ceres");
        expect_fields(solve(rosenbrock(), kStart, options), ceres);
    }
    if (ipopt_solver::is_supported())
    {
        solve_options options;
        options.backend = backend::ipopt;
        SCOPED_TRACE("ipopt");
        expect_fields(solve(rosenbrock_objective(), kStart, options), external_obj);
    }
    if (petsc_tao_solver::is_supported())
    {
        solve_options options;
        options.backend = backend::petsc_tao;
        SCOPED_TRACE("petsc_tao objective");
        expect_fields(solve(rosenbrock_objective(), kStart, options), external_obj);
        // TAO's least-squares path does not expose its counters.
        SCOPED_TRACE("petsc_tao least squares");
        expect_fields(solve(rosenbrock(), kStart, options),
            {false, false, false, false, false, false, false, false});
    }
}

TEST(Results, StalledKeepsUsableIterateButIsNotConverged)
{
    solver_result stalled;
    stalled.status = solver_status::stalled;
    EXPECT_FALSE(stalled.converged());
    EXPECT_TRUE(stalled.has_usable_iterate());
    EXPECT_STREQ(to_string(solver_status::stalled), "stalled");
}
}  // namespace
}  // namespace solverslib::api
