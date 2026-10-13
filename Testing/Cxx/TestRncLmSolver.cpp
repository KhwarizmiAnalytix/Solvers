#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "api/dispatch.h"
#include "solver_options/solver_options_rnc_lm.h"
#include "solvers/integrations/rnc_autodiff.h"

using namespace solverslib;
using namespace solverslib::api;

namespace
{
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Curve derivatives for r(x) = x - target  (linear, J = I constant).
// Higher-order corrections are always zero; all Jacobians are identity.
rnc_derivative_function linear_curve_deriv(const vector_type& target)
{
    return [target](const vector_type&         base,
               const std::vector<vector_type>& coeffs,
               int                             order,
               rnc_curve_derivatives&          out)
    {
        const auto n = static_cast<index_type>(base.size());
        out.residual.resize(order + 1);
        out.jacobian.resize(std::max(0, order - 2) + 1);

        out.residual[0] = base - target;
        out.residual[1] = coeffs.empty() ? vector_type::Zero(n) : coeffs[0];

        for (int k = 2; k <= order; ++k)
        {
            out.residual[k] = vector_type::Zero(n);
        }

        for (std::size_t j = 0; j < out.jacobian.size(); ++j)
        {
            if (j == 0)
            {
                out.jacobian[j] = matrix_type::Identity(n, n);
            }
            else
            {
                out.jacobian[j] = matrix_type::Zero(n, n);
            }
        }
    };
}

// A quadratic residual has nonzero higher derivatives along a curved path.
// Use the same factorial convention and truncated curve as the RNC recursion.
struct rosenbrock_model
{
    template <class T> void operator()(const T* x, T* r) const
    {
        r[0] = T(10) * (x[1] - x[0] * x[0]);
        r[1] = T(1) - x[0];
    }
};

rnc_derivative_function rosenbrock_curve_deriv()
{
    return make_rnc_derivatives(rosenbrock_model{}, 2, 2);
}

least_squares_problem make_linear_1d(double target_val)
{
    vector_type t(1);
    t << target_val;

    least_squares_problem p;
    p.num_parameters    = 1;
    p.num_residuals     = 1;
    p.residuals         = [t](const vector_type& x, vector_type& r) { r = x - t; };
    p.curve_derivatives = std::make_optional(linear_curve_deriv(t));
    return p;
}

std::shared_ptr<const solver_options_rnc_lm> tight_opts(int order = 1)
{
    return solver_options_rnc_lm_builder()
        .with_max_iterations(500)
        .with_function_tolerance(1e-12)
        .with_gradient_tolerance(1e-12)
        .with_parameter_tolerance(1e-12)
        .with_order(order)
        .build();
}
}  // namespace

TEST(RncLm, Solve)
{
    {
        SCOPED_TRACE("InvalidOptionsOrderZero");
        auto        p = make_linear_1d(2.0);
        vector_type x0(1);
        x0 << 3.0;
        const auto opts = solver_options_rnc_lm_builder().with_order(0).build();
        EXPECT_EQ(solve(p, x0, *opts).status, solver_status::invalid_problem);
    }
    {
        SCOPED_TRACE("InvalidOptionsOrderTooHigh");
        auto        p = make_linear_1d(2.0);
        vector_type x0(1);
        x0 << 3.0;
        const auto opts = solver_options_rnc_lm_builder().with_order(5).build();
        EXPECT_EQ(solve(p, x0, *opts).status, solver_status::invalid_problem);
    }
    {
        SCOPED_TRACE("InvalidOptionsAcceptanceThresholdAtOne");
        auto        p = make_linear_1d(2.0);
        vector_type x0(1);
        x0 << 3.0;
        const auto opts = solver_options_rnc_lm_builder().with_acceptance_threshold(1.0).build();
        EXPECT_EQ(solve(p, x0, *opts).status, solver_status::invalid_problem);
    }
    {
        SCOPED_TRACE("NoResidualsCallback");
        vector_type t(1);
        t << 2.0;
        least_squares_problem p;
        p.num_parameters    = 1;
        p.num_residuals     = 1;
        p.curve_derivatives = std::make_optional(linear_curve_deriv(t));

        vector_type x0(1);
        x0 << 3.0;
        const auto opts = solver_options_rnc_lm_builder().build();
        EXPECT_EQ(solve(p, x0, *opts).status, solver_status::invalid_problem);
    }
    {
        SCOPED_TRACE("WrongInitialGuessSize");
        auto        p = make_linear_1d(2.0);
        vector_type x0(3);
        x0 << 1.0, 2.0, 3.0;
        const auto opts = solver_options_rnc_lm_builder().build();
        EXPECT_EQ(solve(p, x0, *opts).status, solver_status::invalid_problem);
    }
    {
        SCOPED_TRACE("NaNInitialGuess");
        auto        p = make_linear_1d(2.0);
        vector_type x0(1);
        x0 << kNaN;
        const auto opts = solver_options_rnc_lm_builder().build();
        EXPECT_EQ(solve(p, x0, *opts).status, solver_status::invalid_problem);
    }
    {
        SCOPED_TRACE("BoundsNotSupported");
        auto p         = make_linear_1d(2.0);
        p.bounds.lower = {0.0};
        vector_type x0(1);
        x0 << 3.0;
        const auto opts = solver_options_rnc_lm_builder().build();
        EXPECT_EQ(solve(p, x0, *opts).status, solver_status::unsupported_capability);
    }
    {
        SCOPED_TRACE("BadBaseDerivativesWrongSize");
        least_squares_problem p;
        p.num_parameters    = 1;
        p.num_residuals     = 1;
        p.residuals         = [](const vector_type& x, vector_type& r) { r = x; };
        p.curve_derivatives = std::make_optional(
            [](const vector_type&, const std::vector<vector_type>&, int, rnc_curve_derivatives& out)
            {
                out.residual.clear();
                out.jacobian.clear();
            });

        vector_type x0(1);
        x0 << 3.0;
        const auto opts = solver_options_rnc_lm_builder().with_order(1).build();
        EXPECT_EQ(solve(p, x0, *opts).status, solver_status::numerical_failure);
    }
    {
        SCOPED_TRACE("BadBaseDerivativesNaN");
        least_squares_problem p;
        p.num_parameters    = 1;
        p.num_residuals     = 1;
        p.residuals         = [](const vector_type& x, vector_type& r) { r = x; };
        p.curve_derivatives = std::make_optional(
            [](const vector_type&,
                const std::vector<vector_type>&,
                int                    order,
                rnc_curve_derivatives& out)
            {
                out.residual.resize(order + 1);
                out.jacobian.resize(std::max(0, order - 2) + 1);
                out.residual[0] = vector_type::Constant(1, kNaN);
                out.residual[1] = vector_type::Zero(1);
                out.jacobian[0] = matrix_type::Identity(1, 1);
            });

        vector_type x0(1);
        x0 << 3.0;
        const auto opts = solver_options_rnc_lm_builder().with_order(1).build();
        EXPECT_EQ(solve(p, x0, *opts).status, solver_status::numerical_failure);
    }
    {
        SCOPED_TRACE("ConvergedAtStart");
        vector_type target(2);
        target << 3.0, 5.0;

        least_squares_problem p;
        p.num_parameters    = 2;
        p.num_residuals     = 2;
        p.residuals         = [&target](const vector_type& x, vector_type& r) { r = x - target; };
        p.curve_derivatives = std::make_optional(linear_curve_deriv(target));

        const auto opts =
            solver_options_rnc_lm_builder().with_function_tolerance(1e-6).with_order(1).build();
        const auto result = solve(p, target, *opts);

        EXPECT_EQ(result.status, solver_status::converged);
        EXPECT_EQ(result.iterations, 0u);
    }
    {
        SCOPED_TRACE("LinearOrder1");
        auto        p = make_linear_1d(2.0);
        vector_type x0(1);
        x0 << 5.0;

        const auto result = solve(p, x0, *tight_opts(1));

        EXPECT_EQ(result.status, solver_status::converged);
        EXPECT_NEAR(result.parameters[0], 2.0, 1e-9);
        EXPECT_EQ(result.backend, backend::native);
        EXPECT_EQ(result.algorithm, algorithm::riemann_normal_coordinate_lm);
        EXPECT_GT(*result.accepted_steps, 0u);
    }
    {
        SCOPED_TRACE("LinearOrders2To4");
        auto        p = make_linear_1d(2.0);
        vector_type x0(1);
        x0 << 5.0;
        EXPECT_EQ(solve(p, x0, *tight_opts(2)).status, solver_status::converged);
        EXPECT_EQ(solve(p, x0, *tight_opts(3)).status, solver_status::converged);
        EXPECT_EQ(solve(p, x0, *tight_opts(4)).status, solver_status::converged);
    }
    {
        SCOPED_TRACE("RosenbrockOrder3");
        least_squares_problem p;
        p.num_parameters = 2;
        p.num_residuals  = 2;
        p.residuals      = [](const vector_type& x, vector_type& r)
        {
            r.resize(2);
            r[0] = 10.0 * (x[1] - x[0] * x[0]);
            r[1] = 1.0 - x[0];
        };
        p.curve_derivatives = std::make_optional(rosenbrock_curve_deriv());

        vector_type x0(2);
        x0 << -1.2, 1.0;

        const auto opts   = solver_options_rnc_lm_builder()
                                .with_max_iterations(1000)
                                .with_function_tolerance(1e-10)
                                .with_gradient_tolerance(1e-10)
                                .with_order(3)
                                .with_verbose(true)
                                .build();
        const auto result = solve(p, x0, *opts);

        EXPECT_EQ(result.status, solver_status::converged);
        EXPECT_NEAR(result.parameters[0], 1.0, 1e-4);
        EXPECT_NEAR(result.parameters[1], 1.0, 1e-4);
        EXPECT_LT(*result.residual_norm, 1e-10);
    }
    {
        SCOPED_TRACE("RosenbrockOrder3AllOptionsSet");
        least_squares_problem p;
        p.num_parameters = 2;
        p.num_residuals  = 2;
        p.residuals      = [](const vector_type& x, vector_type& r)
        {
            r.resize(2);
            r[0] = 10.0 * (x[1] - x[0] * x[0]);
            r[1] = 1.0 - x[0];
        };
        p.curve_derivatives = std::make_optional(rosenbrock_curve_deriv());

        vector_type x0(2);
        x0 << -1.2, 1.0;

        const auto opts   = solver_options_rnc_lm_builder()
                                .with_max_iterations(1500)
                                .with_function_tolerance(1e-10)
                                .with_gradient_tolerance(1e-10)
                                .with_parameter_tolerance(1e-13)
                                .with_verbose(true)
                                .with_log_file("rnc_lm_all_options.log")
                                .with_order(3)
                                .with_max_curve_trials(6)
                                .with_acceptance_threshold(1e-3)
                                .with_contraction_min(0.2)
                                .with_contraction_max(0.6)
                                .with_initial_damping(1e-3)
                                .with_damping_floor(1e-14)
                                .with_damping_ceiling(1e11)
                                .with_diagonal_scaling_floor(1e-11)
                                .build();
        const auto result = solve(p, x0, *opts);

        EXPECT_EQ(result.status, solver_status::converged);
        EXPECT_NEAR(result.parameters[0], 1.0, 1e-4);
        EXPECT_NEAR(result.parameters[1], 1.0, 1e-4);
        EXPECT_EQ(result.algorithm, algorithm::riemann_normal_coordinate_lm);
    }
    {
        SCOPED_TRACE("ParameterToleranceStop");
        auto        p = make_linear_1d(2.0);
        vector_type x0(1);
        x0 << 2.5;

        const auto opts   = solver_options_rnc_lm_builder()
                                .with_max_iterations(200)
                                .with_function_tolerance(0.0)
                                .with_gradient_tolerance(0.0)
                                .with_parameter_tolerance(1.0)
                                .with_order(1)
                                .build();
        const auto result = solve(p, x0, *opts);
        EXPECT_EQ(result.status, solver_status::converged);
        EXPECT_NE(result.message.find("parameter tolerance"), std::string::npos);
    }
    {
        SCOPED_TRACE("MaxIterations");
        auto        p = make_linear_1d(2.0);
        vector_type x0(1);
        x0 << 100.0;

        const auto opts = solver_options_rnc_lm_builder()
                              .with_max_iterations(2)
                              .with_function_tolerance(0.0)
                              .with_gradient_tolerance(0.0)
                              .with_parameter_tolerance(0.0)
                              .with_order(1)
                              .build();
        EXPECT_EQ(solve(p, x0, *opts).status, solver_status::max_iterations);
    }
    {
        SCOPED_TRACE("ExhaustedDamping");
        vector_type x0(1);
        x0 << 3.0;
        vector_type fake_target = vector_type::Zero(1);

        least_squares_problem p;
        p.num_parameters = 1;
        p.num_residuals  = 1;
        p.residuals      = [](const vector_type&, vector_type& r)
        {
            r.resize(1);
            r[0] = kInf;
        };
        p.curve_derivatives = std::make_optional(linear_curve_deriv(fake_target));

        const auto opts   = solver_options_rnc_lm_builder()
                                .with_initial_damping(1e-3)
                                .with_damping_floor(1e-4)
                                .with_damping_ceiling(2e-3)
                                .with_max_curve_trials(2)
                                .with_max_iterations(100)
                                .with_order(1)
                                .with_function_tolerance(0.0)
                                .with_gradient_tolerance(0.0)
                                .build();
        const auto result = solve(p, x0, *opts);

        EXPECT_EQ(result.status, solver_status::numerical_failure);
        EXPECT_NE(result.message.find("exhausted damping"), std::string::npos);
    }
    {
        SCOPED_TRACE("CallbackThrows");
        least_squares_problem p;
        p.num_parameters = 1;
        p.num_residuals  = 1;
        p.residuals      = [](const vector_type&, vector_type& r)
        {
            r.resize(1);
            r[0] = 1.0;
        };
        p.curve_derivatives = std::make_optional(
            [](const vector_type&, const std::vector<vector_type>&, int, rnc_curve_derivatives&)
            { throw std::runtime_error("simulated callback error"); });

        vector_type x0(1);
        x0 << 3.0;
        const auto opts = solver_options_rnc_lm_builder().with_order(1).build();
        EXPECT_THROW(solve(p, x0, *opts), std::runtime_error);
    }
    {
        SCOPED_TRACE("InvalidDerivativesAfterAcceptedStep");
        vector_type target(1);
        target << 2.0;
        int call_count = 0;

        least_squares_problem p;
        p.num_parameters    = 1;
        p.num_residuals     = 1;
        p.residuals         = [&target](const vector_type& x, vector_type& r) { r = x - target; };
        p.curve_derivatives = std::make_optional(
            [&call_count, &target](const vector_type& base,
                const std::vector<vector_type>&,
                int                    order,
                rnc_curve_derivatives& out)
            {
                ++call_count;
                if (call_count == 1)
                {
                    out.residual.resize(order + 1);
                    out.jacobian.resize(std::max(0, order - 2) + 1);
                    out.residual[0] = base - target;
                    out.residual[1] = vector_type::Zero(1);
                    out.jacobian[0] = matrix_type::Identity(1, 1);
                }
                else
                {
                    out.residual.clear();
                    out.jacobian.clear();
                }
            });

        vector_type x0(1);
        x0 << 5.0;
        const auto opts   = solver_options_rnc_lm_builder()
                                .with_max_iterations(10)
                                .with_function_tolerance(0.0)
                                .with_gradient_tolerance(0.0)
                                .with_parameter_tolerance(0.0)
                                .with_order(1)
                                .build();
        const auto result = solve(p, x0, *opts);

        EXPECT_EQ(result.status, solver_status::numerical_failure);
        EXPECT_NE(result.message.find("accepted point"), std::string::npos);
    }
    {
        SCOPED_TRACE("HigherOrderDerivativesBadSize");
        vector_type target(1);
        target << 2.0;

        least_squares_problem p;
        p.num_parameters    = 1;
        p.num_residuals     = 1;
        p.residuals         = [&target](const vector_type& x, vector_type& r) { r = x - target; };
        p.curve_derivatives = std::make_optional(
            [&target](const vector_type& base,
                const std::vector<vector_type>&,
                int                    order,
                rnc_curve_derivatives& out)
            {
                if (order == 1)
                {
                    out.residual.resize(2);
                    out.jacobian.resize(1);
                    out.residual[0] = base - target;
                    out.residual[1] = vector_type::Zero(1);
                    out.jacobian[0] = matrix_type::Identity(1, 1);
                }
                else
                {
                    out.residual.clear();
                    out.jacobian.clear();
                }
            });

        vector_type x0(1);
        x0 << 5.0;
        const auto opts = solver_options_rnc_lm_builder()
                              .with_max_iterations(3)
                              .with_function_tolerance(0.0)
                              .with_gradient_tolerance(0.0)
                              .with_parameter_tolerance(0.0)
                              .with_order(2)
                              .build();
        EXPECT_EQ(solve(p, x0, *opts).status, solver_status::max_iterations);
    }
}
