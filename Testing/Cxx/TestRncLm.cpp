#include "solvers/api/solve.h"
#include "solvers/integrations/rnc_autodiff.h"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

using namespace solverslib;
namespace
{
struct Quadratic
{
    template <class T> bool operator()(const T* x, T* r) const
    {
        r[0] = x[0] * x[0] - T(1);
        return true;
    }
};
struct EmbeddedParabola
{
    template <class T> void operator()(const T* x, T* r) const
    {
        r[0] = x[0];
        r[1] = x[0] * x[0];
    }
};
struct Exponential
{
    template <class T> void operator()(const T* x, T* r) const
    {
        using std::exp;
        r[0] = exp(x[0]);
    }
};
struct ElementaryFunctions
{
    template <class T> void operator()(const T* x, T* r) const
    {
        using std::cos;
        using std::log;
        using std::pow;
        using std::sin;
        using std::sqrt;
        r[0] = sin(x[0]);
        r[1] = cos(x[0]);
        r[2] = log(x[0]);
        r[3] = sqrt(x[0]);
        r[4] = pow(x[0], -2);
        r[5] = pow(x[0], .5);
    }
};
struct GeneralizedRosenbrock
{
    int                     power;
    double                  stiffness;
    template <class T> bool operator()(const T* x, T* r) const
    {
        using std::pow;
        r[0] = x[0];
        r[1] = T(stiffness) * (x[1] - pow(x[0], power) / T(power));
        return true;
    }
};
struct RankDeficient
{
    template <class T> bool operator()(const T* x, T* r) const
    {
        r[0] = x[0] + x[1] - T(3.);
        return true;
    }
};
api::solve_options configuration(int order)
{
    api::solve_options options;
    options.algorithm           = api::algorithm::riemann_normal_coordinate_lm;
    options.rnc_lm              = api::rnc_lm_options{};
    options.rnc_lm->order       = order;
    options.max_iterations      = 100;
    options.function_tolerance  = 1e-10;
    options.gradient_tolerance  = 1e-12;
    options.parameter_tolerance = 1e-14;
    return options;
}
}  // namespace

TEST(RncDerivatives, FourthOrderCurveAndBaseDerivativesAreExact)
{
    auto        derivatives = make_rnc_derivatives(Exponential{}, 1, 1);
    vector_type base(1), v(1), a(1), c3(1);
    base << .3;
    v << .7;
    a << -.2;
    c3 << .4;
    rnc_curve_derivatives d;
    derivatives(base, {v, a, c3}, 4, d);
    const double e = std::exp(.3);
    EXPECT_NEAR(d.residual[2][0], e * (.7 * .7 - .2), 1e-14);
    EXPECT_NEAR(d.residual[3][0], e * (.7 * .7 * .7 + 3. * .7 * (-.2) + .4), 1e-14);
    EXPECT_NEAR(d.residual[4][0],
        e * (std::pow(.7, 4) + 6. * .7 * .7 * (-.2) + 3. * .2 * .2 + 4. * .7 * .4),
        1e-14);
    for (int k = 0; k <= 2; ++k)
        EXPECT_NEAR(d.jacobian[k](0, 0), d.residual[k][0], 1e-14);
}

TEST(RncDerivatives, ElementaryFunctionsHaveCorrectFourthDerivatives)
{
    auto        derivatives = make_rnc_derivatives(ElementaryFunctions{}, 1, 6);
    vector_type base(1), v(1);
    base << 2.;
    v << 1.;
    rnc_curve_derivatives d;
    derivatives(base, {v}, 4, d);
    EXPECT_NEAR(d.residual[4][0], std::sin(2.), 1e-14);
    EXPECT_NEAR(d.residual[4][1], std::cos(2.), 1e-14);
    EXPECT_NEAR(d.residual[4][2], -6. / 16., 1e-14);
    EXPECT_NEAR(d.residual[4][3], -15. / (16. * std::pow(2., 3.5)), 1e-14);
    EXPECT_NEAR(d.residual[4][4], 120. / 64., 1e-14);
    EXPECT_NEAR(d.residual[4][5], d.residual[4][3], 1e-14);
    EXPECT_NEAR(d.jacobian[2](2, 0), 2. / 8., 1e-14);
}

TEST(RncLm, SecondOrderAgreesWithAnalyticGeodesicAcceleration)
{
    auto problem                    = rnc_least_squares(Quadratic{}, 1, 1);
    auto options                    = configuration(2);
    options.max_iterations          = 1;
    options.rnc_lm->initial_damping = .1;
    vector_type x(1);
    x << 2.;
    const double v      = -12. / 17.6;
    const double a      = -8. * v * v / 17.6;
    const auto   result = api::solve(problem, x, options);
    EXPECT_EQ(result.accepted_steps, 1u);
    EXPECT_NEAR(result.parameters[0], 2. + v + .5 * a, 1e-14);
    EXPECT_EQ(result.effective_derivative_source, api::derivative_mode::automatic_differentiation);
    EXPECT_EQ(result.algorithm, api::algorithm::riemann_normal_coordinate_lm);
}

TEST(RncLm, ThirdAndFourthOrderIncludeMovingTangentTerms)
{
    auto        problem = rnc_least_squares(EmbeddedParabola{}, 1, 2);
    vector_type x(1);
    x << 1.;
    const double g = 5.5, v = -3. / g, a = -4. * v * v / g;
    const double c3 = -(16. * v * a + 4. * v * v * v) / g;
    const double c4 = -(24. * v * c3 + 16. * a * a + 28. * v * v * a) / g;
    for (int order : {3, 4})
    {
        auto options                    = configuration(order);
        options.max_iterations          = 1;
        options.rnc_lm->initial_damping = .1;
        const auto   result             = api::solve(problem, x, options);
        const double expected           = 1. + v + .5 * a + c3 / 6. + (order == 4 ? c4 / 24. : 0.);
        ASSERT_EQ(result.accepted_steps, 1u) << result.message;
        EXPECT_NEAR(result.parameters[0], expected, 1e-13);
    }
}

TEST(RncLm, CurveSearchReusesDerivativesAndScalesEachPowerOfT)
{
    auto                problem = rnc_least_squares(Quadratic{}, 1, 1);
    std::vector<double> trials;
    auto                residual = problem.residuals;
    problem.residuals            = [&](const vector_type& x, vector_type& r)
    {
        trials.push_back(x[0]);
        residual(x, r);
    };
    auto options                     = configuration(2);
    options.max_iterations           = 1;
    options.rnc_lm->initial_damping  = .1;
    options.rnc_lm->max_curve_trials = 8;
    vector_type x(1);
    x << .1;
    const double v = .198 / .044, a = -.4 * v * v / .044;
    const auto   result = api::solve(problem, x, options);
    ASSERT_GT(trials.size(), 1u);
    ASSERT_EQ(result.accepted_steps, 1u) << result.message;
    const double cost0 = .5 * .99 * .99, cost1 = .5 * std::pow(trials[0] * trials[0] - 1., 2);
    const double sigma = .198 * v;
    const double t     = std::clamp(sigma / (2. * (cost1 - cost0 + sigma)), .3, .5);
    EXPECT_NEAR(trials[1], .1 + t * v + .5 * t * t * a, 1e-12);
    EXPECT_EQ(result.jacobian_evaluations, 3u);  // base, c2, accepted base
    EXPECT_LT(result.objective, cost0);
}

TEST(RncLm, GeneralizedRosenbrockQuadraticAndCubicValleys)
{
    for (int power : {2, 3})
        for (int order : {2, 3, 4})
        {
            auto problem           = rnc_least_squares(GeneralizedRosenbrock{power, 1000.}, 2, 2);
            auto options           = configuration(order);
            options.max_iterations = 1000;
            options.rnc_lm->max_curve_trials = 8;
            vector_type x(2);
            x << 1., 1. / power;
            const auto result = api::solve(problem, x, options);
            EXPECT_TRUE(result.converged())
                << power << " " << order << " " << result.message << " " << result.objective;
            EXPECT_LT(result.objective, 1e-12);
        }
}

TEST(RncLm, RankDeficientJacobianIsRegularized)
{
    auto        problem = rnc_least_squares(RankDeficient{}, 2, 1);
    vector_type x       = vector_type::Zero(2);
    const auto  result  = api::solve(problem, x, configuration(4));
    EXPECT_TRUE(result.converged()) << result.message;
    EXPECT_NEAR(result.parameters.sum(), 3., 1e-9);
}

TEST(RncLm, StationaryNonzeroResidualStopsWithoutTrials)
{
    auto        problem = rnc_least_squares(Quadratic{}, 1, 1);
    vector_type x       = vector_type::Zero(1);
    const auto  result  = api::solve(problem, x, configuration(3));
    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.iterations, 0u);
    EXPECT_EQ(result.residual_evaluations, 0u);
    EXPECT_DOUBLE_EQ(result.objective, .5);
}

TEST(RncLm, InvalidOptionsAndMissingDerivativesFailBeforeEvaluation)
{
    auto        problem   = rnc_least_squares(Quadratic{}, 1, 1);
    auto        options   = configuration(3);
    vector_type x         = vector_type::Constant(1, 2.);
    options.rnc_lm->order = 5;
    EXPECT_EQ(api::solve(problem, x, options).status, api::solver_status::invalid_problem);
    options                         = configuration(3);
    options.rnc_lm->contraction_min = .9;
    EXPECT_EQ(api::solve(problem, x, options).status, api::solver_status::invalid_problem);
    options             = configuration(3);
    options.derivatives = api::derivative_mode::finite_difference;
    EXPECT_EQ(api::solve(problem, x, options).status, api::solver_status::unsupported_capability);
    options                 = configuration(3);
    problem.rnc_derivatives = {};
    EXPECT_EQ(api::solve(problem, x, options).status, api::solver_status::unsupported_capability);
}

TEST(RncLm, RejectedAndNonfiniteTrialsPreserveAcceptedParameters)
{
    auto problem      = rnc_least_squares(Quadratic{}, 1, 1);
    problem.residuals = [](const vector_type&, vector_type& r)
    { r[0] = std::numeric_limits<double>::infinity(); };
    auto options           = configuration(3);
    options.max_iterations = 2;
    vector_type x          = vector_type::Constant(1, 2.);
    const auto  result     = api::solve(problem, x, options);
    EXPECT_EQ(result.status, api::solver_status::max_iterations);
    EXPECT_EQ(result.iterations, 2u);
    EXPECT_EQ(result.accepted_steps, 0u);
    EXPECT_EQ(result.rejected_steps, 8u);
    EXPECT_DOUBLE_EQ(result.parameters[0], 2.);
    EXPECT_DOUBLE_EQ(result.objective, 4.5);
}

TEST(RncLm, InvalidDerivativeDimensionsAreReported)
{
    auto problem = rnc_least_squares(Quadratic{}, 1, 1);
    problem.rnc_derivatives =
        [](const vector_type&, const std::vector<vector_type>&, int, rnc_curve_derivatives&) {};
    const auto result = api::solve(problem, vector_type::Constant(1, 2.), configuration(3));
    EXPECT_EQ(result.status, api::solver_status::numerical_failure);
}

TEST(RncLm, SevereValleyMatchesThePapersProblemScale)
{
    for (int power : {2, 3})
        for (int order : {3, 4})
        {
            auto problem               = rnc_least_squares(GeneralizedRosenbrock{power, 1e6}, 2, 2);
            auto options               = configuration(order);
            options.max_iterations     = 2000;
            options.function_tolerance = std::sqrt(2e-4);
            vector_type x(2);
            x << 1., 1. / power;
            const auto result = api::solve(problem, x, options);
            EXPECT_TRUE(result.converged()) << power << " " << order << " " << result.message;
            EXPECT_LT(result.objective, 1e-4);
        }
}

TEST(RncLm, AnalyticCurveDerivativeCallbackWorksWithoutAutodiffAdapter)
{
    api::least_squares_problem problem;
    problem.num_parameters  = 1;
    problem.num_residuals   = 1;
    problem.residuals       = [](const vector_type& x, vector_type& r) { r[0] = x[0] - 3.; };
    problem.rnc_derivatives = [](const vector_type&               x,
                                  const std::vector<vector_type>& c,
                                  int                             order,
                                  rnc_curve_derivatives&          d)
    {
        d.residual.assign(order + 1, vector_type::Zero(1));
        d.jacobian.assign(std::max(0, order - 2) + 1, matrix_type::Zero(1, 1));
        d.residual[0][0] = x[0] - 3.;
        for (size_t q = 0; q < c.size(); ++q)
            d.residual[q + 1] = c[q];
        d.jacobian[0](0, 0) = 1.;
    };
    const auto result = api::solve(problem, vector_type::Zero(1), configuration(4));
    EXPECT_TRUE(result.converged());
    EXPECT_NEAR(result.parameters[0], 3., 1e-10);
    EXPECT_EQ(result.effective_derivative_source, api::derivative_mode::supplied);
}

TEST(RncLm, DampingExhaustionAndCallbackExceptionsAreNumericalFailures)
{
    auto problem                    = rnc_least_squares(Quadratic{}, 1, 1);
    auto options                    = configuration(3);
    options.rnc_lm->initial_damping = 1.;
    options.rnc_lm->damping_ceiling = 1.;
    problem.residuals               = [](const vector_type&, vector_type& r) { r[0] = 100.; };
    auto result                     = api::solve(problem, vector_type::Constant(1, 2.), options);
    EXPECT_EQ(result.status, api::solver_status::numerical_failure);
    EXPECT_EQ(result.iterations, 1u);
    EXPECT_DOUBLE_EQ(result.parameters[0], 2.);
    problem.residuals = [](const vector_type&, vector_type&) { throw std::domain_error("test"); };
    result            = api::solve(problem, vector_type::Constant(1, 2.), options);
    EXPECT_EQ(result.status, api::solver_status::numerical_failure);
    EXPECT_NE(result.message.find("test"), std::string::npos);
}

namespace
{
struct CubicResidual
{
    double                  hessian, cubic;
    template <class T> void operator()(const T* x, T* r) const
    {
        r[0] = T(1.) + x[0] + T(.5 * hessian) * x[0] * x[0] + T(cubic) * x[0] * x[0] * x[0];
    }
};
}  // namespace

TEST(RncLm, TrustRatioUsesInitialTangentInsteadOfCorrectedDisplacement)
{
    // Choose a cubic whose first full curve trial has actual reduction .33.
    // The tangent model gives rho=.6655 (keep lambda), while the displaced
    // linear model would give rho>.75 (incorrectly divide lambda by three).
    const double        v = -1. / 1.1, acceleration = .6, s = v + .5 * acceleration;
    const double        hessian     = -acceleration * 1.1 / (v * v);
    const double        target      = std::sqrt(.34);
    const double        cubic       = (target - 1. - s - .5 * hessian * s * s) / (s * s * s);
    auto                problem     = rnc_least_squares(CubicResidual{hessian, cubic}, 1, 1);
    auto                derivatives = problem.rnc_derivatives;
    std::vector<double> velocities;
    problem.rnc_derivatives = [&](const vector_type&              base,
                                  const std::vector<vector_type>& c,
                                  int                             order,
                                  rnc_curve_derivatives&          out)
    {
        if (order == 2)
            velocities.push_back(c[0][0]);
        derivatives(base, c, order, out);
    };
    auto options                    = configuration(2);
    options.max_iterations          = 2;
    options.rnc_lm->initial_damping = .1;
    const auto result               = api::solve(problem, vector_type::Zero(1), options);
    ASSERT_EQ(velocities.size(), 2u) << result.message;
    const double next_jacobian = 1. + hessian * s + 3. * cubic * s * s;
    EXPECT_NEAR(velocities[1], -target / (1.1 * next_jacobian), 1e-12);
}
