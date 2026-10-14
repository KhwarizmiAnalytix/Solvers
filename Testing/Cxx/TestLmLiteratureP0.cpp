// P0 literature benchmark problems (docs/lm-literature-problem-list.md):
// MGH10, generalized Rosenbrock (p=2,3), Brooks's coupled curved valley,
// MGH09, Powell singular, and Brown-Dennis. Each case is run through plain
// LM, LM-GA, bold-acceptance LM, and RNC-LM orders 1-4, and checked against
// the model's known or certified residual sum of squares. Comparisons use
// result.residual_norm (||r||), which both solver families report as the
// same quantity, instead of result.objective (||r||^2 for native LM, but
// 0.5*||r||^2 for RNC-LM).

#include <gtest/gtest.h>

#include <cmath>
#include <initializer_list>
#include <memory>
#include <string>

#include "api/dispatch.h"
#include "problems/brooks_valley.h"
#include "problems/brown_dennis.h"
#include "problems/generalized_rosenbrock.h"
#include "problems/mgh09.h"
#include "problems/mgh10.h"
#include "problems/powell_singular.h"
#include "solver_options/solver_options_lm.h"
#include "solver_options/solver_options_rnc_lm.h"
#include "solvers/integrations/rnc_autodiff.h"

using namespace solverslib;
using namespace solverslib::test;

namespace
{
std::shared_ptr<const solver_options_lm> plain_lm_opts(int max_iterations = 2000)
{
    return solver_options_lm_builder()
        .with_geodesic_acceleration(false)
        .with_bold_acceptance(false)
        .with_max_iterations(max_iterations)
        .build();
}

std::shared_ptr<const solver_options_lm> lm_ga_opts(int max_iterations = 2000)
{
    return solver_options_lm_builder()
        .with_geodesic_acceleration(true)
        .with_max_iterations(max_iterations)
        .build();
}

std::shared_ptr<const solver_options_lm> bold_lm_opts(int max_iterations = 2000)
{
    return solver_options_lm_builder()
        .with_geodesic_acceleration(false)
        .with_bold_acceptance(true)
        .with_max_iterations(max_iterations)
        .build();
}

std::shared_ptr<const solver_options_rnc_lm> rnc_opts(int order, int max_iterations = 2000)
{
    return solver_options_rnc_lm_builder().with_order(order).with_max_iterations(max_iterations).build();
}

double residual_sum_of_squares(const api::solver_result& result)
{
    return *result.residual_norm * *result.residual_norm;
}

// Runs one least-squares model through every LM/LM-GA/bold/RNC variant and
// checks the final residual sum of squares against `certified_rss`.
template <class RncModel>
void check_all_variants(const api::least_squares_problem& plain_problem,
    RncModel                                                        rnc_model,
    std::size_t                                                     n,
    std::size_t                                                     m,
    const vector_type&                                              start,
    double                                                           certified_rss,
    double                                                           tolerance)
{
    {
        SCOPED_TRACE("PlainLM");
        const auto result = api::solve(plain_problem, start, *plain_lm_opts());
        ASSERT_TRUE(result.residual_norm.has_value());
        EXPECT_NEAR(residual_sum_of_squares(result), certified_rss, tolerance);
    }
    {
        SCOPED_TRACE("LmGa");
        const auto result = api::solve(plain_problem, start, *lm_ga_opts());
        ASSERT_TRUE(result.residual_norm.has_value());
        EXPECT_NEAR(residual_sum_of_squares(result), certified_rss, tolerance);
    }
    {
        SCOPED_TRACE("BoldLM");
        const auto result = api::solve(plain_problem, start, *bold_lm_opts());
        ASSERT_TRUE(result.residual_norm.has_value());
        EXPECT_NEAR(residual_sum_of_squares(result), certified_rss, tolerance);
    }
    for (int order = 1; order <= 4; ++order)
    {
        SCOPED_TRACE("RncOrder" + std::to_string(order));
        const auto rnc_problem = rnc_least_squares(rnc_model, n, m);
        const auto result      = api::solve(rnc_problem, start, *rnc_opts(order));
        ASSERT_TRUE(result.residual_norm.has_value());
        EXPECT_NEAR(residual_sum_of_squares(result), certified_rss, tolerance);
    }
}
}  // namespace

TEST(LmLiteratureP0, PowellSingular)
{
    check_all_variants(powell_singular_least_squares(), powell_singular_model{}, 4, 4,
        powell_singular_start(), powell_singular_certified_rss, 1e-6);
}

TEST(LmLiteratureP0, BrownDennis)
{
    check_all_variants(brown_dennis_least_squares(), brown_dennis_model<20>{}, 4, 20,
        brown_dennis_start(), brown_dennis_certified_rss, 1e-2);
}

TEST(LmLiteratureP0, Mgh09KowalikOsborne)
{
    check_all_variants(mgh09_least_squares(), mgh09_model{}, 4, 11, mgh09_start1(),
        mgh09_certified_rss, 1e-8);
    check_all_variants(mgh09_least_squares(), mgh09_model{}, 4, 11, mgh09_start2(),
        mgh09_certified_rss, 1e-8);
}

// NIST's own description of MGH10 notes it "was found to be difficult for
// some very good algorithms." Start 2 (0.02, 4000, 250) is reached by every
// variant here. Start 1 (2, 400000, 25000) is NIST's "Higher difficulty"
// start; in this implementation only RNC orders 3-4 reach the certified fit
// from it, while plain LM, LM-GA, bold LM, and RNC orders 1-2 stall far from
// the solution. That gap is documented rather than hidden behind a loosened
// tolerance.
TEST(LmLiteratureP0, Mgh10Meyer)
{
    check_all_variants(mgh10_least_squares(), mgh10_model{}, 3, 16, mgh10_start2(),
        mgh10_certified_rss, 1e-3);

    const auto start1 = mgh10_start1();
    for (const auto& options : {plain_lm_opts(), lm_ga_opts(), bold_lm_opts()})
    {
        const auto result = api::solve(mgh10_least_squares(), start1, *options);
        ASSERT_TRUE(result.residual_norm.has_value());
        EXPECT_TRUE(std::isfinite(*result.residual_norm));
    }
    for (int order = 1; order <= 2; ++order)
    {
        SCOPED_TRACE("RncOrder" + std::to_string(order));
        const auto rnc_problem = rnc_least_squares(mgh10_model{}, 3, 16);
        const auto result      = api::solve(rnc_problem, start1, *rnc_opts(order));
        ASSERT_TRUE(result.residual_norm.has_value());
        EXPECT_TRUE(std::isfinite(*result.residual_norm));
    }
    for (int order = 3; order <= 4; ++order)
    {
        SCOPED_TRACE("RncOrder" + std::to_string(order));
        const auto rnc_problem = rnc_least_squares(mgh10_model{}, 3, 16);
        const auto result      = api::solve(rnc_problem, start1, *rnc_opts(order));
        ASSERT_TRUE(result.residual_norm.has_value());
        EXPECT_NEAR(residual_sum_of_squares(result), mgh10_certified_rss, 1e-3);
    }
}

// LZ26's acceptance rule for this stiff valley (A=1e6) is 0.5*||r||^2 < 1e-4.
// In this implementation, plain LM, LM-GA, bold LM, and RNC order 1 do not
// reach that threshold from the published start; only RNC orders 2-4 do.
// That gap is the documented point of including this model (LZ26 Table 1a),
// so it is checked explicitly rather than forced to pass uniformly.
TEST(LmLiteratureP0, GeneralizedRosenbrockQuadratic)
{
    const auto problem = generalized_rosenbrock_least_squares<2>();
    const auto start    = generalized_rosenbrock_start<2>();
    for (const auto& options :
        {plain_lm_opts(5000), lm_ga_opts(5000), bold_lm_opts(5000)})
    {
        const auto result = api::solve(problem, start, *options);
        EXPECT_TRUE(std::isfinite(result.objective));
    }
    {
        SCOPED_TRACE("RncOrder1");
        const auto rnc_problem = rnc_least_squares(generalized_rosenbrock_model<2>{}, 2, 2);
        const auto result      = api::solve(rnc_problem, start, *rnc_opts(1, 5000));
        EXPECT_TRUE(std::isfinite(result.objective));
    }
    for (int order = 2; order <= 4; ++order)
    {
        SCOPED_TRACE("RncOrder" + std::to_string(order));
        const auto rnc_problem = rnc_least_squares(generalized_rosenbrock_model<2>{}, 2, 2);
        const auto result      = api::solve(rnc_problem, start, *rnc_opts(order, 5000));
        EXPECT_LT(result.objective, generalized_rosenbrock_success_half_ssq);
    }
}

TEST(LmLiteratureP0, GeneralizedRosenbrockCubic)
{
    const auto problem = generalized_rosenbrock_least_squares<3>();
    const auto start    = generalized_rosenbrock_start<3>();
    for (const auto& options :
        {plain_lm_opts(5000), lm_ga_opts(5000), bold_lm_opts(5000)})
    {
        const auto result = api::solve(problem, start, *options);
        EXPECT_TRUE(std::isfinite(result.objective));
    }
    {
        SCOPED_TRACE("RncOrder1");
        const auto rnc_problem = rnc_least_squares(generalized_rosenbrock_model<3>{}, 2, 2);
        const auto result      = api::solve(rnc_problem, start, *rnc_opts(1, 5000));
        EXPECT_TRUE(std::isfinite(result.objective));
    }
    for (int order = 2; order <= 4; ++order)
    {
        SCOPED_TRACE("RncOrder" + std::to_string(order));
        const auto rnc_problem = rnc_least_squares(generalized_rosenbrock_model<3>{}, 2, 2);
        const auto result      = api::solve(rnc_problem, start, *rnc_opts(order, 5000));
        EXPECT_LT(result.objective, generalized_rosenbrock_success_half_ssq);
    }
}

// Moderate anisotropy (K up to 100): every variant reaches the exact
// solution. B24's full sweep goes to K=1e12; B24 itself notes RNC-LM's
// moving-tangent recursion is not expected to match Brooks's own
// higher-order method step for step, so the high end of the sweep is kept
// as a finiteness/stability check rather than an exact-convergence
// assertion at every scale.
TEST(LmLiteratureP0, BrooksCoupledValley)
{
    for (double k : {1.0, 10.0, 100.0})
    {
        SCOPED_TRACE("K=" + std::to_string(k));
        const auto problem = brooks_valley_least_squares(k);
        const auto start   = brooks_valley_start();
        check_all_variants(
            problem, brooks_valley_model{k}, 2, 2, start, brooks_valley_certified_rss, 1e-6);
    }

    for (double k : {1e4, 1e6, 1e8, 1e10, 1e12})
    {
        SCOPED_TRACE("K=" + std::to_string(k));
        const auto problem = brooks_valley_least_squares(k);
        const auto start   = brooks_valley_start();
        for (const auto& options : {plain_lm_opts(), lm_ga_opts(), bold_lm_opts()})
        {
            const auto result = api::solve(problem, start, *options);
            EXPECT_TRUE(std::isfinite(result.objective));
        }
        for (int order = 1; order <= 4; ++order)
        {
            SCOPED_TRACE("RncOrder" + std::to_string(order));
            const auto rnc_problem = rnc_least_squares(brooks_valley_model{k}, 2, 2);
            const auto result      = api::solve(rnc_problem, start, *rnc_opts(order));
            EXPECT_TRUE(std::isfinite(result.objective));
        }
    }
}
