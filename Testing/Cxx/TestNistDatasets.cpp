// All 27 NIST StRD nonlinear-regression datasets (docs/lm-literature-problem-list.md,
// section A). Each dataset is checked from both of NIST's published starting
// points against the certified residual sum of squares, through the
// library's default LM-GA configuration and RNC-LM order 3 (the project's
// defaults for each solver family). Plain LM, bold-acceptance LM, and RNC
// orders 1, 2, and 4 are additionally sanity-checked (finite, no crash) from
// the first start, without a tight RSS assertion: several of these datasets
// are NIST's own "Higher difficulty" cases, documented to defeat some good
// algorithms from some starts, and forcing every variant to match the
// certified fit from every start would hide that real behavior rather than
// report it.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "api/dispatch.h"
#include "problems/mgh09.h"
#include "problems/mgh10.h"
#include "problems/nist/bennett5.h"
#include "problems/nist/boxbod.h"
#include "problems/nist/chwirut1.h"
#include "problems/nist/chwirut2.h"
#include "problems/nist/danwood.h"
#include "problems/nist/eckerle4.h"
#include "problems/nist/enso.h"
#include "problems/nist/gauss1.h"
#include "problems/nist/gauss2.h"
#include "problems/nist/gauss3.h"
#include "problems/nist/hahn1.h"
#include "problems/nist/kirby2.h"
#include "problems/nist/lanczos1.h"
#include "problems/nist/lanczos2.h"
#include "problems/nist/lanczos3.h"
#include "problems/nist/misra1a.h"
#include "problems/nist/misra1b.h"
#include "problems/nist/misra1c.h"
#include "problems/nist/misra1d.h"
#include "problems/nist/mgh17.h"
#include "problems/nist/nelson.h"
#include "problems/nist/rat42.h"
#include "problems/nist/rat43.h"
#include "problems/nist/roszman1.h"
#include "problems/nist/thurber.h"
#include "solver_options/solver_options_lm.h"
#include "solver_options/solver_options_rnc_lm.h"
#include "solvers/integrations/rnc_autodiff.h"

using namespace solverslib;
using namespace solverslib::test;

namespace
{
struct NistCase
{
    std::string                                  name;
    std::function<api::least_squares_problem()>  plain;
    std::function<api::least_squares_problem()>  rnc;
    vector_type                                  start1;
    vector_type                                  start2;
    double                                        certified_rss = 0.0;
};

template <class PlainFactory, class Model>
NistCase make_case(std::string name,
    PlainFactory plain_factory,
    Model        model,
    std::size_t  n,
    std::size_t  m,
    vector_type  start1,
    vector_type  start2,
    double       certified_rss)
{
    NistCase c;
    c.name          = std::move(name);
    c.plain         = plain_factory;
    c.rnc           = [model, n, m]() { return rnc_least_squares(model, n, m); };
    c.start1        = std::move(start1);
    c.start2        = std::move(start2);
    c.certified_rss = certified_rss;
    return c;
}

// Some NIST models cannot go through RNC-LM's Taylor-series autodiff
// (make_rnc_derivatives / rnc_least_squares) as currently implemented:
// solvers/detail/taylor_series.h has no atan() overload, and its pow()
// only supports an int or compile-time double exponent, not a
// taylor_series exponent. DanWood (b1*x^b2), Rat43 and Bennett5 (both use
// a fitted parameter as the exponent of pow), and Roszman1 (uses atan) hit
// those gaps, so these four are checked under native LM only; c.rnc stays
// unset (RNC coverage is left as a documented autodiff-support gap rather
// than silently skipped).
template <class PlainFactory>
NistCase make_native_only_case(std::string name,
    PlainFactory plain_factory,
    vector_type  start1,
    vector_type  start2,
    double       certified_rss)
{
    NistCase c;
    c.name          = std::move(name);
    c.plain         = plain_factory;
    c.start1        = std::move(start1);
    c.start2        = std::move(start2);
    c.certified_rss = certified_rss;
    return c;
}

std::vector<NistCase> all_cases()
{
    std::vector<NistCase> cases;
    cases.push_back(make_case(
        "MGH09", mgh09_least_squares, mgh09_model{}, 4, 11, mgh09_start1(), mgh09_start2(),
        mgh09_certified_rss));
    cases.push_back(make_case(
        "MGH10", mgh10_least_squares, mgh10_model{}, 3, 16, mgh10_start1(), mgh10_start2(),
        mgh10_certified_rss));
    cases.push_back(make_case("Misra1a", misra1a_least_squares, misra1a_model{}, 2, 14,
        misra1a_start1(), misra1a_start2(), misra1a_certified_rss));
    cases.push_back(make_case("Chwirut2", chwirut2_least_squares, chwirut2_model{}, 3, 54,
        chwirut2_start1(), chwirut2_start2(), chwirut2_certified_rss));
    cases.push_back(make_case("Chwirut1", chwirut1_least_squares, chwirut1_model{}, 3, 214,
        chwirut1_start1(), chwirut1_start2(), chwirut1_certified_rss));
    cases.push_back(make_case("Lanczos3", lanczos3_least_squares, lanczos3_model{}, 6, 24,
        lanczos3_start1(), lanczos3_start2(), lanczos3_certified_rss));
    cases.push_back(make_case("Gauss1", gauss1_least_squares, gauss1_model{}, 8, 250,
        gauss1_start1(), gauss1_start2(), gauss1_certified_rss));
    cases.push_back(make_case("Gauss2", gauss2_least_squares, gauss2_model{}, 8, 250,
        gauss2_start1(), gauss2_start2(), gauss2_certified_rss));
    cases.push_back(make_native_only_case("DanWood", danwood_least_squares, danwood_start1(),
        danwood_start2(), danwood_certified_rss));
    cases.push_back(make_case("Misra1b", misra1b_least_squares, misra1b_model{}, 2, 14,
        misra1b_start1(), misra1b_start2(), misra1b_certified_rss));
    cases.push_back(make_case("Kirby2", kirby2_least_squares, kirby2_model{}, 5, 151,
        kirby2_start1(), kirby2_start2(), kirby2_certified_rss));
    cases.push_back(make_case("Hahn1", hahn1_least_squares, hahn1_model{}, 7, 236, hahn1_start1(),
        hahn1_start2(), hahn1_certified_rss));
    cases.push_back(make_case("Nelson", nelson_least_squares, nelson_model{}, 3, 128,
        nelson_start1(), nelson_start2(), nelson_certified_rss));
    cases.push_back(make_case("MGH17", mgh17_least_squares, mgh17_model{}, 5, 33, mgh17_start1(),
        mgh17_start2(), mgh17_certified_rss));
    cases.push_back(make_case("Lanczos1", lanczos1_least_squares, lanczos1_model{}, 6, 24,
        lanczos1_start1(), lanczos1_start2(), lanczos1_certified_rss));
    cases.push_back(make_case("Lanczos2", lanczos2_least_squares, lanczos2_model{}, 6, 24,
        lanczos2_start1(), lanczos2_start2(), lanczos2_certified_rss));
    cases.push_back(make_case("Gauss3", gauss3_least_squares, gauss3_model{}, 8, 250,
        gauss3_start1(), gauss3_start2(), gauss3_certified_rss));
    cases.push_back(make_case("Misra1c", misra1c_least_squares, misra1c_model{}, 2, 14,
        misra1c_start1(), misra1c_start2(), misra1c_certified_rss));
    cases.push_back(make_case("Misra1d", misra1d_least_squares, misra1d_model{}, 2, 14,
        misra1d_start1(), misra1d_start2(), misra1d_certified_rss));
    cases.push_back(make_native_only_case("Roszman1", roszman1_least_squares, roszman1_start1(),
        roszman1_start2(), roszman1_certified_rss));
    cases.push_back(make_case("ENSO", enso_least_squares, enso_model{}, 9, 168, enso_start1(),
        enso_start2(), enso_certified_rss));
    cases.push_back(make_case("Thurber", thurber_least_squares, thurber_model{}, 7, 37,
        thurber_start1(), thurber_start2(), thurber_certified_rss));
    cases.push_back(make_case("BoxBOD", boxbod_least_squares, boxbod_model{}, 2, 6,
        boxbod_start1(), boxbod_start2(), boxbod_certified_rss));
    cases.push_back(make_case("Rat42", rat42_least_squares, rat42_model{}, 3, 9, rat42_start1(),
        rat42_start2(), rat42_certified_rss));
    cases.push_back(make_case("Eckerle4", eckerle4_least_squares, eckerle4_model{}, 3, 35,
        eckerle4_start1(), eckerle4_start2(), eckerle4_certified_rss));
    cases.push_back(make_native_only_case("Rat43", rat43_least_squares, rat43_start1(),
        rat43_start2(), rat43_certified_rss));
    cases.push_back(make_native_only_case("Bennett5", bennett5_least_squares, bennett5_start1(),
        bennett5_start2(), bennett5_certified_rss));
    return cases;
}

std::shared_ptr<const solver_options_lm> plain_lm_opts()
{
    return solver_options_lm_builder()
        .with_geodesic_acceleration(false)
        .with_bold_acceptance(false)
        .with_max_iterations(2000)
        .build();
}

std::shared_ptr<const solver_options_lm> lm_ga_opts()
{
    return solver_options_lm_builder().with_geodesic_acceleration(true).with_max_iterations(2000).build();
}

std::shared_ptr<const solver_options_lm> bold_lm_opts()
{
    return solver_options_lm_builder()
        .with_geodesic_acceleration(false)
        .with_bold_acceptance(true)
        .with_max_iterations(2000)
        .build();
}

std::shared_ptr<const solver_options_rnc_lm> rnc_opts(int order)
{
    return solver_options_rnc_lm_builder().with_order(order).with_max_iterations(2000).build();
}

double residual_sum_of_squares(const api::solver_result& result)
{
    return *result.residual_norm * *result.residual_norm;
}

double rss_tolerance(double certified_rss)
{
    return std::max(1e-6, std::abs(certified_rss) * 1e-4);
}

class NistDatasets : public ::testing::TestWithParam<NistCase>
{
};

TEST_P(NistDatasets, LmGaAndRncOrder3MatchCertifiedRssFromBothStarts)
{
    const auto& c         = GetParam();
    const auto  tolerance = rss_tolerance(c.certified_rss);

    int start_index = 0;
    for (const auto& start : {c.start1, c.start2})
    {
        ++start_index;
        SCOPED_TRACE("Start" + std::to_string(start_index));
        {
            SCOPED_TRACE("LmGa");
            const auto result = api::solve(c.plain(), start, *lm_ga_opts());
            ASSERT_TRUE(result.residual_norm.has_value());
            EXPECT_NEAR(residual_sum_of_squares(result), c.certified_rss, tolerance);
        }
        if (c.rnc)
        {
            SCOPED_TRACE("RncOrder3");
            const auto result = api::solve(c.rnc(), start, *rnc_opts(3));
            ASSERT_TRUE(result.residual_norm.has_value());
            EXPECT_NEAR(residual_sum_of_squares(result), c.certified_rss, tolerance);
        }
    }
}

// Sanity-only coverage of the remaining variants (plain LM, bold LM, RNC
// orders 1, 2, 4) from NIST's first start: finite, non-throwing results, not
// a certified-fit assertion. Full-matrix convergence is tracked as follow-up
// work in docs/lm-literature-problem-list.md, not asserted here for every
// variant and dataset.
TEST_P(NistDatasets, RemainingVariantsProduceFiniteResults)
{
    const auto& c     = GetParam();
    const auto& start = c.start1;

    for (const auto& options : {plain_lm_opts(), bold_lm_opts()})
    {
        const auto result = api::solve(c.plain(), start, *options);
        EXPECT_TRUE(std::isfinite(result.objective));
    }
    if (c.rnc)
    {
        for (int order : {1, 2, 4})
        {
            SCOPED_TRACE("RncOrder" + std::to_string(order));
            const auto result = api::solve(c.rnc(), start, *rnc_opts(order));
            EXPECT_TRUE(std::isfinite(result.objective));
        }
    }
}

std::string nist_case_name(const ::testing::TestParamInfo<NistCase>& info)
{
    return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(Nist, NistDatasets, ::testing::ValuesIn(all_cases()), nist_case_name);
}  // namespace
