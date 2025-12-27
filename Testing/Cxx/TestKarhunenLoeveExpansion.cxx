#include <algorithm>
#include <cassert>
#include <cmath>

#include "common/constants.h"
#include "common/karhunen_loeve_expansion.h"
#include "expressions/expressions.h"
#include "terminals/matrix.h"
#include "terminals/vector.h"
#include "quarismaTest.h"

namespace
{
std::vector<double> model_dates           = {0.1, 0.3, 1.2, 2., 3., 4., 5.};
std::vector<double> mean_reversion_levels = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};

double              mean_reversion_speed = 0.01;
std::vector<double> volatilities         = {0.45, 0.55, 0.6, 0.65, 0.6, 0.5, 0.4, 0.35};
std::vector<double> volatility_dates     = {0., 1., 2., 3., 4., 5., 6., 7.};

auto create_karhunen_loeve_expansion(double from, double to)
{
    return quarisma::util::make_ptr_const<quarisma::karhunen_loeve_expansion>(
        from,
        to,
        mean_reversion_speed,
        model_dates,
        mean_reversion_levels,
        volatilities,
        volatility_dates,
        0.01);
}
}  // namespace

QUARISMATEST(Math, KarhunenLoeveExpansion)
{
    START_LOG_TO_FILE_NAME(KarhunenLoeveExpansion);

    double decay = 0.05;

    quarisma::vector<double> state_variable = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    quarisma::vector<double> roots          = {-0.1, 0., 0.1};
    quarisma::vector<double> weights        = {0.1, 0.1, 0.1};
    quarisma::vector<double> output(state_variable.size());

    create_karhunen_loeve_expansion(1., 2.5)->survival_probability(
        state_variable, decay, roots, weights, output);

    create_karhunen_loeve_expansion(0., 2.5)->survival_probability(
        state_variable, decay, roots, weights, output);

    create_karhunen_loeve_expansion(15., 20.5)->survival_probability(
        state_variable, decay, roots, weights, output);

    create_karhunen_loeve_expansion(15., 15.)->survival_probability(
        state_variable, decay, roots, weights, output);

    END_LOG_TO_FILE_NAME(KarhunenLoeveExpansion);
    END_TEST();
}