#pragma once

#ifndef __QUARISMA_WRAP__

#include <cstddef>
#include <vector>

#include "MathModule.h"

namespace quarisma
{
template <typename value_t>
class matrix;
template <typename value_t>
class vector;
}  // namespace quarisma

namespace quarisma
{
class karhunen_loeve_expansion
{
public:
    MATH_API karhunen_loeve_expansion(
        double                     from,
        double                     to,
        const double               mean_reversion_speed,
        const std::vector<double>& mean_reversion_levels,
        const std::vector<double>& level_dates,
        const std::vector<double>& variances,
        const std::vector<double>& volatility_dates,
        double                     discretization_step);

    MATH_API void survival_probability(
        const vector<double>& state_variable,
        double                decay,
        const vector<double>& roots,
        const vector<double>& weights,
        vector<double>&       output) const;

private:
    std::vector<double> rho_;
    std::vector<double> means_;
    std::vector<double> stdev_;
    double              dt_;
    int                 define_;
};
}  // namespace quarisma

#endif  // !__QUARISMA_WRAP__
