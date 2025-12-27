#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "interpolator/interpolator_linear.h"
#include "util/exception.h"

namespace quarisma
{
//tex:
// $ \frac{\ln(df_t)} {
//     t - t_r} = \left( \frac{t - t_i} { t_{i + 1} - ti } \right) \frac{\ln(df_{t_{i + 1}})} {
//                    t_{i + 1} - tr} + \left( \frac{t_{i + 1} - t} {
//                    t_{i + 1} - ti
//                } \right) \frac{\ln(df_{t_i})} {t_i - t_r} $ $ t_i \leq t \leq t_{
//                    i + 1} $

template <typename Container, typename T, typename S>
class interpolator_average : public interpolator_linear<Container, T, S>
{
    using base = interpolator_linear<Container, T, S>;

public:
    /**
     * @brief Constructs geometric average interpolator with known points
     * @param x_points X coordinates (time points, must be strictly increasing)
     * @param y_points Y coordinates (discount factors)
     * @throws std::invalid_argument if inputs are invalid
     */
    interpolator_average(std::vector<T> x_points, Container y_points)
        : base(std::move(x_points), std::move(y_points))
    {
        x_ref_ = this->x_[0];
    }

    /**
     * @brief Interpolate/extrapolate transformed value at given time point
     * @param t Time point at which to interpolate
     * @return Interpolated/extrapolated value multiplied by (t-t_ref)
     */
    S interpolate(const T& t) const override
    {
        const double fraction = (t - x_ref_) / 365.25;
        // Get the interpolated rate from base class and multiply by (t-t_ref)
        return fraction * base::interpolate(t);
    }

    void interpolate_aad(
        const S& value_aad, const T& t, double* state_parameters_aad) const override
    {
        if constexpr (std::is_same_v<S, double>)
        {
            if (is_almost_zero(value_aad))
            {
                return;
            }
        }

        // Just use base class implementation
        base::interpolate_aad((t - x_ref_) * value_aad, t, state_parameters_aad);
    }

    S derivative(const T& t) const override
    {
        // Just use base class to get derivative of transformed value
        return base::interpolate(t) + (t - x_ref_) * base::derivative(t);
    }

    void derivative_aad(S value_aad, const T& t, double* state_parameters_aad) const override
    {
        // Just use base class implementation
        //return base::interpolate(t) + (t - x_ref_) * base::derivative(t);
        base::derivative_aad((t - x_ref_) * value_aad, t, state_parameters_aad);
        base::interpolate_aad(value_aad, t, state_parameters_aad);
    }

private:
    /**
     * @brief Transform discount factors to rates for linear interpolation
     * @param y_points Original discount factors
     * @param x_points Time points
     * @return Transformed rates
     */
    static Container transform_y_values(Container& y_points, const std::vector<T>& x_points)
    {
        if constexpr (std::is_arithmetic<S>::value)
        {
            const T t_ref = x_points.front();

            std::transform(
                y_points.begin(),
                y_points.end(),
                x_points.begin(),
                y_points.begin(),
                [t_ref](const S& y, const T& x) { return y / (x - t_ref); });
        }
        else
        {
            // Handle vector/matrix case if needed
        }

        return y_points;
    }

    T x_ref_;

    interpolator_average() = default;

    QUARISMA_SERIALIZATION_TEMPLATE(interpolator_average, x_, y_, x_ref_);
};

}  // namespace quarisma