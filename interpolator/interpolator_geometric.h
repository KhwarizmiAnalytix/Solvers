#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "interpolator/interpolator_linear.h"
#include "terminals/vector.h"
#include "util/exception.h"

namespace quarisma
{
template <typename Container, typename T, typename S>
class interpolator_geometric : public interpolator_linear<Container, T, S>
{
    using base = interpolator_linear<Container, T, S>;

public:
    /**
     * @brief Constructs interpolator with known points, transforming y values to exp(y)
     * @param x_points X coordinates (must be strictly increasing)
     * @param y_points Y coordinates (will be transformed to exp(y))
     * @throws std::invalid_argument if inputs are invalid
     */
    interpolator_geometric(std::vector<T> x_points, Container y_points)
        : base(std::move(x_points), transform_y_values(std::move(y_points)))
    {
    }

    void finalize_aad(double* state_parameters_aad) const override
    {
        auto* y_aad = this->get_state_parameters_aad(this->AAD_OFFSET(y_), 0, state_parameters_aad);
        //y_out=log(y_in)
        vector<double> y_aad_vec(y_aad, this->y_.size());
        vector<double> y_vec(this->y_.data(), this->y_.size());

        y_aad_vec *= exp(-y_vec);
    };

private:
    /**
     * @brief Transform all y values to log(y) before passing to base constructor
     * @param y_points Original y values
     * @return Transformed y values (log(y))
     */
    static Container transform_y_values(Container y_points)
    {
        if constexpr (std::is_arithmetic<S>::value)
        {
            std::transform(
                y_points.begin(),
                y_points.end(),
                y_points.begin(),
                [](const auto& y) { return std::log(y); });
        }
        else
        {
            y_points = log(y_points);
        }

        return y_points;
    }

    interpolator_geometric() = default;

    QUARISMA_SERIALIZATION_TEMPLATE(interpolator_geometric, x_, y_);
};

}  // namespace quarisma