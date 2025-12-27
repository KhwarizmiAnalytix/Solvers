#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "interpolator/interpolator.h"
#include "util/exception.h"

namespace quarisma
{
enum class interpolator_piecewise_constant_enum : int
{
    LEFT,
    RIGHT
};

template <typename Container, typename T, typename S>
class interpolator_flat : public interpolator<Container, T, S>
{
    using base = interpolator<Container, T, S>;

public:
    /**
     * @brief Constructs interpolator with known points
     * @param x_points X coordinates (must be strictly increasing)
     * @param y_points Y coordinates
     * @throws std::invalid_argument if inputs are invalid
     */
    interpolator_flat(
        std::vector<T>                       x_points,
        Container                            y_points,
        interpolator_piecewise_constant_enum type = interpolator_piecewise_constant_enum::LEFT)
        : base(std::move(x_points), std::move(y_points)), type_(type)
    {
    }

    void initialize() override {};
    void finalize_aad(QUARISMA_UNUSED double* parameters) const override {};

    S    derivative(const T& /*x*/) const override { return (S)0; }
    void derivative_aad(
        S /*value_aad*/, const T& /* x*/, double* /*state_parameters_aad*/) const override
    {
    }

    /**
     * @brief Interpolate/extrapolate at given point
     * @param x Point at which to interpolate
     * @param allow_extrapolate Whether to allow extrapolation
     * @return Interpolated/extrapolated value
     */
    S interpolate(const T& x) const override
    {
        if QUARISMA_UNLIKELY (x < this->front())
        {
            return std::move(this->y_[0]);
        }
        if QUARISMA_UNLIKELY (x > this->back())
        {
            const auto to = this->size() - 1;
            return std::move(this->y_[to]);
        }

        const auto from = this->find_interval(x);

        if (x == this->x_[from])
        {
            return std::move(this->y_[from]);
        }

        size_t to = from + 1;

        QUARISMA_CHECK_DEBUG(
            from >= 0 && to < this->x_.size(),
            "values ",
            x,
            " falling outside the boundaries of the interpolation cooredinates");

        return type_ == interpolator_piecewise_constant_enum::LEFT ? std::move(this->y_[from])
                                                                   : std::move(this->y_[to]);
    }

    void interpolate_aad(
        const S& value_aad, const T& x, double* state_parameters_aad) const override
    {
        if constexpr (std::is_same_v<S, double>)
        {
            if (is_almost_zero(value_aad))
            {
                return;
            }
            if QUARISMA_UNLIKELY (x <= this->front())
            {
                //return std::move(this->y_[0]);
                this->update_state_parameters(
                    value_aad, this->AAD_OFFSET(y_), 0, state_parameters_aad);
                return;
            }
            if QUARISMA_UNLIKELY (x >= this->back())
            {
                const auto to = this->size() - 1;
                //return std::move(this->y_[to]);
                this->update_state_parameters(
                    value_aad, this->AAD_OFFSET(y_), to, state_parameters_aad);
                return;
            }

            const auto from = this->find_interval(x);

            if (x == this->x_[from])
            {
                //return std::move(this->y_[from]);
                this->update_state_parameters(
                    value_aad, this->AAD_OFFSET(y_), from, state_parameters_aad);
                return;
            }

            size_t to = from + 1;

            QUARISMA_CHECK_DEBUG(
                from >= 0 && to < this->x_.size(),
                "values ",
                x,
                " falling outside the boundaries of the interpolation cooredinates");

            auto index = (type_ == interpolator_piecewise_constant_enum::LEFT) ? from : to;

            this->update_state_parameters(
                value_aad, this->AAD_OFFSET(y_), index, state_parameters_aad);
        }
    };

private:
    interpolator_piecewise_constant_enum type_;

    interpolator_flat() = default;

    QUARISMA_SERIALIZATION_TEMPLATE(interpolator_flat, x_, y_, type_);
};

}  // namespace quarisma