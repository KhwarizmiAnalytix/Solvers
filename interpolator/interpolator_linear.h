#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "common/serialization_macros.h"
#include "interpolator/interpolator.h"
#include "serialization.h"
#include "util/exception.h"

namespace quarisma
{
template <typename Container, typename T, typename S>
class interpolator_linear : public interpolator<Container, T, S>
{
    using base = interpolator<Container, T, S>;

public:
    /**
     * @brief Constructs interpolator with known points
     * @param x_points X coordinates (must be strictly increasing)
     * @param y_points Y coordinates
     * @throws std::invalid_argument if inputs are invalid
     */
    interpolator_linear(std::vector<T> x_points, Container y_points)
        : base(std::move(x_points), std::move(y_points))
    {
    }

    void initialize() override {};
    void finalize_aad(QUARISMA_UNUSED double* parameters) const override {};

    S derivative(const T& x) const override
    {
        if QUARISMA_UNLIKELY (x <= this->front())
        {
            return (this->y_[1] - this->y_[0]) / (this->x_[1] - this->x_[0]);
        }
        if QUARISMA_LIKELY (x >= this->back())
        {
            const auto to = this->size() - 1;

            return (this->y_[to] - this->y_[to - 1]) / (this->x_[to] - this->x_[to - 1]);
        }

        const auto from = this->find_interval(x);
        size_t     to   = from + 1;

        return (this->y_[to] - this->y_[from]) / (this->x_[to] - this->x_[from]);
    }

    void derivative_aad(S value_aad, const T& x, double* state_parameters_aad) const override
    {
        if constexpr (std::is_same_v<S, double>)
        {
            if (is_almost_zero(value_aad))
            {
                return;
            }

            if QUARISMA_UNLIKELY (x <= this->front())
            {
                //return (this->y_[1] - this->y_[0]) / (this->x_[1] - this->x_[0]);
                const auto t = 1. / (this->x_[1] - this->x_[0]);
                this->update_state_parameters(
                    value_aad * t, this->AAD_OFFSET(y_), 1, state_parameters_aad);
                this->update_state_parameters(
                    -value_aad * t, this->AAD_OFFSET(y_), 0, state_parameters_aad);
                return;
            }
            if QUARISMA_LIKELY (x >= this->back())
            {
                const auto to = this->size() - 1;

                // return (this->y_[to] - this->y_[to - 1]) / (this->x_[to] - this->x_[to - 1]);
                const auto t = 1. / (this->x_[to] - this->x_[to - 1]);
                this->update_state_parameters(
                    value_aad * t, this->AAD_OFFSET(y_), to, state_parameters_aad);
                this->update_state_parameters(
                    -value_aad * t, this->AAD_OFFSET(y_), to - 1, state_parameters_aad);
                return;
            }

            const auto from = this->find_interval(x);
            size_t     to   = from + 1;

            const auto t = 1. / (this->x_[to] - this->x_[from]);
            this->update_state_parameters(
                value_aad * t, this->AAD_OFFSET(y_), to, state_parameters_aad);
            this->update_state_parameters(
                -value_aad * t, this->AAD_OFFSET(y_), to - 1, state_parameters_aad);
        }
    }

    /**
     * @brief Interpolate/extrapolate at given point
     * @param x Point at which to interpolate
     * @param allow_extrapolate Whether to allow extrapolation
     * @return Interpolated/extrapolated value
     */
    S interpolate(const T& x) const override
    {
        if QUARISMA_UNLIKELY (this->x_.size() == 1)
        {
            return this->y_[0];
        }

        if QUARISMA_UNLIKELY (x <= this->front())
        {
            return evaluate(x, 0, 1);
        }
        if QUARISMA_UNLIKELY (x >= this->back())
        {
            const int to = this->size() - 1;
            return evaluate(x, to - 1, to);
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

        return evaluate(x, from, to);
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
        }
        if QUARISMA_UNLIKELY (this->x_.size() == 1)
        {
            //return this->y_[0];

            if constexpr (std::is_arithmetic<S>::value)
            {
                this->update_state_parameters(
                    value_aad, this->AAD_OFFSET(y_), 0, state_parameters_aad);
            }
            else
            {
                size_t i = 0;
                for (const auto& itr : value_aad)
                {
                    this->update_state_parameters(
                        itr, this->AAD_OFFSET(y_), i, state_parameters_aad);
                    i++;
                }
            }
            return;
        }
        if QUARISMA_UNLIKELY (x <= this->front())
        {
            //return evaluate(x, 0, 1);
            evaluate_aad(value_aad, x, 0, 1, state_parameters_aad);
            return;
        }
        if QUARISMA_UNLIKELY (x >= this->back())
        {
            const auto to = this->size() - 1;
            //return evaluate(x, to - 1, to);
            evaluate_aad(value_aad, x, to - 1, to, state_parameters_aad);
            return;
        }

        const auto from = this->find_interval(x);

        if (x == this->x_[from])
        {
            //return std::move(this->y_[from]);
            if constexpr (std::is_arithmetic<S>::value)
            {
                this->update_state_parameters(
                    value_aad, this->AAD_OFFSET(y_), from, state_parameters_aad);
            }
            else
            {
                const auto offset_from = from * value_aad.size();

                size_t i = 0;
                for (const auto& itr : value_aad)
                {
                    this->update_state_parameters(
                        itr, this->AAD_OFFSET(y_), offset_from + i, state_parameters_aad);
                    i++;
                }
            }
            return;
        }

        size_t to = from + 1;

        QUARISMA_CHECK_DEBUG(
            from >= 0 && to < this->x_.size(),
            "values ",
            x,
            " falling outside the boundaries of the interpolation cooredinates");

        evaluate_aad(value_aad, x, from, to, state_parameters_aad);
    };

protected:
    void initialize() const {}

    interpolator_linear() = default;

    QUARISMA_SERIALIZATION_TEMPLATE(interpolator_linear, x_, y_);

    auto evaluate(T x, size_t from, size_t to) const
    {
        const auto x1 = this->x_[from];
        const auto x2 = this->x_[to];
        const auto y1 = this->y_[from];
        const auto y2 = this->y_[to];

        const auto t = (x - x1) / (x2 - x1);

        return y1 + t * (y2 - y1);
    }

    auto evaluate_aad(
        const S& value_aad, const T& x, size_t from, size_t to, double* state_parameters_aad) const
    {
        const auto x1 = this->x_[from];
        const auto x2 = this->x_[to];

        const auto t = (x - x1) / (x2 - x1);

        // return y1 + t * (y2 - y1);

        if constexpr (std::is_arithmetic<S>::value)
        {
            this->update_state_parameters(
                value_aad * (1 - t), this->AAD_OFFSET(y_), from, state_parameters_aad);
            this->update_state_parameters(
                value_aad * t, this->AAD_OFFSET(y_), to, state_parameters_aad);
        }
        else
        {
            const auto offset_from = from * value_aad.size();
            const auto offset_to   = to * value_aad.size();

            size_t i = 0;
            for (const auto& itr : value_aad)
            {
                this->update_state_parameters(
                    itr * (1 - t), this->AAD_OFFSET(y_), offset_from + i, state_parameters_aad);
                this->update_state_parameters(
                    itr * t, this->AAD_OFFSET(y_), offset_to + i, state_parameters_aad);
                i++;
            }
        }
    }
};
}  // namespace quarisma