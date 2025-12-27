#pragma once

#include "common/pointer.h"
#include "interpolator/2d/interpolator_2d.h"
#include "interpolator/2d/interpolator_2d_constant.h"
#include "interpolator/2d/interpolator_2d_cubic_splin.h"
#include "interpolator/2d/interpolator_bilinear.h"
#include "interpolator/interpolator_average.h"
#include "interpolator/interpolator_cubic_hermite.h"
#include "interpolator/interpolator_cubic_spline.h"
#include "interpolator/interpolator_enum.h"
#include "interpolator/interpolator_flat.h"
#include "interpolator/interpolator_geometric.h"
#include "interpolator/interpolator_geometric_average.h"
#include "interpolator/interpolator_linear.h"
#include "interpolator/interpolator_linear_exponential.h"

namespace quarisma
{
template <typename Container, typename T, typename S>
class interpolator_factory
{
public:
    static ptr_mutable<interpolator<Container, T, S>> create_ptr_mutable(
        interpolation_enum          type,
        std::vector<T>              x,
        Container                   y,
        cubic_spline_condition_enum left_condition = cubic_spline_condition_enum::SECOND_DERIVATIVE,
        S                           left_value     = 0,
        cubic_spline_condition_enum right_condition =
            cubic_spline_condition_enum::SECOND_DERIVATIVE,
        S right_value = 0)
    {
        switch (type)
        {
        case interpolation_enum::LINEAR:
            return util::make_ptr_mutable<interpolator_linear<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::AVERAGE:
            return util::make_ptr_mutable<interpolator_average<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::LINEAR_EXPONENTIAL:
            return util::make_ptr_mutable<interpolator_linear_exponential<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::GEOMETRIC:
            return util::make_ptr_mutable<interpolator_geometric<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::GEOMETRIC_AVERAGE:
            return util::make_ptr_mutable<interpolator_geometric_average<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::CUBIC_HERMITE:
            return util::make_ptr_mutable<interpolator_cubic_hermite<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::CUBIC_SPLINE:
            return util::make_ptr_mutable<interpolator_cubic_spline<Container, T, S>>(
                std::move(x),
                std::move(y),
                left_condition,
                left_value,
                right_condition,
                right_value);

        case interpolation_enum::PIECEWISE_CONSTANT_LEFT:
            return util::make_ptr_mutable<interpolator_flat<Container, T, S>>(
                std::move(x), std::move(y), interpolator_piecewise_constant_enum::LEFT);

        case interpolation_enum::PIECEWISE_CONSTANT_RIGHT:
            return util::make_ptr_mutable<interpolator_flat<Container, T, S>>(
                std::move(x), std::move(y), interpolator_piecewise_constant_enum::RIGHT);

        default:
            throw std::invalid_argument("Unknown interpolator type");
        }
    }

    static ptr_const<interpolator<Container, T, S>> create_ptr(
        interpolation_enum          type,
        std::vector<T>              x,
        Container                   y,
        cubic_spline_condition_enum left_condition = cubic_spline_condition_enum::SECOND_DERIVATIVE,
        S                           left_value     = 0,
        cubic_spline_condition_enum right_condition =
            cubic_spline_condition_enum::SECOND_DERIVATIVE,
        S right_value = 0)
    {
        switch (type)
        {
        case interpolation_enum::LINEAR:
            return util::make_ptr_const<interpolator_linear<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::AVERAGE:
            return util::make_ptr_const<interpolator_average<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::LINEAR_EXPONENTIAL:
            return util::make_ptr_const<interpolator_linear_exponential<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::GEOMETRIC:
            return util::make_ptr_const<interpolator_geometric<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::GEOMETRIC_AVERAGE:
            return util::make_ptr_const<interpolator_geometric_average<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::CUBIC_HERMITE:
            return util::make_ptr_const<interpolator_cubic_hermite<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::CUBIC_SPLINE:
            return util::make_ptr_const<interpolator_cubic_spline<Container, T, S>>(
                std::move(x),
                std::move(y),
                left_condition,
                left_value,
                right_condition,
                right_value);

        case interpolation_enum::PIECEWISE_CONSTANT_LEFT:
            return util::make_ptr_const<interpolator_flat<Container, T, S>>(
                std::move(x), std::move(y), interpolator_piecewise_constant_enum::LEFT);

        case interpolation_enum::PIECEWISE_CONSTANT_RIGHT:
            return util::make_ptr_const<interpolator_flat<Container, T, S>>(
                std::move(x), std::move(y), interpolator_piecewise_constant_enum::RIGHT);

        default:
            throw std::invalid_argument("Unknown interpolator type");
        }
    }

    static ptr_unique_const<interpolator<Container, T, S>> create(
        const interpolation_enum&   type,
        std::vector<T>              x,
        Container                   y,
        cubic_spline_condition_enum left_condition = cubic_spline_condition_enum::SECOND_DERIVATIVE,
        S                           left_value     = 0,
        cubic_spline_condition_enum right_condition =
            cubic_spline_condition_enum::SECOND_DERIVATIVE,
        S right_value = 0)
    {
        switch (type)
        {
        case interpolation_enum::LINEAR:
            return util::make_ptr_unique_const<interpolator_linear<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::AVERAGE:
            return util::make_ptr_unique_const<interpolator_average<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::GEOMETRIC:
            return util::make_ptr_unique_const<interpolator_geometric<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::GEOMETRIC_AVERAGE:
            return util::make_ptr_unique_const<interpolator_geometric_average<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::LINEAR_EXPONENTIAL:
            return util::make_ptr_unique_const<interpolator_linear_exponential<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::CUBIC_HERMITE:
            return util::make_ptr_unique_const<interpolator_cubic_hermite<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::CUBIC_SPLINE:
            return util::make_ptr_unique_const<interpolator_cubic_spline<Container, T, S>>(
                std::move(x),
                std::move(y),
                left_condition,
                left_value,
                right_condition,
                right_value);

        case interpolation_enum::PIECEWISE_CONSTANT_LEFT:
            return util::make_ptr_unique_const<interpolator_flat<Container, T, S>>(
                std::move(x), std::move(y), interpolator_piecewise_constant_enum::LEFT);

        case interpolation_enum::PIECEWISE_CONSTANT_RIGHT:
            return util::make_ptr_unique_const<interpolator_flat<Container, T, S>>(
                std::move(x), std::move(y), interpolator_piecewise_constant_enum::RIGHT);

        default:
            throw std::invalid_argument("Unknown interpolator type");
        }
    }

    static ptr_unique_mutable<interpolator<Container, T, S>> create_mutable(
        const interpolation_enum&   type,
        std::vector<T>              x,
        Container                   y,
        cubic_spline_condition_enum left_condition = cubic_spline_condition_enum::SECOND_DERIVATIVE,
        S                           left_value     = 0,
        cubic_spline_condition_enum right_condition =
            cubic_spline_condition_enum::SECOND_DERIVATIVE,
        S right_value = 0)
    {
        switch (type)
        {
        case interpolation_enum::LINEAR:
            return util::make_ptr_unique_mutable<interpolator_linear<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::AVERAGE:
            return util::make_ptr_unique_mutable<interpolator_average<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::GEOMETRIC:
            return util::make_ptr_unique_mutable<interpolator_geometric<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::GEOMETRIC_AVERAGE:
            return util::make_ptr_unique_mutable<interpolator_geometric_average<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::LINEAR_EXPONENTIAL:
            return util::make_ptr_unique_mutable<interpolator_linear_exponential<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::CUBIC_HERMITE:
            return util::make_ptr_unique_mutable<interpolator_cubic_hermite<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::CUBIC_SPLINE:
            return util::make_ptr_unique_mutable<interpolator_cubic_spline<Container, T, S>>(
                std::move(x),
                std::move(y),
                left_condition,
                left_value,
                right_condition,
                right_value);

        case interpolation_enum::PIECEWISE_CONSTANT_LEFT:
            return util::make_ptr_unique_mutable<interpolator_flat<Container, T, S>>(
                std::move(x), std::move(y), interpolator_piecewise_constant_enum::LEFT);

        case interpolation_enum::PIECEWISE_CONSTANT_RIGHT:
            return util::make_ptr_unique_mutable<interpolator_flat<Container, T, S>>(
                std::move(x), std::move(y), interpolator_piecewise_constant_enum::RIGHT);

        default:
            throw std::invalid_argument("Unknown interpolator type");
        }
    }

    static ptr_unique_const<interpolator_2d<Container, T, S>> create_2d(
        const interpolation_enum& type, std::vector<T> x, std::vector<T> y, Container z)
    {
        switch (type)
        {
        case interpolation_enum::LINEAR:
            return util::make_ptr_unique_const<interpolator_bilinear<Container, T, S>>(
                std::move(x), std::move(y), std::move(z));

            /* case interpolation_enum::GEOMETRIC:
            return util::make_ptr_unique_const<interpolator_2d_cubic_splin<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::CUBIC_HERMITE:
            return util::make_ptr_unique_const<interpolator_cubic_hermite<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::CUBIC_SPLINE:
            return util::make_ptr_unique_const<interpolator_2d_cubic_splin<Container, T, S>>(
                std::move(x), std::move(y), std::move(z));*/
        default:
            throw std::invalid_argument("Unknown interpolator type");
        }
    }

    static ptr_const<interpolator_2d<Container, T, S>> create_2d_ptr(
        const interpolation_enum& type, std::vector<T> x, std::vector<T> y, Container z)
    {
        switch (type)
        {
        case interpolation_enum::LINEAR:
            return util::make_ptr_const<interpolator_bilinear<Container, T, S>>(
                std::move(x), std::move(y), std::move(z));

            /* case interpolation_enum::GEOMETRIC:
            return util::make_ptr_unique_const<interpolator_2d_cubic_splin<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::CUBIC_HERMITE:
            return util::make_ptr_unique_const<interpolator_cubic_hermite<Container, T, S>>(
                std::move(x), std::move(y));

        case interpolation_enum::CUBIC_SPLINE:
            return util::make_ptr_unique_const<interpolator_2d_cubic_splin<Container, T, S>>(
                std::move(x), std::move(y), std::move(z));*/
        default:
            throw std::invalid_argument("Unknown interpolator type");
        }
    }
};
}  // namespace quarisma