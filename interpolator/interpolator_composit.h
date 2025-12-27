#pragma once
#ifndef __QUARISMA_WRAP__

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <tuple>

#include "common/wrapping_hints.h"
#include "interpolator/interpolator.h"
#include "interpolator/interpolator_enum.h"
#include "interpolator/interpolator_factory.h"
#include "util/exception.h"

namespace quarisma
{
template <typename Container, typename T, typename S, typename... Interpolators>
class interpolator_composite : public interpolator<Container, T, S>
{
    using factory            = interpolator_factory<Container, T, S>;
    using base               = interpolator<Container, T, S>;
    using interpolator_tuple = std::tuple<ptr_const<Interpolators>...>;

    /// @brief Optimized segment extraction with bounds checking for any container type
    template <typename ContainerType>
    ContainerType extract_segment(const ContainerType& data, size_t start, size_t end) const
    {
        if (start >= data.size() || end > data.size() || start >= end)
        {
            return ContainerType();
        }

        // Use SFINAE to handle different container types
        return extract_segment_impl(
            data,
            start,
            end,
            std::is_same<ContainerType, std::vector<typename ContainerType::value_type>>{});
    }

private:
    /// @brief Implementation for std::vector - use iterators for performance
    template <typename ContainerType>
    ContainerType extract_segment_impl(
        const ContainerType& data, size_t start, size_t end, std::true_type) const
    {
        return ContainerType(data.begin() + start, data.begin() + end);
    }

    /// @brief Implementation for other containers - use element-wise copy
    template <typename ContainerType>
    ContainerType extract_segment_impl(
        const ContainerType& data, size_t start, size_t end, std::false_type) const
    {
        const size_t segment_size = end - start;

        // Handle quarisma::matrix specifically
        if constexpr (std::is_same_v<
                          ContainerType,
                          quarisma::matrix<typename ContainerType::value_type>>)
        {
            // For matrix, we assume it's a column vector (rows x 1)
            ContainerType result(segment_size, 1);
            for (size_t i = 0; i < segment_size; ++i)
            {
                result.at(i, 0) = data.at(start + i, 0);
            }
            return result;
        }
        else
        {
            // Generic fallback for other container types
            ContainerType result;
            // Try to use constructor with size if available
            if constexpr (std::is_constructible_v<ContainerType, size_t>)
            {
                result = ContainerType(segment_size);
                for (size_t i = 0; i < segment_size; ++i)
                {
                    result[i] = data[start + i];
                }
            }
            else
            {
                // Last resort: return empty container
                result = ContainerType();
            }
            return result;
        }
    }

public:
    /// @brief Legacy method for backward compatibility
    template <typename v>
    std::vector<v> subvector(const std::vector<v>& y, size_t i, size_t j)
    {
        return extract_segment(y, i, j);
    }

    /// @brief Calculate segment boundaries for interpolator region I
    template <size_t I>
    std::pair<size_t, size_t> calculate_segment_bounds() const
    {
        const size_t start_idx =
            (I > 0) ? std::distance(
                          this->x_.begin(),
                          std::lower_bound(this->x_.begin(), this->x_.end(), switch_points_[I - 1]))
                    : 0;

        const size_t end_idx =
            (I < sizeof...(Interpolators) - 1)
                ? std::distance(
                      this->x_.begin(),
                      std::upper_bound(this->x_.begin(), this->x_.end(), switch_points_[I]))
                : this->x_.size();

        return {start_idx, end_idx};
    }

    /// @brief Create interpolator with continuity constraints for smooth transitions
    template <size_t I>
    auto create_interpolator_with_continuity(
        interpolation_enum interp_type, std::vector<T>&& x_segment, Container&& y_segment)
    {
        // For middle segments, ensure C1 continuity by using derivative from previous interpolator
        if constexpr (I > 0 && I < sizeof...(Interpolators))
        {
            const auto& prev_interp = std::get<I - 1>(interpolators_);
            if (prev_interp && !x_segment.empty())
            {
                const S first_derivative = prev_interp->derivative(x_segment.front());
                return factory::create(
                    interp_type,
                    std::move(x_segment),
                    std::move(y_segment),
                    cubic_spline_condition_enum::FIRST_DERIVATIVE,
                    first_derivative);
            }
        }

        // For first and last segments, use natural boundary conditions
        return factory::create(interp_type, std::move(x_segment), std::move(y_segment));
    }

    /// @brief Reset all interpolators to nullptr for reinitialization
    void reset_interpolators()
    {
        reset_interpolators_impl(std::make_index_sequence<sizeof...(Interpolators)>{});
    }

    /// @brief Implementation of interpolator reset
    template <size_t... Is>
    void reset_interpolators_impl(std::index_sequence<Is...>)
    {
        ((std::get<Is>(interpolators_) = nullptr), ...);
    }

    template <typename v>
    vector<T> subvector(const vector<v>& y, size_t i, size_t j)
    {
        if (i >= y.size() || j > y.size() || i > j)
        {
            return vector<v>();
        }

        return vector<v>(y.data() + i, j - i);
    }

    template <typename v>
    matrix<v> subvector(const matrix<v>& y, size_t i, size_t j)
    {
        if (i >= y.size() || j > y.size() || i > j)
        {
            return matrix<v>();
        }

        return matrix<v>(y.data() + i * y.columns(), (j - i) * y.columns());
    }

    template <size_t... Is>
    void validate_interpolators(std::index_sequence<Is...>) const
    {
        ((validate_interpolator_at_index<Is>()), ...);
    }

    template <size_t I>
    void validate_interpolator_at_index() const
    {
        const auto& interpolator_ptr = std::get<I>(interpolators_);

        // Calculate segment bounds to check if interpolator should exist
        const auto [start_idx, end_idx]     = calculate_segment_bounds<I>();
        const bool should_have_interpolator = (end_idx > start_idx + 1);

        if (should_have_interpolator)
        {
            QUARISMA_CHECK(
                interpolator_ptr != nullptr,
                "Interpolator at index " + std::to_string(I) +
                    " cannot be null when segment has sufficient data points");
        }
        // Note: It's valid for interpolator to be null if segment has insufficient data points
    }

public:
    interpolator_composite() = default;

    interpolator_composite(
        std::array<T, sizeof...(Interpolators) - 1> switch_points,
        ptr_const<Interpolators>&&... interpolators)
        : switch_points_(std::move(switch_points)),
          interpolators_(std::move(interpolators)...),
          initialized_(true)  // Pre-constructed interpolators are already initialized
    {
        static_assert(sizeof...(Interpolators) > 0, "Must provide at least one interpolator");
        validate_switch_points();
        validate_interpolators(std::make_index_sequence<sizeof...(Interpolators)>{});
        update_aad_offsets_init(std::make_index_sequence<sizeof...(Interpolators)>{});
    }

    interpolator_composite(
        std::vector<T>                                           x_points,
        Container                                                y_points,
        std::array<T, sizeof...(Interpolators) - 1>              switch_points,
        std::array<interpolation_enum, sizeof...(Interpolators)> enums)
        : base(std::move(x_points), std::move(y_points)),
          switch_points_(std::move(switch_points)),
          interpolation_enums_(std::move(enums)),
          initialized_(false)
    {
        static_assert(sizeof...(Interpolators) > 0, "Must provide at least one interpolator");
        validate_switch_points();
        align_switch_points_to_x();
        // Initialize interpolators immediately for backward compatibility
        initialize();
    }

    // New method to align switch points with actual x values
    void align_switch_points_to_x()
    {
        // Skip if x_ is empty
        if (this->x_.empty())
            return;

        for (size_t i = 0; i < switch_points_.size(); ++i)
        {
            auto it = std::lower_bound(this->x_.begin(), this->x_.end(), switch_points_[i]);

            // If the switch point isn't exactly an x value and isn't smaller than all x values
            if (it != this->x_.end() && it != this->x_.begin() && *it > switch_points_[i])
            {
                // If the switch point is larger than the largest x value, use the largest x value
                switch_points_[i] = *(--it);
            }
        }

        // Re-validate after alignment
        validate_switch_points();
    }

    /// @brief Optimized interpolator creation with better memory management
    template <size_t... Is>
    void create_interpolators_optimized(std::index_sequence<Is...>)
    {
        ((create_interpolator_at_index_optimized<Is>()), ...);
    }

    /// @brief Legacy method for backward compatibility
    template <size_t... Is>
    void create_interpolators(
        const std::array<interpolation_enum, sizeof...(Interpolators)>& enums,
        std::index_sequence<Is...>)
    {
        interpolation_enums_ = enums;
        create_interpolators_optimized(std::index_sequence<Is...>{});
    }

    /// @brief Optimized interpolator creation with improved performance and caching
    template <size_t I>
    void create_interpolator_at_index_optimized()
    {
        const auto interp_type = interpolation_enums_[I];

        // Calculate segment boundaries with optimized bounds checking
        const auto [start_idx, end_idx] = calculate_segment_bounds<I>();

        // Only create interpolator if segment has sufficient points
        if (end_idx > start_idx + 1)
        {
            // Use move semantics for better performance
            auto x_segment = extract_segment(this->x_, start_idx, end_idx);
            auto y_segment = extract_segment(this->y_, start_idx, end_idx);

            // Create interpolator with continuity constraints for smooth transitions
            std::get<I>(interpolators_) = create_interpolator_with_continuity<I>(
                interp_type, std::move(x_segment), std::move(y_segment));
        }
        else
        {
            std::get<I>(interpolators_) = nullptr;
        }
    }

    /// @brief Legacy method for backward compatibility
    template <size_t I>
    void create_interpolator_at_index(interpolation_enum interp_type)
    {
        interpolation_enums_[I] = interp_type;
        create_interpolator_at_index_optimized<I>();
    }

    template <size_t... Is>
    void update_aad_offsets_init(std::index_sequence<Is...>)
    {
        ((update_aad_offsets_init_impl<Is>()), ...);
    }

    template <size_t I>
    void update_aad_offsets_init_impl()
    {
        const auto& tmp = std::get<I>(interpolators_);

        AAD_OFFSET(interpolators_)[I] = this->state_parameters_size_;
        if (tmp != nullptr)
        {
            this->state_parameters_size_ += tmp->state_parameters_size();
        }
    }

    /// @brief Initialize interpolators with optimized performance and caching
    /// Creates sub-interpolators for each region defined by switch points
    /// Uses lazy initialization to improve construction performance
    void initialize() override
    {
        if (initialized_)
            return;  // Early exit if already initialized

        // Reset interpolators and AAD state
        reset_interpolators();

        // Create interpolators for each region with performance optimization
        create_interpolators_optimized(std::make_index_sequence<sizeof...(Interpolators)>{});

        // Validate all created interpolators
        validate_interpolators(std::make_index_sequence<sizeof...(Interpolators)>{});

        // Initialize AAD offsets for automatic differentiation
        update_aad_offsets_init(std::make_index_sequence<sizeof...(Interpolators)>{});

        initialized_ = true;
    }

    void finalize_aad(double* state_parameters_aad) const override
    {
        finalize_aad_impl(
            state_parameters_aad, std::make_index_sequence<sizeof...(Interpolators)>{});

        // Set the rest of the state parameters to zero
        const auto size = this->state_parameters_size_ - AAD_OFFSET(interpolators_)[0];
        std::memset(
            state_parameters_aad + AAD_OFFSET(interpolators_)[0], 0.0, size * sizeof(double));
    }

    template <size_t... Is>
    void finalize_aad_impl(double* state_parameters_aad, std::index_sequence<Is...>) const
    {
        // Process in reverse order
        (void)std::initializer_list<int>{
            (process_interpolator<sizeof...(Is) - 1 - Is>(state_parameters_aad), 0)...};
    }

    template <size_t I>
    void process_interpolator(double* state_parameters_aad) const
    {
        const auto& interpolator_i = *std::get<I>(interpolators_);
        //fixme: case interpolator is null?

        const auto offset                 = AAD_OFFSET(interpolators_)[I];
        auto*      state_parameters_aad_i = state_parameters_aad + offset;

        interpolator_i.finalize_aad(state_parameters_aad_i);

        if constexpr (I > 0 && I < sizeof...(Interpolators))
        {
            if (!this->x_.empty())  // Only do this part if x_ is not empty
            {
                const auto& value_aad = interpolator_i.first_derivative_aad(state_parameters_aad_i);

                auto* state_parameters_aad_i_1 =
                    state_parameters_aad + AAD_OFFSET(interpolators_)[I - 1];

                const auto& interpolator = std::get<I - 1>(interpolators_).get();
                interpolator->derivative_aad(
                    value_aad, interpolator->back(), state_parameters_aad_i_1);
            }
        }
        size_t start_idx =
            (I > 0) ? std::distance(
                          this->x_.begin(),
                          std::lower_bound(this->x_.begin(), this->x_.end(), switch_points_[I - 1]))
                    : 0;
        size_t end_idx =
            (I < sizeof...(Interpolators) - 1)
                ? std::distance(
                      this->x_.begin(),
                      std::upper_bound(this->x_.begin(), this->x_.end(), switch_points_[I]))
                : this->x_.size();

        auto* y_aad = this->get_state_parameters_aad(this->AAD_OFFSET(y_), 0, state_parameters_aad);
        vector<double> tmp1(y_aad, this->y_.size());

        const auto*    val_aad = interpolator_i.y_aad(state_parameters_aad_i);
        vector<double> tmp2(val_aad, interpolator_i.size());

        for (size_t i = start_idx; i < end_idx; ++i, ++val_aad)
        {
            y_aad[i] += *val_aad;
        }
    }

    S derivative(const T& x) const override
    {
        const auto idx = find_segment(x);

        return interpolator_get(idx, std::make_index_sequence<sizeof...(Interpolators)>{})
            ->derivative(x);
    };

    void derivative_aad(S value_aad, const T& x, double* state_parameters_aad) const override
    {
        const auto idx = find_segment(x);

        return interpolator_get(idx, std::make_index_sequence<sizeof...(Interpolators)>{})
            ->derivative_aad(value_aad, x, state_parameters_aad + AAD_OFFSET(interpolators_)[idx]);
    }

    // Interpolate value at a given point
    S interpolate(const T& x) const override
    {
        const auto idx = find_segment(x);

        return interpolator_get(idx, std::make_index_sequence<sizeof...(Interpolators)>{})
            ->interpolate(x);
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
        const auto idx = find_segment(x);

        return interpolator_get(idx, std::make_index_sequence<sizeof...(Interpolators)>{})
            ->interpolate_aad(value_aad, x, state_parameters_aad + AAD_OFFSET(interpolators_)[idx]);
    }

    size_t find_interval(T x) const override
    {
        const auto idx = find_segment(x);

        return interpolator_get(idx, std::make_index_sequence<sizeof...(Interpolators)>{})
            ->find_interval(x);
    }

private:
    std::array<size_t, sizeof...(Interpolators)> AAD_OFFSET(interpolators_);
    std::array<T, sizeof...(Interpolators) - 1>  switch_points_;
    interpolator_tuple                           interpolators_;
    std::array<interpolation_enum, sizeof...(Interpolators)>
         interpolation_enums_;  ///< Cached interpolation types
    bool initialized_;          ///< Initialization state flag

    QUARISMA_SERIALIZATION_TEMPLATE(interpolator_composite, x_, y_, switch_points_, interpolators_);

    void validate_switch_points() const
    {
        for (size_t i = 1; i < switch_points_.size(); ++i)
        {
            QUARISMA_CHECK(
                switch_points_[i] > switch_points_[i - 1],
                "Switch points must be strictly increasing");
        }
    }

    // Find which segment contains x
    size_t find_segment(T x) const
    {
        if (switch_points_.empty() || x <= switch_points_[0])
        {
            return 0;
        }

        if (x >= switch_points_.back())
        {
            return switch_points_.size();
        }

        auto it = std::lower_bound(switch_points_.begin(), switch_points_.end(), x);
        return std::distance(switch_points_.begin(), it);
    }

    template <size_t... Is>
    const interpolator<Container, T, S>* interpolator_get(
        size_t idx, std::index_sequence<Is...>) const
    {
        const interpolator<Container, T, S>* result = nullptr;
        // Use fold expression to find the interpolator at the given index
        ((Is == idx ? result = std::get<Is>(interpolators_).get() : nullptr), ...);

        QUARISMA_CHECK(result != nullptr, "Invalid interpolator index");

        return result;
    }
};

// Helper function to deduce types and create a composite interpolator
template <typename Container, typename T, typename S, typename... Interpolators>
auto interpolator_make_composite(
    std::array<T, sizeof...(Interpolators) - 1> switch_points, Interpolators&&... interpolators)
{
    return util::make_ptr_unique_const<interpolator_composite<Container, T, S, Interpolators...>>(
        std::move(switch_points), std::forward<Interpolators>(interpolators)...);
}

// Example specialized versions for common cases
template <typename Container, typename T, typename S>
using interpolator_dual = interpolator_composite<
    Container,
    T,
    S,
    interpolator<Container, T, S>,
    interpolator<Container, T, S>>;

template <typename Container, typename T, typename S>
using interpolator_triple = interpolator_composite<
    Container,
    T,
    S,
    interpolator<Container, T, S>,
    interpolator<Container, T, S>,
    interpolator<Container, T, S>>;

template <
    typename Container,
    typename T,
    typename S,
    size_t N,
    typename = std::make_index_sequence<N>>
struct n_interpolators;

template <typename Container, typename T, typename S, size_t N, size_t... Is>
struct n_interpolators<Container, T, S, N, std::index_sequence<Is...>>
{
    template <size_t>
    using interp_type = interpolator<Container, T, S>;

    using type = interpolator_composite<Container, T, S, interp_type<Is>...>;
};

template <typename Container, typename T, typename S, size_t N>
using compose_interpolators = typename n_interpolators<Container, T, S, N>::type;

}  // namespace quarisma

#endif  // !__QUARISMA_WRAP__