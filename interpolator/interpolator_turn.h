#pragma once

#include <cmath>

#include "interpolator.h"

namespace quarisma
{
// Turn interpolator class for implementing "Turn" in curve construction
// This creates a spike in forward rates between specified dates
template <typename Container = std::vector<double>, typename T = double, typename S = double>
class interpolator_turn : public interpolator<Container, T, S>
{
public:
    // Constructor with x points (dates), y points (discount factors), and turn parameters
    interpolator_turn(
        std::vector<T> x_points,
        Container      y_points,
        T              turn_start_date,
        T              turn_end_date,
        S              turn_rate)
        : interpolator<Container, T, S>(std::move(x_points), std::move(y_points)),
          turn_start_(turn_start_date),
          turn_end_(turn_end_date),
          turn_rate_(turn_rate)
    {
        initialize();
    }

    // Default constructor
    interpolator_turn() = default;

    // Initialize the interpolator
    void initialize() override
    {
        // Find indices for turn start and end in the x_ array
        size_t start_idx = 0;
        size_t end_idx   = 0;

        for (size_t i = 0; i < this->x_.size(); ++i)
        {
            if (this->x_[i] == turn_start_)
            {
                start_idx = i;
            }
            if (this->x_[i] == turn_end_)
            {
                end_idx = i;
            }
        }

        // Validate that we found turn points
        QUARISMA_CHECK(start_idx < this->x_.size(), "Turn start date not found in x points");
        QUARISMA_CHECK(end_idx < this->x_.size(), "Turn end date not found in x points");
        QUARISMA_CHECK(start_idx < end_idx, "Turn start date must be before turn end date");

        // Store indices for later use
        turn_start_idx_ = start_idx;
        turn_end_idx_   = end_idx;
    }

    // Interpolate at point x
    S interpolate(const T& x) const override
    {
        // For points outside the range, use standard geometric interpolation
        if (x <= this->x_.front())
        {
            return this->y_.front();
        }
        if (x >= this->x_.back())
        {
            return this->y_.back();
        }

        // Find the interval containing x
        size_t idx = this->find_interval(x);

        // If we're between turn_start_ and turn_end_, apply the turn effect
        if (this->x_[idx] <= turn_start_ && x <= turn_end_)
        {
            // Calculate how far into the turn period we are
            T t_start = std::max(this->x_[idx], turn_start_);
            T t_end   = std::min(this->x_[idx + 1], turn_end_);

            // Day count fraction for the turn segment
            S day_count = t_end - t_start;

            // Calculate discount factor at t_start using geometric interpolation
            S df_start = geometric_interpolate(t_start, idx);

            // Apply turn rate to get discount factor at t_end
            S df_end = df_start * std::exp(-turn_rate_ * day_count);

            // Now interpolate between df_start and df_end
            S weight = (x - t_start) / (t_end - t_start);
            S log_df = std::log(df_start) * (1 - weight) + std::log(df_end) * weight;

            return std::exp(log_df);
        }
        else
        {
            // Standard geometric interpolation for points outside the turn
            return geometric_interpolate(x, idx);
        }
    }

    // AAD version of interpolate
    void interpolate_aad(
        const S& value_aad, const T& x, double* state_parameters_aad) const override
    {
        // Implementation would depend on your AAD framework
        // This is a simplified placeholder version
        size_t idx = this->find_interval(x);

        if (idx >= this->x_.size() - 1)
        {
            // At or beyond the last point
            this->add_to_state_parameter_aad(
                AAD_OFFSET(this->y_) + this->y_.size() - 1, value_aad, state_parameters_aad);
            return;
        }

        // Get AAD pointer to y array
        auto* y_aad = this->y_aad(state_parameters_aad);

        // Calculate weights for the interpolation
        T x0     = this->x_[idx];
        T x1     = this->x_[idx + 1];
        S weight = (x - x0) / (x1 - x0);

        // Apply AAD for geometric interpolation
        y_aad[idx] += value_aad * (1 - weight);
        y_aad[idx + 1] += value_aad * weight;
    }

    // First derivative at x
    S derivative(const T& x) const override
    {
        if (x <= this->x_.front())
        {
            return 0.0;
        }
        if (x >= this->x_.back())
        {
            return 0.0;
        }

        size_t idx = this->find_interval(x);

        // If we're in the turn period, calculate the derivative with the turn effect
        if (this->x_[idx] <= turn_start_ && x <= turn_end_)
        {
            // In turn period - high rate of change
            T t_start = std::max(this->x_[idx], turn_start_);
            T t_end   = std::min(this->x_[idx + 1], turn_end_);

            S day_count = t_end - t_start;
            S df_start  = geometric_interpolate(t_start, idx);
            S df_end    = df_start * std::exp(-turn_rate_ * day_count);

            // Derivative of ln(df) with respect to x
            return (std::log(df_end) - std::log(df_start)) / (t_end - t_start);
        }
        else
        {
            // Standard derivative for geometric interpolation
            T x0 = this->x_[idx];
            T x1 = this->x_[idx + 1];
            S y0 = this->y_[idx];
            S y1 = this->y_[idx + 1];

            return (std::log(y1) - std::log(y0)) / (x1 - x0);
        }
    }

    // AAD version of derivative
    void derivative_aad(S value_aad, const T& x, double* state_parameters_aad) const override
    {
        // Simplified placeholder - implementation would depend on AAD framework
        // Similar to interpolate_aad but for derivatives
    }

    void finalize_aad(double* parameters) const override
    {
        // Finalize AAD state
        // Implementation would depend on AAD framework
    }

    // Get first derivative for AAD
    S first_derivative_aad(double* state_parameters_aad) const override
    {
        // Implementation would depend on AAD framework
        return 0.0;
    }

private:
    // Helper method for geometric interpolation
    S geometric_interpolate(const T& x, size_t idx) const
    {
        T x0 = this->x_[idx];
        T x1 = this->x_[idx + 1];
        S y0 = this->y_[idx];
        S y1 = this->y_[idx + 1];

        // Linear interpolation in log space (geometric)
        S weight = (x - x0) / (x1 - x0);
        S log_y  = std::log(y0) * (1 - weight) + std::log(y1) * weight;

        return std::exp(log_y);
    }

    T      turn_start_;      // Start date of the turn
    T      turn_end_;        // End date of the turn
    S      turn_rate_;       // Turn rate
    size_t turn_start_idx_;  // Index of turn start in x_
    size_t turn_end_idx_;    // Index of turn end in x_

    // Serialization support
    SERIALIZE_MEMBERS(
        SERIALIZE_NVP(turn_start_),
        SERIALIZE_NVP(turn_end_),
        SERIALIZE_NVP(turn_rate_),
        SERIALIZE_NVP(turn_start_idx_),
        SERIALIZE_NVP(turn_end_idx_));
};

}  // namespace quarisma