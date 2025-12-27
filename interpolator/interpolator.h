#pragma once

#include <algorithm>
#include <vector>

#include "common/aad_state_parameters_manager.h"
#include "common/constants.h"
#include "common/serialization_macros.h"
#include "util/exception.h"

namespace quarisma
{
template <typename Container, typename T, typename S>
class interpolator : public aad_state_parameters_manager
{
public:
    virtual S interpolate(const T& x) const = 0;

    virtual void interpolate_aad(
        const S& value_aad, const T& x, double* state_parameters_aad) const = 0;

    ~interpolator() override = default;

    virtual size_t find_interval(T x) const
    {
        if (x <= x_.front())
        {
            return 0;
        }
        if (x >= x_.back())
        {
            return x_.size() - 1;
        }

        auto       it  = std::upper_bound(x_.begin(), x_.end(), x);
        const auto idx = std::distance(x_.begin(), it) - 1;
        return static_cast<size_t>(idx);
    }

    const auto& x() const noexcept { return x_; }

    const auto& y() const noexcept { return y_; }

    const auto& dates() const noexcept { return x_; }

    const auto& rates() const noexcept { return y_; }

    const auto* y_aad(double* state_parameters_aad) const noexcept
    {
        return get_state_parameters_aad(AAD_OFFSET(y_), 0, state_parameters_aad);
    }

    auto& y() noexcept { return y_; }

    void finalize_aad(QUARISMA_UNUSED double* parameters) const override = 0;

    virtual void initialize() = 0;

    virtual S    derivative(const T& x) const                                                = 0;
    virtual void derivative_aad(S value_aad, const T& x, double* state_parameters_aad) const = 0;

    virtual S first_derivative_aad(double* /*state_parameters_aad*/) const { return 0.; }

    const auto front() const noexcept { return x_.front(); };

    const auto back() const noexcept { return x_.back(); };

    const auto size() const noexcept { return x_.size(); };

    void update(T x, double val)
    {
        const size_t i = find_interval(x);

        y_.data()[i] = val;
    }

protected:
    interpolator(std::vector<T> x_points, Container y_points)
        : x_(std::move(x_points)), y_(std::move(y_points))
    {
        validate();
        AAD_REGISTER_PARAMETER(y_, y_.size());
    }
    interpolator() = default;

    void validate() const
    {
        QUARISMA_CHECK(x_.size() > 0, "at least one point required");
        QUARISMA_CHECK_ALL_FINITE_DEBUG(x_);
        QUARISMA_CHECK_STRICTLY_INCREASING_DEBUG(x_);
        //QUARISMA_CHECK_ALL_FINITE_DEBUG(y_);
    }

    std::vector<T> x_;  // x coordinates
    Container      y_;  // y coordinates

    AAD_STATE_PARAMETERS(y_);
};
}  // namespace quarisma