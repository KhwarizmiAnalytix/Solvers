#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include "util/exception.h"

namespace quarisma
{

struct grid_point
{
    size_t i, j;    // Grid indices
    double tx, ty;  // Normalized coordinates [0,1]

    constexpr grid_point(size_t i_, size_t j_, double tx_, double ty_) noexcept
        : i(i_), j(j_), tx(tx_), ty(ty_)
    {
    }
};

template <typename Container, typename T, typename S>
class interpolator_2d
{
public:
    virtual ~interpolator_2d()                          = default;
    virtual S interpolate(const T& x, const T& y) const = 0;

    // Optimization: Allow compiler to inline simple accessors
    constexpr size_t   size_x() const noexcept { return x_.size(); }
    constexpr size_t   size_y() const noexcept { return y_.size(); }
    constexpr const T& front_x() const noexcept { return x_.front(); }
    constexpr const T& back_x() const noexcept { return x_.back(); }
    constexpr const T& front_y() const noexcept { return y_.front(); }
    constexpr const T& back_y() const noexcept { return y_.back(); }

protected:
    interpolator_2d(std::vector<T>&& x_points, std::vector<T>&& y_points, Container&& z_points)
        : x_(std::move(x_points)), y_(std::move(y_points)), z_(std::move(z_points))
    {
        validate();
    }

    // Optimization: Cache grid point calculation
    [[nodiscard]] grid_point find_grid_point(const T& x, const T& y) const noexcept
    {
        const size_t i = find_interval_x(x);
        const size_t j = find_interval_y(y);

        const size_t i2 = std::min(i + 1, size_x() - 1);
        const size_t j2 = std::min(j + 1, size_y() - 1);

        const auto tx = (i2 > i) ? (x - x_[i]) / (x_[i2] - x_[i]) : 0.;
        const auto ty = (j2 > j) ? (y - y_[j]) / (y_[j2] - y_[j]) : 0.;

        return grid_point(i, j, tx, ty);
    }

    // Optimization: Use binary search with hint for interval finding
    [[nodiscard]] size_t find_interval_x(T x) const noexcept
    {
        if (x <= front_x())
        {
            return 0;
        }
        if (x >= back_x())
        {
            return size_x() - 1;
        }

        return std::lower_bound(x_.begin(), x_.end(), x) - x_.begin() - 1;
    }

    [[nodiscard]] size_t find_interval_y(T y) const noexcept
    {
        if (y <= front_y())
        {
            return 0;
        }
        if (y >= back_y())
        {
            return size_y() - 1;
        }

        return std::lower_bound(y_.begin(), y_.end(), y) - y_.begin() - 1;
    }

    // Optimization: Inline z-value access
    [[nodiscard]] constexpr S get_z(size_t i, size_t j) const noexcept { return z_[i][j]; }

protected:
    interpolator_2d() = default;

    void validate() const
    {
        QUARISMA_CHECK(size_x() >= 1, "at least two x points required");
        QUARISMA_CHECK(size_y() >= 1, "at least two y points required");
        QUARISMA_CHECK_STRICTLY_INCREASING_DEBUG(x_);
        QUARISMA_CHECK_STRICTLY_INCREASING_DEBUG(y_);
    }

    std::vector<T> x_;
    std::vector<T> y_;
    Container      z_;
};
}  // namespace quarisma