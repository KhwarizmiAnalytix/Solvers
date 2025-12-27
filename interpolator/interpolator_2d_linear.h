#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "common/serialization_macros.h"
#include "interpolator/interpolator_2d.h"
#include "serialization.h"
#include "util/exception.h"

namespace quarisma
{
template <typename Container, typename T, typename S>
class interpolator_bilinear final : public interpolator_2d<Container, T, S>
{
    using base = interpolator_2d<Container, T, S>;

public:
    interpolator_bilinear(
        std::vector<T>&& x_points, std::vector<T>&& y_points, Container&& z_points)
        : base(std::move(x_points), std::move(y_points), std::move(z_points))
    {
    }

    [[nodiscard]] S interpolate(const T& x, const T& y) const override
    {
        const auto point = this->find_grid_point(x, y);

        // Fast path for exact matches
        if (point.tx == 0. && point.ty == 0.)
        {
            return this->get_z(point.i, point.j);
        }

        size_t i_next = point.i + 1;
        if (x >= this->x_.back())
        {
            i_next = point.i;
        }
        size_t j_next = point.j + 1;
        if (y >= this->y_.back())
        {
            j_next = point.j;
        }

        // Optimization: Cache values and minimize array access
        const auto f11 = this->get_z(point.i, point.j);
        const auto f21 = this->get_z(i_next, point.j);
        const auto f12 = this->get_z(point.i, j_next);
        const auto f22 = this->get_z(i_next, j_next);

        // Optimization: Minimize multiplications
        const auto tx1 = 1. - point.tx;
        const auto ty1 = 1. - point.ty;

        return tx1 * ty1 * f11 + point.tx * ty1 * f21 + tx1 * point.ty * f12 +
               point.tx * point.ty * f22;
    }

protected:
    void initialize() const { this->validate(); };

    interpolator_bilinear() = default;

    QUARISMA_SERIALIZATION_TEMPLATE(interpolator_bilinear, x, y, z);
};
}  // namespace quarisma
