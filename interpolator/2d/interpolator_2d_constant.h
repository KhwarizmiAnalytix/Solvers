#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "common/serialization_macros.h"
#include "interpolator/2d/interpolator_2d.h"
#include "serialization.h"
#include "util/exception.h"

namespace quarisma
{
template <typename Container, typename T, typename S>
class interpolator_2d_constant final : public interpolator_2d<Container, T, S>
{
    using base = interpolator_2d<Container, T, S>;

public:
    interpolator_2d_constant(
        std::vector<T>&& x_points, std::vector<T>&& y_points, Container&& z_points)
        : base(std::move(x_points), std::move(y_points), std::move(z_points))
    {
    }

    [[nodiscard]] S interpolate(const T& x, const T& y) const override
    {
        const auto point = this->find_grid_point(x, y);
        return this->get_z(point.i, point.j);
    }

protected:
    void initialize() const { this->validate(); };

    interpolator_2d_constant() = default;

    QUARISMA_SERIALIZATION_TEMPLATE(interpolator_2d_constant, x_, y_, z_);
};
}  // namespace quarisma
