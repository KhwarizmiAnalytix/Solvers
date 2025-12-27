//fixme:
//#pragma once
//
//#include <algorithm>
//#include <array>
//#include <vector>
//
//#include "util/exception.h"
//
//namespace quarisma
//{
//
//// Helper struct for bicubic coefficients
//template <typename S>
//struct bicubic_patch
//{
//    std::array<std::array<S, 4>, 4> coefs;  // [i][j] corresponds to x^i * y^j term
//
//    [[nodiscard]] S evaluate(T tx, T ty) const noexcept
//    {
//        // Horner's method for efficient polynomial evaluation
//        S result = 0;
//        for (int i = 3; i >= 0; --i)
//        {
//            S temp = 0;
//            for (int j = 3; j >= 0; --j)
//            {
//                temp = temp * ty + coefs[i][j];
//            }
//            result = result * tx + temp;
//        }
//        return result;
//    }
//};
//
//template <typename Container, typename T, typename S>
//class interpolator_2d_cubic_splin final : public interpolator_2d<Container, T, S>
//{
//    using base = interpolator_2d<Container, T, S>;
//
//public:
//    // Constructor with optional derivatives
//    interpolator_2d_cubic_splin(
//        std::vector<T>&& x_points, std::vector<T>&& y_points, Container&& z_points)
//        : base(std::move(x_points), std::move(y_points), std::move(z_points))
//    {
//        initialize_derivatives();
//    }
//
//    template <typename DX, typename DY, typename DXY>
//    interpolator_2d_cubic_splin(
//        std::vector<T>&& x_points,
//        std::vector<T>&& y_points,
//        Container&&      z_points,
//        DX&&             dx_points,
//        DY&&             dy_points,
//        DXY&&            dxy_points)
//        : base(std::move(x_points), std::move(y_points), std::move(z_points)),
//          dx_(std::forward<DX>(dx_points)),
//          dy_(std::forward<DY>(dy_points)),
//          dxy_(std::forward<DXY>(dxy_points))
//    {
//        validate_derivatives();
//    }
//
//    [[nodiscard]] S interpolate(const T& x, const T& y) const override
//    {
//        const auto point = this->find_grid_point(x, y);
//
//        // Fast path for exact matches
//       /* if (point.tx == 0 && point.ty == 0)
//        {
//            return this->get_z(point.i, point.j);
//        }*/
//
//        const auto patch = get_patch(point.i, point.j);
//        return patch.evaluate(point.tx, point.ty);
//    }
//
//private:
//    Container dx_;   // x derivatives (df/dx)
//    Container dy_;   // y derivatives (df/dy)
//    Container dxy_;  // cross derivatives (d2f/dxdy)
//
//    void validate_derivatives() const
//    {
//        const auto size = this->size_x() * this->size_y();
//        QUARISMA_CHECK(dx_.size() == size, "dx size must match grid size");
//        QUARISMA_CHECK(dy_.size() == size, "dy size must match grid size");
//        QUARISMA_CHECK(dxy_.size() == size, "dxy size must match grid size");
//    }
//
//    void initialize_derivatives()
//    {
//        const auto size = this->size_x() * this->size_y();
//        dx_.resize(size);
//        dy_.resize(size);
//        dxy_.resize(size);
//
//        estimate_derivatives();
//    }
//
//    // Get derivatives at grid point (i,j)
//    [[nodiscard]] S get_dx(size_t i, size_t j) const noexcept { return dx_[i][j]; }
//
//    [[nodiscard]] S get_dy(size_t i, size_t j) const noexcept { return dy_[i][j]; }
//
//    [[nodiscard]] S get_dxy(size_t i, size_t j) const noexcept { return dxy_[i][j]; }
//
//    // Estimate derivatives using finite differences
//    void estimate_derivatives()
//    {
//        estimate_dx_derivatives();
//        estimate_dy_derivatives();
//        estimate_cross_derivatives();
//    }
//
//    void estimate_dx_derivatives()
//    {
//        for (size_t i = 0; i < this->size_x(); ++i)
//        {
//            for (size_t j = 0; j < this->size_y(); ++j)
//            {
//                if (i == 0)
//                {
//                    // Forward difference at left boundary
//                    const T h = this->x_[1] - this->x_[0];
//                    dx_[i][j] = (this->get_z(1, j) - this->get_z(0, j)) / h;
//                }
//                else if (i == this->size_x() - 1)
//                {
//                    // Backward difference at right boundary
//                    const T h = this->x_[i] - this->x_[i - 1];
//                    dx_[i][j] = (this->get_z(i, j) - this->get_z(i - 1, j)) / h;
//                }
//                else
//                {
//                    // Central difference for interior points
//                    const T h = this->x_[i + 1] - this->x_[i - 1];
//                    dx_[i][j] = (this->get_z(i + 1, j) - this->get_z(i - 1, j)) / h;
//                }
//            }
//        }
//    }
//
//    void estimate_dy_derivatives()
//    {
//        for (size_t i = 0; i < this->size_x(); ++i)
//        {
//            for (size_t j = 0; j < this->size_y(); ++j)
//            {
//                if (j == 0)
//                {
//                    // Forward difference at bottom boundary
//                    const T h = this->y_[1] - this->y_[0];
//                    dy_[i][j] = (this->get_z(i, 1) - this->get_z(i, 0)) / h;
//                }
//                else if (j == this->size_y() - 1)
//                {
//                    // Backward difference at top boundary
//                    const T h = this->y_[j] - this->y_[j - 1];
//                    dy_[i][j] = (this->get_z(i, j) - this->get_z(i, j - 1)) / h;
//                }
//                else
//                {
//                    // Central difference for interior points
//                    const T h = this->y_[j + 1] - this->y_[j - 1];
//                    dy_[i][j] = (this->get_z(i, j + 1) - this->get_z(i, j - 1)) / h;
//                }
//            }
//        }
//    }
//
//    void estimate_cross_derivatives()
//    {
//        for (size_t i = 0; i < this->size_x(); ++i)
//        {
//            for (size_t j = 0; j < this->size_y(); ++j)
//            {
//                // Use central differences where possible
//                const size_t i1 = (i == 0) ? 0 : i - 1;
//                const size_t i2 = (i == this->size_x() - 1) ? i : i + 1;
//                const size_t j1 = (j == 0) ? 0 : j - 1;
//                const size_t j2 = (j == this->size_y() - 1) ? j : j + 1;
//
//                const T hx = this->x_[i2] - this->x_[i1];
//                const T hy = this->y_[j2] - this->y_[j1];
//
//                dxy_[i][j] = (this->get_z(i2, j2) - this->get_z(i2, j1) - this->get_z(i1, j2) +
//                              this->get_z(i1, j1)) /
//                             (hx * hy);
//            }
//        }
//    }
//
//    // Get bicubic patch for interpolation
//    [[nodiscard]] bicubic_patch<S> get_patch(size_t i, size_t j) const
//    {
//        // Get grid cell size
//        const T hx = this->x_[i + 1] - this->x_[i];
//        const T hy = this->y_[j + 1] - this->y_[j];
//
//        // Get function values and derivatives at corners
//        std::array<std::array<S, 4>, 4> p;
//
//        // Function values
//        p[0][0] = this->get_z(i, j);
//        p[0][1] = this->get_z(i, j + 1);
//        p[1][0] = this->get_z(i + 1, j);
//        p[1][1] = this->get_z(i + 1, j + 1);
//
//        // x derivatives
//        p[2][0] = get_dx(i, j) * hx;
//        p[2][1] = get_dx(i, j + 1) * hx;
//        p[3][0] = get_dx(i + 1, j) * hx;
//        p[3][1] = get_dx(i + 1, j + 1) * hx;
//
//        // y derivatives
//        p[0][2] = get_dy(i, j) * hy;
//        p[0][3] = get_dy(i, j + 1) * hy;
//        p[1][2] = get_dy(i + 1, j) * hy;
//        p[1][3] = get_dy(i + 1, j + 1) * hy;
//
//        // Cross derivatives
//        p[2][2] = get_dxy(i, j) * hx * hy;
//        p[2][3] = get_dxy(i, j + 1) * hx * hy;
//        p[3][2] = get_dxy(i + 1, j) * hx * hy;
//        p[3][3] = get_dxy(i + 1, j + 1) * hx * hy;
//
//        // Convert to bicubic coefficients
//        return compute_bicubic_coefficients(p);
//    }
//
//    // Compute bicubic coefficients using the matrix method
//    [[nodiscard]] static bicubic_patch<S> compute_bicubic_coefficients(
//        const std::array<std::array<S, 4>, 4>& p)
//    {
//        bicubic_patch<S> result;
//
//        // Pre-computed inverse of the bicubic matrix
//        static constexpr double M[16][16] = {
//            // ... (16x16 matrix coefficients would go here)
//            // This is the standard bicubic interpolation matrix
//        };
//
//        // Convert point data to coefficient matrix
//        // This would involve matrix multiplication with the pre-computed inverse
//        // For brevity, implementation details are omitted
//
//        return result;
//    }
//};
//}  // namespace quarisma