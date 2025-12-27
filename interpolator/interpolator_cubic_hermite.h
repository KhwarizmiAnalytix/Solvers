#pragma once

#include <cmath>
#include <vector>

#include "common/vectorization_type_traits.h"
#include "interpolator/interpolator.h"
#include "terminals/matrix.h"
#include "util/exception.h"

namespace quarisma
{
template <typename Container, typename T, typename S>
class interpolator_cubic_hermite : public interpolator<Container, T, S>
{
    using base = interpolator<Container, T, S>;

    inline void aad_update_state_parameters_hermite(
        double value_aad, size_t idx, double hy, double hm, double* state_parameters_aad) const
    {
        this->update_state_parameters(
            value_aad * hy, this->AAD_OFFSET(y_), idx, state_parameters_aad);
        this->update_state_parameters(
            value_aad * hm, this->AAD_OFFSET(m_), idx, state_parameters_aad);
    }

public:
    // Constructor with points and derivatives
    interpolator_cubic_hermite(std::vector<T> x_points, Container y_points, Container&& derivatives)
        : base(std::move(x_points), std::move(y_points)), m_(std::move(derivatives))
    {
        AAD_REGISTER_PARAMETER(m_, m_.size());
    }

    // Constructor with points only - will estimate derivatives
    interpolator_cubic_hermite(std::vector<T> x_points, Container y_points)
        : base(std::move(x_points), std::move(y_points))
    {
        initialize();
    }

    S derivative(const T& x) const override
    {
        if QUARISMA_UNLIKELY (x <= this->front())
        {
            return m_[0];
        }
        if QUARISMA_LIKELY (x >= this->back())
        {
            const auto to = this->size() - 1;
            return m_[to];
        }

        // Find appropriate interval
        const auto from = this->find_interval(x);

        size_t to = from + 1;

        // Compute normalized distance
        const auto h = this->x_[to] - this->x_[from];
        const auto t = (x - this->x_[from]) / h;

        QUARISMA_CHECK_DEBUG(
            to < this->x_.size(),
            "values ",
            x,
            " falling outside the boundaries of the interpolation cooredinates");
        const auto t_2 = t * t;

        // Compute Hermite basis functions
        //const auto alpha = -2. * t_3 + 3. * t_2;
        //const auto h10   = t_3 - 2. * t_2 + t;
        //const auto h11   = t_3 - t_2;

        const auto dalpha_dt = -6. * t_2 + 6. * t;
        const auto dh10_dt   = 3. * t_2 - 4. * t + 1.;
        const auto dh11_dt   = 3. * t_2 - 2. * t;

        // Interpolate
        return dalpha_dt / h * (this->y_[to] - this->y_[from]) +
               (dh10_dt * m_[from] + dh11_dt * m_[to]);
    };

    void derivative_aad(S value_aad, const T& x, double* state_parameters_aad) const override
    {
        if constexpr (std::is_same_v<S, double>)
        {
            if QUARISMA_UNLIKELY (x <= this->front())
            {
                this->update_state_parameters(
                    value_aad, this->AAD_OFFSET(m_), 0, state_parameters_aad);
                return;
            }
            if (x >= this->back())
            {
                const auto to = this->size() - 1;
                this->update_state_parameters(
                    value_aad, this->AAD_OFFSET(m_), to, state_parameters_aad);
                return;
            }

            // Find appropriate interval
            const auto from = this->find_interval(x);

            size_t to = from + 1;

            // Compute normalized distance
            const auto h = this->x_[to] - this->x_[from];
            const auto t = (x - this->x_[from]) / h;

            QUARISMA_CHECK_DEBUG(
                to < this->x_.size(),
                "values ",
                x,
                " falling outside the boundaries of the interpolation cooredinates");
            const auto t_2 = t * t;

            // Compute Hermite basis functions
            const auto dalpha_dt = -6. * t_2 + 6. * t;
            const auto dh10_dt   = 3. * t_2 - 4. * t + 1.;
            const auto dh11_dt   = 3. * t_2 - 2. * t;

            // Interpolate
            //return dalpha_dt / h * (this->y_[to] - this->y_[from]) + (dh10_dt * m_[from] + dh11_dt * m_[to]);

            // Interpolate
            aad_update_state_parameters_hermite(
                value_aad, from, -dalpha_dt / h, dh10_dt, state_parameters_aad);
            aad_update_state_parameters_hermite(
                value_aad, to, dalpha_dt / h, dh11_dt, state_parameters_aad);
        }
    }

    // Evaluate interpolated value at point x
    S interpolate(const T& x) const override
    {
        if QUARISMA_UNLIKELY (x < this->front())
        {
            const double dx = (double)(x - this->x_[0]);
            return this->y_[0] + m_[0] * dx;
        }
        if QUARISMA_UNLIKELY (x > this->back())
        {
            const auto   to = this->size() - 1;
            const double dx = (double)(x - this->x_[to]);
            return this->y_[to] + m_[to] * dx;
        }

        // Find appropriate interval
        const auto from = this->find_interval(x);

        if (x == this->x_[from])
        {
            return this->y_[from];
        }

        size_t to = from + 1;

        // Compute normalized distance
        const auto h = this->x_[to] - this->x_[from];
        const auto t = (x - this->x_[from]) / h;

        QUARISMA_CHECK_DEBUG(
            to < this->x_.size(),
            "values ",
            x,
            " falling outside the boundaries of the interpolation cooredinates");
        const auto t_2 = t * t;
        const auto t_3 = t_2 * t;

        // Compute Hermite basis functions
        const auto alpha = -2. * t_3 + 3. * t_2;
        const auto h10   = t_3 - 2. * t_2 + t;
        const auto h11   = t_3 - t_2;

        // Interpolate
        return this->y_[from] + alpha * (this->y_[to] - this->y_[from]) +
               (h10 * m_[from] + h11 * m_[to]) * h;
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
            if QUARISMA_UNLIKELY (x < this->front())
            {
                const double dx = (double)(x - this->x_[0]);
                //return this->y_[0] + m_[0] * dx;
                aad_update_state_parameters_hermite(value_aad, 0, 1, dx, state_parameters_aad);
                return;
            }
            if QUARISMA_UNLIKELY (x > this->back())
            {
                const auto   to = this->size() - 1;
                const double dx = (double)(x - this->x_[to]);
                //return this->y_[to] + m_[to] * dx;
                aad_update_state_parameters_hermite(value_aad, to, 1, dx, state_parameters_aad);
                return;
            }

            // Find appropriate interval
            const auto from = this->find_interval(x);

            if (x == this->x_[from])
            {
                //return this->y_[from];
                this->update_state_parameters(
                    value_aad, this->AAD_OFFSET(y_), from, state_parameters_aad);
                return;
            }

            size_t to = from + 1;

            // Compute normalized distance
            const auto h = this->x_[to] - this->x_[from];
            const auto t = (x - this->x_[from]) / h;

            QUARISMA_CHECK_DEBUG(
                to < this->x_.size(),
                "values ",
                x,
                " falling outside the boundaries of the interpolation cooredinates");
            const auto t_2 = t * t;
            const auto t_3 = t_2 * t;

            // Compute Hermite basis functions
            const auto alpha = -2. * t_3 + 3. * t_2;
            const auto h10   = t_3 - 2. * t_2 + t;
            const auto h11   = t_3 - t_2;

            // Interpolate
            //return this->y_[from] + alpha * (this->y_[to] - this->y_[from]) + (h10 * m_[from] + h11 * m_[to]) * h;
            aad_update_state_parameters_hermite(
                value_aad, from, 1. - alpha, h10 * h, state_parameters_aad);
            aad_update_state_parameters_hermite(
                value_aad, to, alpha, h11 * h, state_parameters_aad);
        }
    };

    void initialize() override
    {
        size_t n = this->x_.size();
        if constexpr (is_matrix<Container>::value)
        {
            m_ = Container(this->y_.rows(), this->y_.columns());
        }
        else if constexpr (is_vector<Container>::value)
        {
            m_ = Container(n);
        }
        else
        {
            m_.resize(n);
        }
        for (size_t i = 1; i < n - 1; ++i)
        {
            const auto h1 = this->x_[i] - this->x_[i - 1];
            const auto h2 = this->x_[i + 1] - this->x_[i];
            const auto s1 = (this->y_[i] - this->y_[i - 1]) / h1;
            const auto s2 = (this->y_[i + 1] - this->y_[i]) / h2;

            m_[i] = 0.5 * (s1 + s2);
        }

        m_[0]     = (this->y_[1] - this->y_[0]) / (this->x_[1] - this->x_[0]);
        m_[n - 1] = (this->y_[n - 1] - this->y_[n - 2]) / (this->x_[n - 1] - this->x_[n - 2]);

        AAD_REGISTER_PARAMETER(m_, m_.size());
    };

    void finalize_aad(double* state_parameters_aad) const override
    {
        if constexpr (std::is_arithmetic<S>::value)
        {
            size_t      n = this->x_.size();
            const auto* m_aad =
                this->get_state_parameters_aad(AAD_OFFSET(m_), 0, state_parameters_aad);

            //m_[n - 1] = (this->y_[n - 1] - this->y_[n - 2]) / (this->x_[n - 1] - this->x_[n - 2]);
            auto tmp_aad = m_aad[n - 1] / (this->x_[n - 1] - this->x_[n - 2]);
            this->update_state_parameters(
                tmp_aad, this->AAD_OFFSET(y_), n - 1, state_parameters_aad);
            this->update_state_parameters(
                -tmp_aad, this->AAD_OFFSET(y_), n - 2, state_parameters_aad);

            for (size_t i = n - 2; i > 0; --i)
            {
                const auto h1 = this->x_[i] - this->x_[i - 1];
                const auto h2 = this->x_[i + 1] - this->x_[i];
                //const auto s1 = (this->y_[i] - this->y_[i - 1]) / h1;
                //const auto s2 = (this->y_[i + 1] - this->y_[i]) / h2;

                // Weighted harmonic mean of slopes
                //m_[i] = 0.5*(s1+s2);
                tmp_aad     = m_aad[i] * 0.5;
                auto s1_aad = tmp_aad / h1;
                auto s2_aad = tmp_aad / h2;

                this->update_state_parameters(
                    s2_aad, this->AAD_OFFSET(y_), i + 1, state_parameters_aad);
                this->update_state_parameters(
                    s1_aad - s2_aad, this->AAD_OFFSET(y_), i, state_parameters_aad);
                this->update_state_parameters(
                    -s1_aad, this->AAD_OFFSET(y_), i - 1, state_parameters_aad);
            }

            //m_[0] = (this->y_[1] - this->y_[0]) / (this->x_[1] - this->x_[0]);
            tmp_aad = m_aad[0] / (this->x_[1] - this->x_[0]);
            this->update_state_parameters(tmp_aad, this->AAD_OFFSET(y_), 1, state_parameters_aad);
            this->update_state_parameters(-tmp_aad, this->AAD_OFFSET(y_), 0, state_parameters_aad);
        }
    }

private:
    Container m_;  // derivatives
    AAD_STATE_PARAMETERS(m_);

    interpolator_cubic_hermite() = default;

    QUARISMA_SERIALIZATION_TEMPLATE(interpolator_cubic_hermite, x_, y_, m_);
};
}  // namespace quarisma