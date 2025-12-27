#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "common/tridiagonal_operations.h"
#include "common/vectorization_type_traits.h"
#include "interpolator.h"
#include "terminals/matrix.h"
#include "terminals/vector.h"

namespace quarisma
{
enum class cubic_spline_condition_enum : int
{
    NOT_A_KNOT,
    FIRST_DERIVATIVE,
    SECOND_DERIVATIVE
};

template <typename Container, typename T, typename S>
class interpolator_cubic_spline : public interpolator<Container, T, S>
{
    inline void update_aad(
        size_t idx, S value_aad, double dx_i, double dx, double* state_parameters_aad) const
    {
        if constexpr (std::is_arithmetic<S>::value)
        {
            QUARISMA_CHECK_FINITE_DEBUG(value_aad);
            QUARISMA_CHECK_FINITE_DEBUG(value_aad * dx);
            QUARISMA_CHECK_FINITE_DEBUG(value_aad * dx_i * dx);
            QUARISMA_CHECK_FINITE_DEBUG(value_aad * dx_i * dx_i * dx);

            this->update_state_parameters(
                value_aad, this->AAD_OFFSET(y_), idx, state_parameters_aad);
            this->update_state_parameters(
                value_aad * dx, this->AAD_OFFSET(r_), idx, state_parameters_aad);
            this->update_state_parameters(
                value_aad * dx_i * dx, this->AAD_OFFSET(q_), idx, state_parameters_aad);
            this->update_state_parameters(
                value_aad * dx_i * dx_i * dx, this->AAD_OFFSET(p_), idx, state_parameters_aad);
        }
        else
        {
            const auto offset_idx = idx * value_aad.size();
            const auto dx2        = dx_i * dx;
            const auto dx3        = dx_i * dx_i * dx;
            size_t     i          = 0;
            for (const auto& itr : value_aad)
            {
                this->update_state_parameters(
                    itr, this->AAD_OFFSET(y_), offset_idx + i, state_parameters_aad);
                this->update_state_parameters(
                    itr * dx, this->AAD_OFFSET(r_), offset_idx + i, state_parameters_aad);
                this->update_state_parameters(
                    itr * dx2, this->AAD_OFFSET(q_), offset_idx + i, state_parameters_aad);
                this->update_state_parameters(
                    itr * dx3, this->AAD_OFFSET(p_), offset_idx + i, state_parameters_aad);
                i++;
            }
        }
    }

    inline void build_initial_condition(Container& q) const
    {
        const size_t n = this->x_.size();

        auto h_i_1 = this->x_[1] - this->x_[0];
        for (size_t i = 1; i < n - 1; ++i)
        {
            const auto h_i = this->x_[i + 1] - this->x_[i];

            q[i] = 3.0 * ((this->y_[i + 1] - this->y_[i]) / h_i -
                          (this->y_[i] - this->y_[i - 1]) / h_i_1);

            h_i_1 = h_i;
        }

        switch (left_condition_)
        {
        case cubic_spline_condition_enum::FIRST_DERIVATIVE:
        {
            const auto h = (this->x_[1] - this->x_[0]);

            q[0] = 3.0 * ((this->y_[1] - this->y_[0]) / h - left_value_);
            break;
        }
        case cubic_spline_condition_enum::SECOND_DERIVATIVE:
        {
            q[0] = left_value_;
            break;
        }
        case cubic_spline_condition_enum::NOT_A_KNOT:
        {
            q[0] = q[1];
        }
        }

        switch (right_condition_)
        {
        case cubic_spline_condition_enum::FIRST_DERIVATIVE:
        {
            const auto h = (this->x_[n - 1] - this->x_[n - 2]);
            q[n - 1]     = 3.0 * (right_value_ - (this->y_[n - 1] - this->y_[n - 2]) / h);
            break;
        }
        case cubic_spline_condition_enum::SECOND_DERIVATIVE:
        {
            q[n - 1] = right_value_;
            break;
        }
        case cubic_spline_condition_enum::NOT_A_KNOT:
        {
            q[n - 1] = q[n - 2];
        }
        }
    }

    inline void build_initial_condition_aad(double* q_aad, double* state_parameters_aad) const
    {
        const size_t n = this->x_.size();

        auto* y_aad = this->get_state_parameters_aad(this->AAD_OFFSET(y_), 0, state_parameters_aad);
        switch (left_condition_)
        {
        case cubic_spline_condition_enum::FIRST_DERIVATIVE:
        {
            const auto h = (this->x_[1] - this->x_[0]);

            //q[0] = 3.0 * ((this->y_[1] - this->y_[0]) / h - left_value_);
            auto tmp_aad = 3. * q_aad[0];
            this->update_state_parameters(
                -tmp_aad, this->AAD_OFFSET(left_value_), 0, state_parameters_aad);

            tmp_aad /= h;
            this->update_state_parameters(tmp_aad, this->AAD_OFFSET(y_), 1, state_parameters_aad);
            this->update_state_parameters(-tmp_aad, this->AAD_OFFSET(y_), 0, state_parameters_aad);
            break;
        }
        case cubic_spline_condition_enum::SECOND_DERIVATIVE:
        {
            //q[0] = left_value_;
            this->update_state_parameters(
                q_aad[0], this->AAD_OFFSET(left_value_), 0, state_parameters_aad);
            break;
        }
        case cubic_spline_condition_enum::NOT_A_KNOT:
        {
            //q[0] = q[1];
            q_aad[1] += q_aad[0];
        }
        }

        switch (right_condition_)
        {
        case cubic_spline_condition_enum::FIRST_DERIVATIVE:
        {
            const auto h = (this->x_[n - 1] - this->x_[n - 2]);
            //q[n - 1]     = 3.0 * (right_value_ - (this->y_[n - 1] - this->y_[n - 2]) / h);
            auto tmp_aad = 3. * q_aad[n - 1];
            this->update_state_parameters(
                tmp_aad, this->AAD_OFFSET(right_value_), 0, state_parameters_aad);

            tmp_aad /= h;
            this->update_state_parameters(
                -tmp_aad, this->AAD_OFFSET(y_), n - 1, state_parameters_aad);
            this->update_state_parameters(
                tmp_aad, this->AAD_OFFSET(y_), n - 2, state_parameters_aad);
            break;
        }
        case cubic_spline_condition_enum::SECOND_DERIVATIVE:
        {
            //q[n - 1] = right_value_;
            this->update_state_parameters(
                q_aad[n - 1], this->AAD_OFFSET(right_value_), 0, state_parameters_aad);
            break;
        }
        case cubic_spline_condition_enum::NOT_A_KNOT:
        {
            //q[n - 1] = q[n - 2];
            q_aad[n - 2] += q_aad[n - 1];
        }
        }
        auto h_i = this->x_[n - 1] - this->x_[n - 2];
        for (size_t i = n - 2; i > 0; --i)
        {
            const auto h_i_1 = this->x_[i] - this->x_[i - 1];

            auto tmp_aad = 3. * q_aad[i];
            if (!is_almost_zero(tmp_aad))
            {
                auto tmp_aad_i   = tmp_aad / h_i;
                auto tmp_aad_i_1 = tmp_aad / h_i_1;

                y_aad[i + 1] += tmp_aad_i;
                y_aad[i] -= tmp_aad_i_1 + tmp_aad_i;
                y_aad[i - 1] += tmp_aad_i_1;
            }

            h_i = h_i_1;
        }
    }

    inline void build_transition_matrix(matrix<double>& A) const
    {
        size_t n  = A.columns();
        auto*  A0 = A[0].data();
        auto*  A1 = A[1].data();
        auto*  A2 = A[2].data();

        auto h_i_1 = this->x_[1] - this->x_[0];
        for (size_t i = 1; i < n - 1; ++i)
        {
            const auto h_i = this->x_[i + 1] - this->x_[i];

            A0[i] = h_i_1;
            A1[i] = 2 * (h_i + h_i_1);
            A2[i] = h_i;

            h_i_1 = h_i;
        }

        switch (left_condition_)
        {
        case cubic_spline_condition_enum::FIRST_DERIVATIVE:
        {
            const auto h = (this->x_[1] - this->x_[0]);

            A0[0] = 0.;
            A1[0] = 2.0 * h;
            A2[0] = h;
            break;
        }
        case cubic_spline_condition_enum::SECOND_DERIVATIVE:
        {
            A0[0] = 0.;
            A1[0] = 2.;
            A2[0] = 0.;
            break;
        }
        case cubic_spline_condition_enum::NOT_A_KNOT:
        {
            const auto h0 = (this->x_[1] - this->x_[0]);
            const auto h1 = (this->x_[2] - this->x_[1]);

            const auto w = h1 * h1 / h0;

            //tex:
            // $$\frac{q_1-q_0}{h_0}=\frac{q_2-q_1}{h_1}$$
            // $$h_1q_2-q_1(h_1+\frac{h^2_1}{h_0})+\frac{h^2_1}{h_0}q_0=0$$
            // $$h_0q_0+2(h_0+h_1)q_1+h_1q_2=a_1$$
            // $$(h_0-\frac{h^2_1}{h_0})q_0+(2h_0+3h_1+\frac{h^2_1}{h_0})q_1=a_1$$

            A0[0] = 0;
            A1[0] = h0 - w;
            A2[0] = 2. * h0 + 3. * h1 + w;
        }
        }

        switch (right_condition_)
        {
        case cubic_spline_condition_enum::FIRST_DERIVATIVE:
        {
            const auto h = (this->x_[n - 1] - this->x_[n - 2]);
            A0[n - 1]    = h;
            A1[n - 1]    = 2.0 * h;
            A2[n - 1]    = 0;
            break;
        }
        case cubic_spline_condition_enum::SECOND_DERIVATIVE:
        {
            A0[n - 1] = 0.;
            A1[n - 1] = 2.;
            A2[n - 1] = 0.;
            break;
        }
        case cubic_spline_condition_enum::NOT_A_KNOT:
        {
            const auto h0 = (this->x_[n - 1] - this->x_[n - 2]);
            const auto h1 = (this->x_[n - 2] - this->x_[n - 3]);

            //tex:
            // $$\frac{q_{n-1}-q_{n-2}}{h_0}=\frac{q_{n-2}-q_{n-3}}{h_1}$$
            // $$h_1q_{n-3}-q_{n-2}(h_1+\frac{h^2_1}{h_0})+\frac{h^2_1}{h_0}q_{n-1}=0$$
            // $$h_0q_{n-1}+2(h_0+h_1)q_{n-2}+h_1q_{n-3}=a_{n-2}$$
            // $$(h_0-\frac{h^2_1}{h_0})q_{n-1}+(2h_0+3h_1+\frac{h^2_1}{h_0})q_{n-2}=a_{n-2}$$

            const auto w = h1 * h1 / h0;
            A0[n - 1]    = 2 * h0 + 3. * h1 + w;
            A1[n - 1]    = h0 - w;
            A2[n - 1]    = 0;
        }
        }
    }

    //tex:
    //$$y(t)=y_i+p_i(t-t_i)^3+q_i(t-t_i)^2+r_i(t-t_i)$$
    //$$y_{t_{i+1}}=y_{t_{i}}$$
    //$$y'_{t_{i+1}}=y'_{t_{i}}$$
    //$$y''_{t_{i+1}}=y''_{t_{i}}$$

public:
    interpolator_cubic_spline(
        std::vector<T>              x_points,
        Container                   y_points,
        cubic_spline_condition_enum left_condition = cubic_spline_condition_enum::SECOND_DERIVATIVE,
        S                           left_value     = 0,
        cubic_spline_condition_enum right_condition =
            cubic_spline_condition_enum::SECOND_DERIVATIVE,
        S right_value = 0)
        : interpolator<Container, T, S>(std::move(x_points), std::move(y_points)),
          left_condition_(left_condition),
          right_condition_(right_condition),
          left_value_(left_value),
          right_value_(right_value)
    {
        initialize();
    }

    void initialize() override
    {
        const size_t n = this->x_.size();
        A_             = matrix<double>(3, n);

        if constexpr (is_matrix<Container>::value)
        {
            p_ = Container(this->y_.rows() - 1, this->y_.columns());
            q_ = Container(this->y_.rows(), this->y_.columns());
            r_ = Container(this->y_.rows() - 1, this->y_.columns());
        }
        else if constexpr (is_vector<Container>::value)
        {
            p_ = Container(n - 1);
            q_ = Container(n);
            r_ = Container(n - 1);
        }
        else
        {
            p_.resize(n - 1);
            q_.resize(n);
            r_.resize(n - 1);
        }

        AAD_REGISTER_PARAMETER(p_, p_.size());
        AAD_REGISTER_PARAMETER(q_, q_.size());
        AAD_REGISTER_PARAMETER(r_, r_.size());
        AAD_REGISTER_PARAMETER(right_value_, 1);
        AAD_REGISTER_PARAMETER(left_value_, 1);

        build_transition_matrix(A_);
        tridiagonal_operations::decomposition(A_, 1, n, 1);
        build_initial_condition(q_);

        if constexpr (std::is_arithmetic<S>::value)
        {
            quarisma::vector x(q_.data(), q_.size());
            tridiagonal_operations::solve_decomposed(x, A_, 1, n, 1);
        }
        else
        {
            //fixme: support only natural spline!
            q_[0]     = 0.;
            q_[n - 1] = 0.;

            tridiagonal_operations::solve_decomposed_vectorised(q_, A_, 1, n, 1);
        }

        for (size_t i = 0; i < n - 1; ++i)
        {
            auto dx = this->x_[i + 1] - this->x_[i];
            r_[i]   = (this->y_[i + 1] - this->y_[i]) / dx - dx * (2 * q_[i] + q_[i + 1]) / 3.;
            p_[i]   = (q_[i + 1] - q_[i]) / (3. * dx);
        }
    };

    void finalize_aad(double* state_parameters_aad) const override
    {
        const size_t n = this->x_.size();
        if constexpr (std::is_arithmetic<S>::value)
        {
            auto* r_aad = this->get_state_parameters_aad(AAD_OFFSET(r_), 0, state_parameters_aad);
            auto* p_aad = this->get_state_parameters_aad(AAD_OFFSET(p_), 0, state_parameters_aad);
            auto* q_aad = this->get_state_parameters_aad(AAD_OFFSET(q_), 0, state_parameters_aad);
            auto* y_aad =
                this->get_state_parameters_aad(this->AAD_OFFSET(y_), 0, state_parameters_aad);
            double q_aad_i = 0;
            double y_aad_i = 0;
            for (int i = (int)n - 2; i >= 0; --i)
            {
                const auto dx = this->x_[i + 1] - this->x_[i];
                //r_[i]   = (this->y_[i + 1] - this->y_[i]) / dx - dx / 3. * (2 * q_[i] + q_[i + 1]);
                //p_[i]   = (q_[i + 1] - q_[i]) / (3. * dx);

                auto tmp_q_aad = p_aad[i] / (3. * dx);
                auto tmp_r_aad = r_aad[i] * dx / 3.;

                q_aad[i + 1] += q_aad_i + tmp_q_aad - tmp_r_aad;
                q_aad_i = -tmp_q_aad - 2. * tmp_r_aad;

                tmp_r_aad = r_aad[i] / dx;

                y_aad[i + 1] += y_aad_i + tmp_r_aad;
                y_aad_i = -tmp_r_aad;
            }
            q_aad[0] += q_aad_i;
            y_aad[0] += y_aad_i;

            vector<double> x_aad(q_aad, q_.size());
            tridiagonal_operations::solve_decomposed_aad(x_aad, A_, 1, n, 1);
            build_initial_condition_aad(q_aad, state_parameters_aad);
            x_aad = 0.;
            vector<double> x1_aad(r_aad, r_.size());
            x1_aad = 0.;
            vector<double> x2_aad(p_aad, p_.size());
            x2_aad = 0.;
        }
    }

    S interpolate(const T& x) const override
    {
        if QUARISMA_UNLIKELY (x <= this->front())
        {
            const auto dx    = x - this->x_[0];
            S          value = this->y_[0] + r_[0] * dx;
            if constexpr (std::is_arithmetic<S>::value)
            {
                QUARISMA_CHECK_FINITE_DEBUG(value);
            }

            return std::move(value);
        }
        if QUARISMA_UNLIKELY (x >= this->back())
        {
            const auto to    = this->size() - 1;
            const auto dx    = x - this->x_[to];
            const auto dx_to = this->x_[to] - this->x_[to - 1];

            S value = this->y_[to] +
                      (r_[to - 1] + (2. * q_[to - 1] + 3. * p_[to - 1] * dx_to) * dx_to) * dx;
            if constexpr (std::is_arithmetic<S>::value)
            {
                QUARISMA_CHECK_FINITE_DEBUG(value);
            }
            return std::move(value);
        }

        const size_t from = this->find_interval(x);

        if (x == this->x_[from])
        {
            return std::move(this->y_[from]);
        }

        const auto dx    = x - this->x_[from];
        S          value = this->y_[from] + (r_[from] + (q_[from] + p_[from] * dx) * dx) * dx;

        if constexpr (std::is_arithmetic<S>::value)
        {
            QUARISMA_CHECK_FINITE_DEBUG(value);
        }

        return std::move(value);
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
            if QUARISMA_UNLIKELY (x <= this->front())
            {
                const auto dx = x - this->x_[0];
                //return this->y_[0] + r_[0]* dx;
                this->update_state_parameters(
                    value_aad, this->AAD_OFFSET(y_), 0, state_parameters_aad);
                this->update_state_parameters(
                    value_aad * dx, this->AAD_OFFSET(r_), 0, state_parameters_aad);
                return;
            }
            if QUARISMA_UNLIKELY (x >= this->back())
            {
                const auto to    = this->size() - 1;
                const auto dx    = x - this->x_[to];
                const auto dx_to = this->x_[to] - this->x_[to - 1];

                //return this->y_[to] +(r_[to - 1] + (2. * q_[to - 1] + 3. * p_[to - 1] * dx_to) * dx_to) * dx;

                QUARISMA_CHECK_FINITE_DEBUG(value_aad);
                QUARISMA_CHECK_FINITE_DEBUG(value_aad * dx);
                QUARISMA_CHECK_FINITE_DEBUG(value_aad * dx_to * dx);
                QUARISMA_CHECK_FINITE_DEBUG(value_aad * dx_to * dx_to * dx);

                this->update_state_parameters(
                    value_aad, this->AAD_OFFSET(y_), to, state_parameters_aad);
                this->update_state_parameters(
                    value_aad * dx, this->AAD_OFFSET(r_), to - 1, state_parameters_aad);
                this->update_state_parameters(
                    2. * value_aad * dx_to * dx,
                    this->AAD_OFFSET(q_),
                    to - 1,
                    state_parameters_aad);
                this->update_state_parameters(
                    3. * value_aad * dx_to * dx_to * dx,
                    this->AAD_OFFSET(p_),
                    to - 1,
                    state_parameters_aad);

                return;
            }

            const size_t from = this->find_interval(x);

            if (x == this->x_[from])
            {
                //return std::move(this->y_[from]);
                QUARISMA_CHECK_FINITE_DEBUG(value_aad);

                this->update_state_parameters(
                    value_aad, this->AAD_OFFSET(y_), from, state_parameters_aad);

                return;
            }

            const auto dx = x - this->x_[from];
            //return this->y_[from] + r_[from] * dx + q_[from] * dx * dx + p_[from] * dx * dx * dx;
            update_aad(from, value_aad, dx, dx, state_parameters_aad);
        }
    };

    S first_derivative_aad(double* state_parameters_aad) const override
    {
        return *this->get_state_parameters_aad(AAD_OFFSET(left_value_), 0, state_parameters_aad);
    }

    S derivative(const T& x) const override
    {
        if QUARISMA_UNLIKELY (x <= this->x_.front())
        {
            return r_[0];
        }
        if (x >= this->x_.back())
        {
            const auto to    = this->size() - 1;
            const auto dx_to = this->x_[to] - this->x_[to - 1];

            return (r_[to - 1] + (2. * q_[to - 1] + 3. * p_[to - 1] * dx_to) * dx_to);
        }

        const size_t i = this->find_interval(x);

        const auto dx = x - this->x_[i];
        return r_[i] + 2.0 * q_[i] * dx + 3.0 * p_[i] * dx * dx;
    }

    void derivative_aad(S value_aad, const T& x, double* state_parameters_aad) const override
    {
        if constexpr (std::is_same_v<S, double>)
        {
            if QUARISMA_UNLIKELY (x <= this->x_.front())
            {
                //return r_[0];
                this->update_state_parameters(
                    value_aad, this->AAD_OFFSET(r_), 0, state_parameters_aad);
                return;
            }
            if (x >= this->x_.back())
            {
                const auto to    = this->size() - 1;
                const auto dx_to = this->x_[to] - this->x_[to - 1];

                //return (r_[to - 1] + (2. * q_[to - 1] + 3. * p_[to - 1] * dx_to) * dx_to);
                this->update_state_parameters(
                    value_aad, this->AAD_OFFSET(r_), to - 1, state_parameters_aad);
                this->update_state_parameters(
                    2. * value_aad * dx_to, this->AAD_OFFSET(q_), to - 1, state_parameters_aad);
                this->update_state_parameters(
                    3. * value_aad * dx_to * dx_to,
                    this->AAD_OFFSET(p_),
                    to - 1,
                    state_parameters_aad);
                return;
            }

            const size_t i = this->find_interval(x);

            const auto dx = x - this->x_[i];
            //return r_[i] + 2.0 * q_[i] * dx + 3.0 * p_[i] * dx * dx;

            this->update_state_parameters(value_aad, this->AAD_OFFSET(r_), i, state_parameters_aad);
            this->update_state_parameters(
                2. * value_aad * dx, this->AAD_OFFSET(q_), i, state_parameters_aad);
            this->update_state_parameters(
                3. * value_aad * dx * dx, this->AAD_OFFSET(p_), i, state_parameters_aad);
        }
    }

    S second_derivative(const T& x) const
    {
        const size_t i = this->find_interval(x);

        if (i == 0 && x <= this->x_.front())
        {
            return 2.0 * q_[0];
        }
        if (i >= this->x_.size() - 1 && x >= this->x_.back())
        {
            return 2.0 * q_.back();
        }

        const auto dx = x - this->x_[i];
        return 2.0 * q_[i] + 6.0 * r_[i] * dx;
    }

    interpolator_cubic_spline() = default;

private:
    Container p_;  // coefficients b (linear terms)
    Container q_;  // coefficients c (quadratic terms)
    Container r_;  // coefficients d (cubic terms)

    AAD_STATE_PARAMETERS(p_);
    AAD_STATE_PARAMETERS(q_);
    AAD_STATE_PARAMETERS(r_);

    cubic_spline_condition_enum left_condition_;
    cubic_spline_condition_enum right_condition_;
    S                           left_value_;
    S                           right_value_;
    AAD_STATE_PARAMETERS(left_value_);
    AAD_STATE_PARAMETERS(right_value_);

    matrix<double> A_;

    QUARISMA_SERIALIZATION_TEMPLATE(
        interpolator_cubic_spline,
        x_,
        y_,
        left_condition_,
        right_condition_,
        left_value_,
        right_value_);
};

}  // namespace quarisma