#pragma once

#include "detail/taylor_series.h"
#include "solvers/api/problem.h"
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace solverslib
{
namespace detail
{
template <class Model, class Scalar>
void evaluate_rnc_model(const Model& model, const Scalar* x, Scalar* r)
{
    if constexpr (std::is_same_v<decltype(model(x, r)), bool>)
    {
        if (!model(x, r))
            throw std::domain_error("RNC model evaluation failed");
    }
    else
        model(x, r);
}
}  // namespace detail

// Dense Taylor-over-forward AD adapter. One model sweep per parameter yields
// the base-point derivatives of every Taylor coefficient. This implements the
// same J_k^T R_{n-k} contractions as a VJP, with a dense forward-mode cost.
// Model signature: template<class T> bool/void operator()(const T*, T*) const.
// Use ADL math (e.g. using std::exp; exp(x)), not qualified std::exp(x).
template <class Model> rnc_derivative_function make_rnc_derivatives(Model model, size_t n, size_t m)
{
    if (n == 0 || m == 0)
        throw std::invalid_argument("RNC dimensions must be positive");
    return [model = std::move(model), n, m](const vector_type& base,
               const std::vector<vector_type>&                 coefficients,
               int                                             order,
               rnc_curve_derivatives&                          out)
    {
        if (base.size() != static_cast<index_type>(n) || order < 1 || order > 4 ||
            coefficients.size() >= static_cast<size_t>(order))
            throw std::invalid_argument("Invalid RNC Taylor derivative request");
        for (const auto& c : coefficients)
            if (c.size() != base.size())
                throw std::invalid_argument("RNC coefficient dimension mismatch");
        using base_dual                 = detail::taylor_series<double, 1>;
        using series                    = detail::taylor_series<base_dual, 4>;
        constexpr double factorial[]    = {1., 1., 2., 6., 24.};
        const int        jacobian_order = std::max(0, order - 2);
        out.residual.assign(order + 1, vector_type::Zero(m));
        out.jacobian.assign(jacobian_order + 1, matrix_type::Zero(m, n));
        for (size_t column = 0; column < n; ++column)
        {
            std::vector<series> x(n), r(m);
            for (size_t j = 0; j < n; ++j)
            {
                x[j].coefficient[0].coefficient[0] = base[j];
                x[j].coefficient[0].coefficient[1] = column == j ? 1. : 0.;
                for (size_t k = 0; k < coefficients.size(); ++k)
                    x[j].coefficient[k + 1].coefficient[0] = coefficients[k][j] / factorial[k + 1];
            }
            detail::evaluate_rnc_model(model, x.data(), r.data());
            for (size_t i = 0; i < m; ++i)
            {
                if (column == 0)
                    for (int k = 0; k <= order; ++k)
                        out.residual[k][i] = factorial[k] * r[i].coefficient[k].coefficient[0];
                for (int k = 0; k <= jacobian_order; ++k)
                    out.jacobian[k](i, column) = factorial[k] * r[i].coefficient[k].coefficient[1];
            }
        }
    };
}

// Complete least-squares problem with Taylor derivatives for RNC-LM and an
// ordinary Jacobian for other backends. Does not require Ceres.
template <class Model> api::least_squares_problem rnc_least_squares(Model model, size_t n, size_t m)
{
    api::least_squares_problem problem;
    problem.num_parameters = n;
    problem.num_residuals  = m;
    problem.residuals      = [model = model, n, m](const vector_type& x, vector_type& r)
    {
        if (x.size() != static_cast<index_type>(n))
            throw std::invalid_argument("RNC parameter dimension mismatch");
        r.resize(m);
        detail::evaluate_rnc_model(model, x.data(), r.data());
    };
    // One provider serves RNC-LM (curve derivatives) and, through its order-1
    // evaluation, every other backend's ordinary Jacobian.
    problem.set_curve_derivatives(make_rnc_derivatives(std::move(model), n, m),
        api::derivative_mode::automatic_differentiation);
    return problem;
}
}  // namespace solverslib
