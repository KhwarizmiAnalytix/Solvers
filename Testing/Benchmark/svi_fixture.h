#pragma once

#include "api/dispatch.h"
#include "solvers/integrations/rnc_autodiff.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

// Raw-SVI model and EURO STOXX 50 calibration data from Ferhati (2020).
// Shared by the native and external benchmark executables.

namespace solverslib::svi
{
// Ferhati (2020) EURO STOXX 50 slice: Tables 3.1 & 3.2, T = 1.01 Y.
struct paper_quote
{
    double strike;
    double call;
    double put;
    double market_total_variance;
};

constexpr std::array<paper_quote, 13> kPaperQuotes{{
    {2068.48, 1268.59, 7.25, 0.06249},
    {2413.23, 936.83, 21.56, 0.05000},
    {2757.98, 621.99, 52.79, 0.03780},
    {3016.54, 407.05, 97.39, 0.02964},
    {3585.37, 77.18, 338.53, 0.01662},
    {3964.59, 15.10, 657.11, 0.01501},
    {4481.71, 1.87, 1162.98, 0.01694},
    {4998.83, 0.35, 1680.55, 0.02018},
    {5688.33, 0.05, 2372.38, 0.02462},
    {6033.07, 0.02, 2718.41, 0.02678},
    {6377.82, 0.01, 3064.46, 0.02892},
    {6722.57, 0.01, 3410.52, 0.03100},
    {6894.94, 0.00, 3583.55, 0.03207},
}};

constexpr std::size_t kSviN = kPaperQuotes.size();

// w(k; a,b,rho,m,sigma) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
inline double svi_model(double k, double a, double b, double rho, double m, double sigma) noexcept
{
    const double c = k - m;
    return a + b * (rho * c + std::sqrt(c * c + sigma * sigma));
}

// Pre-computed problem data (log-moneyness, market variances, bounds, x0).
struct svi_data
{
    std::array<double, kSviN> log_moneyness{};
    std::array<double, kSviN> market_variance{};
    std::array<double, 5>     lower{};
    std::array<double, 5>     upper{};
    vector_type               x0;
};

// Put-call parity regression to recover the implied forward.
static double infer_forward()
{
    double mean_k = 0.0, mean_parity = 0.0;
    for (const auto& q : kPaperQuotes)
    {
        mean_k += q.strike;
        mean_parity += q.call - q.put;
    }
    mean_k /= static_cast<double>(kSviN);
    mean_parity /= static_cast<double>(kSviN);

    double cov = 0.0, var = 0.0;
    for (const auto& q : kPaperQuotes)
    {
        const double dk = q.strike - mean_k;
        cov += dk * (q.call - q.put - mean_parity);
        var += dk * dk;
    }
    const double beta = cov / var;
    return (mean_parity - beta * mean_k) / (-beta);
}

static const svi_data& svi_problem_data()
{
    static const svi_data data = []()
    {
        svi_data     d;
        const double forward = infer_forward();

        double min_w = 1e30, max_w = 0.0, min_k = 1e30, max_k = -1e30;
        for (std::size_t i = 0; i < kSviN; ++i)
        {
            d.log_moneyness[i]   = std::log(kPaperQuotes[i].strike / forward);
            d.market_variance[i] = kPaperQuotes[i].market_total_variance;
            min_w                = std::min(min_w, d.market_variance[i]);
            max_w                = std::max(max_w, d.market_variance[i]);
            min_k                = std::min(min_k, d.log_moneyness[i]);
            max_k                = std::max(max_k, d.log_moneyness[i]);
        }

        // Box bounds: a in (0, max_w], b in (0,1], rho in (-1,1], m in [2*min_k, 2*max_k], sigma in
        // (0,1]
        d.lower = {1e-5, 0.001, -1.0, 2.0 * min_k, 0.01};
        d.upper = {max_w, 1.0, 1.0, 2.0 * max_k, 1.0};

        // Paper equation 3.17 initial guess
        d.x0.resize(5);
        d.x0 << 0.5 * min_w, 0.1, -0.5, 0.1, 0.1;
        return d;
    }();
    return data;
}

// ---------------------------------------------------------------------------
// Residuals: r_i = svi_model(k_i, p) - w_market[i]
// ---------------------------------------------------------------------------

inline void svi_residuals(const vector_type& p, vector_type& r)
{
    const auto& d = svi_problem_data();
    r.resize(static_cast<int>(kSviN));
    for (std::size_t i = 0; i < kSviN; ++i)
    {
        r[static_cast<int>(i)] =
            svi_model(d.log_moneyness[i], p[0], p[1], p[2], p[3], p[4]) - d.market_variance[i];
    }
}

// Analytic Jacobian: dr_i/dp_j
inline void svi_jacobian(const vector_type& p, matrix_type& j)
{
    const auto& d = svi_problem_data();
    j.resize(static_cast<int>(kSviN), 5);
    for (std::size_t i = 0; i < kSviN; ++i)
    {
        const double c    = d.log_moneyness[i] - p[3];
        const double root = std::sqrt(c * c + p[4] * p[4]);
        const int    row  = static_cast<int>(i);
        j(row, 0)         = 1.0;
        j(row, 1)         = p[2] * c + root;
        j(row, 2)         = p[1] * c;
        j(row, 3)         = p[1] * (-p[2] - c / root);
        j(row, 4)         = p[1] * p[4] / root;
    }
}

inline double svi_objective(const vector_type& p)
{
    vector_type r;
    svi_residuals(p, r);
    return 0.5 * r.squaredNorm();
}

inline void svi_gradient(const vector_type& p, vector_type& g)
{
    vector_type r;
    matrix_type j;
    svi_residuals(p, r);
    svi_jacobian(p, j);
    g = j.transpose() * r;
}

inline api::least_squares_problem make_svi_ls(bool with_jacobian)
{
    const auto&                d = svi_problem_data();
    api::least_squares_problem problem;
    problem.num_parameters = 5;
    problem.num_residuals  = kSviN;
    problem.residuals      = svi_residuals;
    problem.bounds.lower   = {d.lower[0], d.lower[1], d.lower[2], d.lower[3], d.lower[4]};
    problem.bounds.upper   = {d.upper[0], d.upper[1], d.upper[2], d.upper[3], d.upper[4]};
    if (with_jacobian)
        problem.jacobian = svi_jacobian;
    return problem;
}

inline api::optimization_problem make_svi_opt(bool with_gradient)
{
    const auto&               d = svi_problem_data();
    api::optimization_problem problem;
    problem.num_parameters = 5;
    problem.objective      = svi_objective;
    problem.bounds.lower   = {d.lower[0], d.lower[1], d.lower[2], d.lower[3], d.lower[4]};
    problem.bounds.upper   = {d.upper[0], d.upper[1], d.upper[2], d.upper[3], d.upper[4]};
    if (with_gradient)
        problem.gradient = svi_gradient;
    return problem;
}

// ---------------------------------------------------------------------------
// Exact Taylor derivatives along base + sum(c_q*t^q/q!), including order 4.
// The adapter converts normalized series coefficients to ordinary derivatives.
// ---------------------------------------------------------------------------
struct svi_rnc_model
{
    template <class T> void operator()(const T* p, T* r) const
    {
        using std::sqrt;
        const auto& d = svi_problem_data();
        for (std::size_t i = 0; i < kSviN; ++i)
        {
            const T c = T(d.log_moneyness[i]) - p[3];
            r[i] = p[0] + p[1] * (p[2] * c + sqrt(c * c + p[4] * p[4])) - T(d.market_variance[i]);
        }
    }
};

inline api::rnc_derivative_function svi_curve_deriv()
{
    return make_rnc_derivatives(svi_rnc_model{}, 5, kSviN);
}

// RNC-LM problem: no bounds (RNC-LM does not support box constraints)
inline api::least_squares_problem make_svi_rnc_ls()
{
    api::least_squares_problem problem;
    problem.num_parameters    = 5;
    problem.num_residuals     = kSviN;
    problem.residuals         = svi_residuals;
    problem.curve_derivatives = std::make_optional(svi_curve_deriv());
    return problem;
}

// Native LM problem: no bounds (native LM rejects box constraints)
inline api::least_squares_problem make_svi_unbounded_ls(bool with_jacobian)
{
    auto problem   = make_svi_ls(with_jacobian);
    problem.bounds = {};
    return problem;
}

// Built once and reused across benchmark iterations.
inline const auto svi_ls_analytic           = make_svi_ls(true);
inline const auto svi_ls_fd                 = make_svi_ls(false);
inline const auto svi_ls_unbounded_analytic = make_svi_unbounded_ls(true);
inline const auto svi_ls_unbounded_fd       = make_svi_unbounded_ls(false);
inline const auto svi_opt_grad              = make_svi_opt(true);
inline const auto svi_opt_fd                = make_svi_opt(false);

}  // namespace solverslib::svi
