// Benchmark every backend on the Rosenbrock problem and the raw-SVI
// calibration problem from Ferhati (2020).
//
// Run with: ./BenchmarkSolverComparison [--benchmark_filter=...]

#include <benchmark/benchmark.h>

#include "api/dispatch.h"
#include "problems/rosenbrock.h"
#include "solver_options/solver_options_bfgs.h"
#include "solver_options/solver_options_ceres.h"
#include "solver_options/solver_options_gn.h"
#include "solver_options/solver_options_ipopt.h"
#include "solver_options/solver_options_lm.h"
#include "solver_options/solver_options_petsc.h"
#include "solver_options/solver_options_rnc_lm.h"
#include "solvers/ceres_solver.h"
#include "solvers/ipopt_solver.h"
#include "solvers/petsc_tao_solver.h"
#include "solvers/rnc_lm_solver.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

using namespace solverslib;
using namespace solverslib::api;
using namespace solverslib::test;

// ============================================================================
// Raw-SVI model and calibration data
// ============================================================================

namespace
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
        mean_k      += q.strike;
        mean_parity += q.call - q.put;
    }
    mean_k      /= static_cast<double>(kSviN);
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
        svi_data d;
        const double forward = infer_forward();

        double min_w = 1e30, max_w = 0.0, min_k = 1e30, max_k = -1e30;
        for (std::size_t i = 0; i < kSviN; ++i)
        {
            d.log_moneyness[i]  = std::log(kPaperQuotes[i].strike / forward);
            d.market_variance[i] = kPaperQuotes[i].market_total_variance;
            min_w = std::min(min_w, d.market_variance[i]);
            max_w = std::max(max_w, d.market_variance[i]);
            min_k = std::min(min_k, d.log_moneyness[i]);
            max_k = std::max(max_k, d.log_moneyness[i]);
        }

        // Box bounds: a in (0, max_w], b in (0,1], rho in (-1,1], m in [2*min_k, 2*max_k], sigma in (0,1]
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

void svi_residuals(const vector_type& p, vector_type& r)
{
    const auto& d = svi_problem_data();
    r.resize(static_cast<int>(kSviN));
    for (std::size_t i = 0; i < kSviN; ++i)
    {
        r[static_cast<int>(i)] =
            svi_model(d.log_moneyness[i], p[0], p[1], p[2], p[3], p[4]) -
            d.market_variance[i];
    }
}

// Analytic Jacobian: dr_i/dp_j
void svi_jacobian(const vector_type& p, matrix_type& j)
{
    const auto& d = svi_problem_data();
    j.resize(static_cast<int>(kSviN), 5);
    for (std::size_t i = 0; i < kSviN; ++i)
    {
        const double c    = d.log_moneyness[i] - p[3];
        const double root = std::sqrt(c * c + p[4] * p[4]);
        const int    row  = static_cast<int>(i);
        j(row, 0) = 1.0;
        j(row, 1) = p[2] * c + root;
        j(row, 2) = p[1] * c;
        j(row, 3) = p[1] * (-p[2] - c / root);
        j(row, 4) = p[1] * p[4] / root;
    }
}

double svi_objective(const vector_type& p)
{
    vector_type r;
    svi_residuals(p, r);
    return 0.5 * r.squaredNorm();
}

void svi_gradient(const vector_type& p, vector_type& g)
{
    vector_type r;
    matrix_type j;
    svi_residuals(p, r);
    svi_jacobian(p, j);
    g = j.transpose() * r;
}

least_squares_problem make_svi_ls(bool with_jacobian)
{
    const auto& d = svi_problem_data();
    least_squares_problem problem;
    problem.num_parameters = 5;
    problem.num_residuals  = kSviN;
    problem.residuals      = svi_residuals;
    problem.bounds.lower   = {d.lower[0], d.lower[1], d.lower[2], d.lower[3], d.lower[4]};
    problem.bounds.upper   = {d.upper[0], d.upper[1], d.upper[2], d.upper[3], d.upper[4]};
    if (with_jacobian)
        problem.jacobian = svi_jacobian;
    return problem;
}

optimization_problem make_svi_opt(bool with_gradient)
{
    const auto& d = svi_problem_data();
    optimization_problem problem;
    problem.num_parameters = 5;
    problem.objective      = svi_objective;
    problem.bounds.lower   = {d.lower[0], d.lower[1], d.lower[2], d.lower[3], d.lower[4]};
    problem.bounds.upper   = {d.upper[0], d.upper[1], d.upper[2], d.upper[3], d.upper[4]};
    if (with_gradient)
        problem.gradient = svi_gradient;
    return problem;
}

// ---------------------------------------------------------------------------
// RNC-LM curve derivatives: Taylor-expands R(p(t)) and J(p(t)) along the
// curve p(t) = base + c1*t + c2*t^2 + ... for orders 1–4.
// ---------------------------------------------------------------------------

rnc_derivative_function svi_curve_deriv()
{
    return [](const vector_type& base, const std::vector<vector_type>& coeffs,
              int order, rnc_curve_derivatives& out)
    {
        const auto& d  = svi_problem_data();
        const int   m  = static_cast<int>(kSviN);

        const double a0   = base[0];
        const double b0   = base[1];
        const double rho0 = base[2];
        const double m0   = base[3];
        const double s0   = base[4];

        const bool have1 = !coeffs.empty();
        const bool have2 = coeffs.size() >= 2;

        out.residual.resize(order + 1);
        out.jacobian.resize(static_cast<std::size_t>(std::max(0, order - 2)) + 1u);

        // Order 0: residual and Jacobian at base
        out.residual[0].resize(m);
        out.jacobian[0].resize(m, 5);
        for (int i = 0; i < m; ++i)
        {
            const double c    = d.log_moneyness[i] - m0;
            const double root = std::sqrt(c * c + s0 * s0);
            out.residual[0][i]    = a0 + b0 * (rho0 * c + root) - d.market_variance[i];
            out.jacobian[0](i, 0) = 1.0;
            out.jacobian[0](i, 1) = rho0 * c + root;
            out.jacobian[0](i, 2) = b0 * c;
            out.jacobian[0](i, 3) = b0 * (-rho0 - c / root);
            out.jacobian[0](i, 4) = b0 * s0 / root;
        }

        // Order 1: residual[1] = J0 * c1
        if (have1)
            out.residual[1] = out.jacobian[0] * coeffs[0];
        else
            out.residual[1] = vector_type::Zero(m);

        if (order >= 2)
        {
            // residual[2][i] = (1/2) * c1^T * H_i * c1
            // Non-zero Hessian entries for r_i (at base):
            //   H[1,2] = c,  H[1,3] = -rho-c/root,  H[1,4] = s/root,
            //   H[2,3] = -b, H[3,3] = b*s^2/r3,    H[3,4] = b*c*s/r3,
            //   H[4,4] = b*c^2/r3
            out.residual[2] = vector_type::Zero(m);
            if (have1)
            {
                const double v1 = coeffs[0][1];
                const double v2 = coeffs[0][2];
                const double v3 = coeffs[0][3];
                const double v4 = coeffs[0][4];
                for (int i = 0; i < m; ++i)
                {
                    const double c    = d.log_moneyness[i] - m0;
                    const double root = std::sqrt(c * c + s0 * s0);
                    const double r3   = root * root * root;
                    out.residual[2][i] =
                        c * v1 * v2 +
                        (-rho0 - c / root) * v1 * v3 +
                        (s0 / root) * v1 * v4 +
                        (-b0) * v2 * v3 +
                        0.5 * (b0 * s0 * s0 / r3) * v3 * v3 +
                        (b0 * c * s0 / r3) * v3 * v4 +
                        0.5 * (b0 * c * c / r3) * v4 * v4;
                }
            }
        }

        if (order >= 3)
        {
            // jacobian[1](i,j) = (d/dt J_i(p(t)))|t=0 = sum_k H_i[j,k]*c1[k]
            out.jacobian[1] = matrix_type::Zero(m, 5);
            // residual[3][i] = c1^T * H_i * c2  +  (1/6) * sum_jkl T_i[j,k,l]*c1_j*c1_k*c1_l
            out.residual[3] = vector_type::Zero(m);
            if (have1)
            {
                const double v1 = coeffs[0][1];
                const double v2 = coeffs[0][2];
                const double v3 = coeffs[0][3];
                const double v4 = coeffs[0][4];
                for (int i = 0; i < m; ++i)
                {
                    const double c    = d.log_moneyness[i] - m0;
                    const double root = std::sqrt(c * c + s0 * s0);
                    const double r3   = root * root * root;
                    const double r5   = r3 * root * root;
                    const double h13  = -rho0 - c / root;
                    const double h14  = s0 / root;
                    const double h33  = b0 * s0 * s0 / r3;
                    const double h34  = b0 * c * s0 / r3;
                    const double h44  = b0 * c * c / r3;

                    // jacobian[1] columns (using symmetry H[j,k] = H[k,j])
                    out.jacobian[1](i, 0) = 0.0;
                    out.jacobian[1](i, 1) = c * v2 + h13 * v3 + h14 * v4;
                    out.jacobian[1](i, 2) = c * v1 + (-b0) * v3;
                    out.jacobian[1](i, 3) = h13 * v1 + (-b0) * v2 + h33 * v3 + h34 * v4;
                    out.jacobian[1](i, 4) = h14 * v1 + h34 * v3 + h44 * v4;

                    // c1^T * H_i * c2 cross term
                    if (have2)
                    {
                        const double u1 = coeffs[1][1];
                        const double u2 = coeffs[1][2];
                        const double u3 = coeffs[1][3];
                        const double u4 = coeffs[1][4];
                        out.residual[3][i] +=
                            c * (v1 * u2 + v2 * u1) +
                            h13 * (v1 * u3 + v3 * u1) +
                            h14 * (v1 * u4 + v4 * u1) +
                            (-b0) * (v2 * u3 + v3 * u2) +
                            h33 * v3 * u3 +
                            h34 * (v3 * u4 + v4 * u3) +
                            h44 * v4 * u4;
                    }

                    // (1/6) * sum T[j,k,l] c1_j c1_k c1_l — non-zero T values:
                    //   T[1,2,3]=-1,  T[1,3,3]=s^2/r3,  T[1,3,4]=c*s/r3,  T[1,4,4]=c^2/r3
                    //   T[3,3,3]=3b*c*s^2/r5, T[3,3,4]=b*s*(2c^2-s^2)/r5,
                    //   T[3,4,4]=b*c*(c^2-2s^2)/r5, T[4,4,4]=-3b*c^2*s/r5
                    out.residual[3][i] +=
                        -v1 * v2 * v3 +
                        0.5 * (s0 * s0 / r3) * v1 * v3 * v3 +
                        (c * s0 / r3) * v1 * v3 * v4 +
                        0.5 * (c * c / r3) * v1 * v4 * v4 +
                        0.5 * (b0 * c * s0 * s0 / r5) * v3 * v3 * v3 +
                        0.5 * (b0 * s0 * (2.0 * c * c - s0 * s0) / r5) * v3 * v3 * v4 +
                        0.5 * (b0 * c * (c * c - 2.0 * s0 * s0) / r5) * v3 * v4 * v4 +
                        (-0.5) * (b0 * c * c * s0 / r5) * v4 * v4 * v4;
                }
            }
        }

        if (order >= 4)
        {
            // Higher-order corrections are zero-approximated; the solver degrades
            // gracefully to the order-3 curve when these are zero.
            out.residual[4]  = vector_type::Zero(m);
            out.jacobian[2] = matrix_type::Zero(m, 5);
        }
    };
}

// RNC-LM problem: no bounds (RNC-LM does not support box constraints)
least_squares_problem make_svi_rnc_ls()
{
    least_squares_problem problem;
    problem.num_parameters   = 5;
    problem.num_residuals    = kSviN;
    problem.residuals        = svi_residuals;
    problem.curve_derivatives = std::make_optional(svi_curve_deriv());
    return problem;
}

// ============================================================================
// Shared solver option objects
// ============================================================================

// ---------------------------------------------------------------------------
// Rosenbrock options
// ---------------------------------------------------------------------------

const auto lm_options = solver_options_lm_builder()
                            .with_max_iterations(400)
                            .with_function_tolerance(1e-14)
                            .with_gradient_tolerance(1e-12)
                            .with_parameter_tolerance(1e-14)
                            .build();

const auto gn_options = solver_options_gn_builder()
                            .with_max_iterations(200)
                            .with_function_tolerance(1e-14)
                            .with_gradient_tolerance(1e-12)
                            .with_parameter_tolerance(1e-14)
                            .build();

const auto bfgs_options = solver_options_bfgs_builder()
                              .with_max_iterations(1000)
                              .with_function_tolerance(1e-14)
                              .with_gradient_tolerance(1e-10)
                              .build();

const auto ceres_options = solver_options_ceres_builder().with_max_iterations(200).build();

const auto ipopt_options = solver_options_ipopt_builder().with_max_iterations(500).build();

const auto petsc_brgn = solver_options_petsc_builder()
                            .with_tao_type(tao_algorithm_enum::BRGN)
                            .with_max_iterations(200)
                            .build();

const auto petsc_lmvm = solver_options_petsc_builder()
                            .with_tao_type(tao_algorithm_enum::LMVM)
                            .with_max_iterations(500)
                            .build();

// ---------------------------------------------------------------------------
// SVI-specific LM option variants
// ---------------------------------------------------------------------------

// Default: geodesic on, bold acceptance off, analytic Jacobian
const auto svi_lm_default = solver_options_lm_builder()
                                .with_max_iterations(1000)
                                .with_function_tolerance(1e-12)
                                .with_gradient_tolerance(1e-10)
                                .with_parameter_tolerance(1e-12)
                                .with_geodesic_acceleration(true)
                                .with_bold_acceptance(false)
                                .build();

// Geodesic off
const auto svi_lm_no_geodesic = solver_options_lm_builder()
                                     .with_max_iterations(1000)
                                     .with_function_tolerance(1e-12)
                                     .with_gradient_tolerance(1e-10)
                                     .with_parameter_tolerance(1e-12)
                                     .with_geodesic_acceleration(false)
                                     .with_bold_acceptance(false)
                                     .build();

// Bold acceptance on, geodesic on
const auto svi_lm_bold = solver_options_lm_builder()
                             .with_max_iterations(1000)
                             .with_function_tolerance(1e-12)
                             .with_gradient_tolerance(1e-10)
                             .with_parameter_tolerance(1e-12)
                             .with_geodesic_acceleration(true)
                             .with_bold_acceptance(true)
                             .build();

// Bold acceptance on, geodesic off
const auto svi_lm_bold_no_geodesic = solver_options_lm_builder()
                                         .with_max_iterations(1000)
                                         .with_function_tolerance(1e-12)
                                         .with_gradient_tolerance(1e-10)
                                         .with_parameter_tolerance(1e-12)
                                         .with_geodesic_acceleration(false)
                                         .with_bold_acceptance(true)
                                         .build();

const auto svi_gn_options = solver_options_gn_builder()
                                .with_max_iterations(1000)
                                .with_function_tolerance(1e-12)
                                .with_gradient_tolerance(1e-10)
                                .with_parameter_tolerance(1e-12)
                                .build();

const auto svi_ceres_options = solver_options_ceres_builder().with_max_iterations(500).build();

const auto svi_ipopt_options = solver_options_ipopt_builder().with_max_iterations(1000).build();

const auto svi_petsc_brgn = solver_options_petsc_builder()
                                .with_tao_type(tao_algorithm_enum::BRGN)
                                .with_max_iterations(500)
                                .build();

const auto svi_petsc_lmvm = solver_options_petsc_builder()
                                .with_tao_type(tao_algorithm_enum::LMVM)
                                .with_max_iterations(1000)
                                .build();

// ---------------------------------------------------------------------------
// SVI RNC-LM options — vary order (1–4)
// ---------------------------------------------------------------------------

const auto svi_rnc_order1 = solver_options_rnc_lm_builder()
                                .with_max_iterations(1000)
                                .with_function_tolerance(1e-12)
                                .with_gradient_tolerance(1e-10)
                                .with_parameter_tolerance(1e-12)
                                .with_order(1)
                                .build();

const auto svi_rnc_order2 = solver_options_rnc_lm_builder()
                                .with_max_iterations(1000)
                                .with_function_tolerance(1e-12)
                                .with_gradient_tolerance(1e-10)
                                .with_parameter_tolerance(1e-12)
                                .with_order(2)
                                .build();

const auto svi_rnc_order3 = solver_options_rnc_lm_builder()
                                .with_max_iterations(1000)
                                .with_function_tolerance(1e-12)
                                .with_gradient_tolerance(1e-10)
                                .with_parameter_tolerance(1e-12)
                                .with_order(3)
                                .build();

const auto svi_rnc_order4 = solver_options_rnc_lm_builder()
                                .with_max_iterations(1000)
                                .with_function_tolerance(1e-12)
                                .with_gradient_tolerance(1e-10)
                                .with_parameter_tolerance(1e-12)
                                .with_order(4)
                                .build();

// ---------------------------------------------------------------------------
// SVI L-BFGS options — vary line search method/type and memory size (tau)
// ---------------------------------------------------------------------------

// Default: Armijo + Nocedal-Wright, tau=6
const auto svi_lbfgs_armijo = solver_options_bfgs_builder()
                                   .with_max_iterations(1000)
                                   .with_function_tolerance(1e-12)
                                   .with_gradient_tolerance(1e-10)
                                   .with_method_type(lbfgs_line_search_method_type::ARMIJO)
                                   .with_type(lbfgs_line_search_type::NOCEDAL_WRIGHT)
                                   .with_tau(6)
                                   .build();

const auto svi_lbfgs_wolfe = solver_options_bfgs_builder()
                                  .with_max_iterations(1000)
                                  .with_function_tolerance(1e-12)
                                  .with_gradient_tolerance(1e-10)
                                  .with_method_type(lbfgs_line_search_method_type::WOLFE)
                                  .with_type(lbfgs_line_search_type::NOCEDAL_WRIGHT)
                                  .with_tau(6)
                                  .build();

const auto svi_lbfgs_strong_wolfe = solver_options_bfgs_builder()
                                        .with_max_iterations(1000)
                                        .with_function_tolerance(1e-12)
                                        .with_gradient_tolerance(1e-10)
                                        .with_method_type(lbfgs_line_search_method_type::STRONG_WOLFE)
                                        .with_type(lbfgs_line_search_type::NOCEDAL_WRIGHT)
                                        .with_tau(6)
                                        .build();

const auto svi_lbfgs_backtracking = solver_options_bfgs_builder()
                                        .with_max_iterations(1000)
                                        .with_function_tolerance(1e-12)
                                        .with_gradient_tolerance(1e-10)
                                        .with_method_type(lbfgs_line_search_method_type::ARMIJO)
                                        .with_type(lbfgs_line_search_type::BACKTRACKING)
                                        .with_tau(6)
                                        .build();

const auto svi_lbfgs_bracketing = solver_options_bfgs_builder()
                                       .with_max_iterations(1000)
                                       .with_function_tolerance(1e-12)
                                       .with_gradient_tolerance(1e-10)
                                       .with_method_type(lbfgs_line_search_method_type::ARMIJO)
                                       .with_type(lbfgs_line_search_type::BRACKETING)
                                       .with_tau(6)
                                       .build();

const auto svi_lbfgs_tau3 = solver_options_bfgs_builder()
                                 .with_max_iterations(1000)
                                 .with_function_tolerance(1e-12)
                                 .with_gradient_tolerance(1e-10)
                                 .with_method_type(lbfgs_line_search_method_type::ARMIJO)
                                 .with_type(lbfgs_line_search_type::NOCEDAL_WRIGHT)
                                 .with_tau(3)
                                 .build();

const auto svi_lbfgs_tau10 = solver_options_bfgs_builder()
                                  .with_max_iterations(1000)
                                  .with_function_tolerance(1e-12)
                                  .with_gradient_tolerance(1e-10)
                                  .with_method_type(lbfgs_line_search_method_type::ARMIJO)
                                  .with_type(lbfgs_line_search_type::NOCEDAL_WRIGHT)
                                  .with_tau(10)
                                  .build();

// Pre-built problems (built once, reused across benchmark iterations)
const auto svi_ls_analytic = make_svi_ls(true);
const auto svi_ls_fd       = make_svi_ls(false);
const auto svi_opt_grad    = make_svi_opt(true);
const auto svi_opt_fd      = make_svi_opt(false);
const auto svi_rnc_problem = make_svi_rnc_ls();

// Rosenbrock fixtures
const auto start = rosenbrock_start();
const auto ls    = rosenbrock_least_squares(true);
const auto ls_fd = rosenbrock_least_squares(false);
const auto opt   = rosenbrock_optimization(true);

}  // namespace

// ============================================================================
// Rosenbrock benchmarks
// ============================================================================

// ---------------------------------------------------------------------------
// Native Least Squares
// ---------------------------------------------------------------------------

static void BM_NativeLM_LeastSquares(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(ls, start, *lm_options));
}
BENCHMARK(BM_NativeLM_LeastSquares);

static void BM_NativeLM_FiniteDiff(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(ls_fd, start, *lm_options));
}
BENCHMARK(BM_NativeLM_FiniteDiff);

static void BM_NativeGN_LeastSquares(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(ls, start, *gn_options));
}
BENCHMARK(BM_NativeGN_LeastSquares);

// ---------------------------------------------------------------------------
// Ceres Least Squares
// ---------------------------------------------------------------------------

static void BM_Ceres_LeastSquares(benchmark::State& state)
{
    if (!ceres_solver::is_supported())
    {
        state.SkipWithMessage("Ceres backend not compiled in");
        return;
    }
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(ls, start, *ceres_options));
}
BENCHMARK(BM_Ceres_LeastSquares);

// ---------------------------------------------------------------------------
// Ipopt Least Squares
// ---------------------------------------------------------------------------

static void BM_Ipopt_LeastSquares(benchmark::State& state)
{
    if (!ipopt_solver::is_supported())
    {
        state.SkipWithMessage("Ipopt backend not compiled in");
        return;
    }
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(ls, start, *ipopt_options));
}
BENCHMARK(BM_Ipopt_LeastSquares)->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// PETSc/TAO Least Squares
// ---------------------------------------------------------------------------

static void BM_PETSc_BRGN_LeastSquares(benchmark::State& state)
{
    if (!petsc_tao_solver::is_supported())
    {
        state.SkipWithMessage("PETSc/TAO backend not compiled in");
        return;
    }
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(ls, start, *petsc_brgn));
}
BENCHMARK(BM_PETSc_BRGN_LeastSquares);

// ---------------------------------------------------------------------------
// Native Optimization
// ---------------------------------------------------------------------------

static void BM_NativeLBFGS_Optimization(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(opt, start, *bfgs_options));
}
BENCHMARK(BM_NativeLBFGS_Optimization);

// ---------------------------------------------------------------------------
// Ipopt Optimization
// ---------------------------------------------------------------------------

static void BM_Ipopt_Optimization(benchmark::State& state)
{
    if (!ipopt_solver::is_supported())
    {
        state.SkipWithMessage("Ipopt backend not compiled in");
        return;
    }
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(opt, start, *ipopt_options));
}
BENCHMARK(BM_Ipopt_Optimization)->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// PETSc/TAO Optimization
// ---------------------------------------------------------------------------

static void BM_PETSc_LMVM_Optimization(benchmark::State& state)
{
    if (!petsc_tao_solver::is_supported())
    {
        state.SkipWithMessage("PETSc/TAO backend not compiled in");
        return;
    }
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(opt, start, *petsc_lmvm));
}
BENCHMARK(BM_PETSc_LMVM_Optimization);

// ============================================================================
// SVI calibration benchmarks
// ============================================================================

// ---------------------------------------------------------------------------
// Native LM — analytic Jacobian variants
// ---------------------------------------------------------------------------

static void BM_SVI_NativeLM_Analytic(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_analytic, x0, *svi_lm_default));
}
BENCHMARK(BM_SVI_NativeLM_Analytic);

static void BM_SVI_NativeLM_Analytic_NoGeodesic(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_analytic, x0, *svi_lm_no_geodesic));
}
BENCHMARK(BM_SVI_NativeLM_Analytic_NoGeodesic);

static void BM_SVI_NativeLM_Analytic_Bold(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_analytic, x0, *svi_lm_bold));
}
BENCHMARK(BM_SVI_NativeLM_Analytic_Bold);

static void BM_SVI_NativeLM_Analytic_Bold_NoGeodesic(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_analytic, x0, *svi_lm_bold_no_geodesic));
}
BENCHMARK(BM_SVI_NativeLM_Analytic_Bold_NoGeodesic);

// ---------------------------------------------------------------------------
// Native LM — finite differences
// ---------------------------------------------------------------------------

static void BM_SVI_NativeLM_FiniteDiff(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_fd, x0, *svi_lm_default));
}
BENCHMARK(BM_SVI_NativeLM_FiniteDiff);

static void BM_SVI_NativeLM_FiniteDiff_NoGeodesic(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_fd, x0, *svi_lm_no_geodesic));
}
BENCHMARK(BM_SVI_NativeLM_FiniteDiff_NoGeodesic);

static void BM_SVI_NativeLM_FiniteDiff_Bold(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_fd, x0, *svi_lm_bold));
}
BENCHMARK(BM_SVI_NativeLM_FiniteDiff_Bold);

static void BM_SVI_NativeLM_FiniteDiff_Bold_NoGeodesic(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_fd, x0, *svi_lm_bold_no_geodesic));
}
BENCHMARK(BM_SVI_NativeLM_FiniteDiff_Bold_NoGeodesic);

// ---------------------------------------------------------------------------
// Native GN
// ---------------------------------------------------------------------------

static void BM_SVI_NativeGN_Analytic(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_analytic, x0, *svi_gn_options));
}
BENCHMARK(BM_SVI_NativeGN_Analytic);

static void BM_SVI_NativeGN_FiniteDiff(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_fd, x0, *svi_gn_options));
}
BENCHMARK(BM_SVI_NativeGN_FiniteDiff);

// ---------------------------------------------------------------------------
// Ceres
// ---------------------------------------------------------------------------

static void BM_SVI_Ceres_Analytic(benchmark::State& state)
{
    if (!ceres_solver::is_supported())
    {
        state.SkipWithMessage("Ceres backend not compiled in");
        return;
    }
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_analytic, x0, *svi_ceres_options));
}
BENCHMARK(BM_SVI_Ceres_Analytic);

static void BM_SVI_Ceres_FiniteDiff(benchmark::State& state)
{
    if (!ceres_solver::is_supported())
    {
        state.SkipWithMessage("Ceres backend not compiled in");
        return;
    }
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_fd, x0, *svi_ceres_options));
}
BENCHMARK(BM_SVI_Ceres_FiniteDiff);

// ---------------------------------------------------------------------------
// Ipopt (optimization form)
// ---------------------------------------------------------------------------

static void BM_SVI_Ipopt_Grad(benchmark::State& state)
{
    if (!ipopt_solver::is_supported())
    {
        state.SkipWithMessage("Ipopt backend not compiled in");
        return;
    }
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_grad, x0, *svi_ipopt_options));
}
BENCHMARK(BM_SVI_Ipopt_Grad)->Unit(benchmark::kMillisecond);

static void BM_SVI_Ipopt_FiniteDiff(benchmark::State& state)
{
    if (!ipopt_solver::is_supported())
    {
        state.SkipWithMessage("Ipopt backend not compiled in");
        return;
    }
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_fd, x0, *svi_ipopt_options));
}
BENCHMARK(BM_SVI_Ipopt_FiniteDiff)->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// PETSc/TAO BRGN (least squares) and LMVM (optimization)
// ---------------------------------------------------------------------------

static void BM_SVI_PETSc_BRGN_Analytic(benchmark::State& state)
{
    if (!petsc_tao_solver::is_supported())
    {
        state.SkipWithMessage("PETSc/TAO backend not compiled in");
        return;
    }
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_analytic, x0, *svi_petsc_brgn));
}
BENCHMARK(BM_SVI_PETSc_BRGN_Analytic);

static void BM_SVI_PETSc_LMVM_Grad(benchmark::State& state)
{
    if (!petsc_tao_solver::is_supported())
    {
        state.SkipWithMessage("PETSc/TAO backend not compiled in");
        return;
    }
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_grad, x0, *svi_petsc_lmvm));
}
BENCHMARK(BM_SVI_PETSc_LMVM_Grad);

static void BM_SVI_PETSc_LMVM_FiniteDiff(benchmark::State& state)
{
    if (!petsc_tao_solver::is_supported())
    {
        state.SkipWithMessage("PETSc/TAO backend not compiled in");
        return;
    }
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_fd, x0, *svi_petsc_lmvm));
}
BENCHMARK(BM_SVI_PETSc_LMVM_FiniteDiff);

// ============================================================================
// SVI RNC-LM benchmarks — vary curve order
// ============================================================================

static void BM_SVI_RncLM_Order1(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve_rnc_lm(svi_rnc_problem, x0, *svi_rnc_order1));
}
BENCHMARK(BM_SVI_RncLM_Order1);

static void BM_SVI_RncLM_Order2(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve_rnc_lm(svi_rnc_problem, x0, *svi_rnc_order2));
}
BENCHMARK(BM_SVI_RncLM_Order2);

static void BM_SVI_RncLM_Order3(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve_rnc_lm(svi_rnc_problem, x0, *svi_rnc_order3));
}
BENCHMARK(BM_SVI_RncLM_Order3);

static void BM_SVI_RncLM_Order4(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve_rnc_lm(svi_rnc_problem, x0, *svi_rnc_order4));
}
BENCHMARK(BM_SVI_RncLM_Order4);

// ============================================================================
// SVI L-BFGS benchmarks — vary line search method, type, memory size
// ============================================================================

static void BM_SVI_LBFGS_Armijo_Grad(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_grad, x0, *svi_lbfgs_armijo));
}
BENCHMARK(BM_SVI_LBFGS_Armijo_Grad);

static void BM_SVI_LBFGS_Wolfe_Grad(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_grad, x0, *svi_lbfgs_wolfe));
}
BENCHMARK(BM_SVI_LBFGS_Wolfe_Grad);

static void BM_SVI_LBFGS_StrongWolfe_Grad(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_grad, x0, *svi_lbfgs_strong_wolfe));
}
BENCHMARK(BM_SVI_LBFGS_StrongWolfe_Grad);

static void BM_SVI_LBFGS_Backtracking_Grad(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_grad, x0, *svi_lbfgs_backtracking));
}
BENCHMARK(BM_SVI_LBFGS_Backtracking_Grad);

static void BM_SVI_LBFGS_Bracketing_Grad(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_grad, x0, *svi_lbfgs_bracketing));
}
BENCHMARK(BM_SVI_LBFGS_Bracketing_Grad);

static void BM_SVI_LBFGS_Armijo_FiniteDiff(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_fd, x0, *svi_lbfgs_armijo));
}
BENCHMARK(BM_SVI_LBFGS_Armijo_FiniteDiff);

static void BM_SVI_LBFGS_Tau3_Grad(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_grad, x0, *svi_lbfgs_tau3));
}
BENCHMARK(BM_SVI_LBFGS_Tau3_Grad);

static void BM_SVI_LBFGS_Tau10_Grad(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_grad, x0, *svi_lbfgs_tau10));
}
BENCHMARK(BM_SVI_LBFGS_Tau10_Grad);

BENCHMARK_MAIN();
