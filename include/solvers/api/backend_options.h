#ifndef SOLVERS_BACKEND_OPTIONS_H_
#define SOLVERS_BACKEND_OPTIONS_H_

#include <cstdint>
#include <string>

#include "detail/support.h"

// Backend-specific tuning for the problem-structure API. These mirror the knobs
// that the internal Ceres option builders expose, but as API-level enums
// and structs so third-party types never appear in a public header (redesign
// review section 10). The dispatcher maps them to the internal option objects.
namespace solverslib::api
{
// RNC-LM (Liu & Zhang, arXiv:2607.07623v2, Sections 4.2 and 4.4).
// Algorithm-specific controls; budgets/tolerances live in solve_options.
struct rnc_lm_options
{
    int    order                  = 3;  // Truncation order K, 1..4
    int    max_curve_trials       = 4;  // Includes the initial t=1 trial
    double acceptance_threshold   = 1e-4;
    double contraction_min        = 0.3;
    double contraction_max        = 0.5;
    double initial_damping        = 1e-4;
    double damping_floor          = 1e-15;
    double damping_ceiling        = 1e12;
    double diagonal_scaling_floor = 1e-12;
};

// -- Native Levenberg-Marquardt -----------------------------------------------
// Mirrors solver_options_lm (Transtrum & Sethna, arXiv:1201.5885; definitions
// in docs/levenberg-marquardt.md) so the damping, geodesic and variant controls
// are reachable from api::solve. Budgets and tolerances live in solve_options.
enum class lm_variant : std::uint8_t
{
    levenberg_marquardt,      // multiplicative damping with diagonal scaling
    quadratic_interpolation,  // damping from a quadratic line interpolation
    nielsen                   // Nielsen's gain-ratio damping update (default)
};

// How each damped LM step is solved. normal_ldlt factors J^T J + damping (fast
// for many more residuals than parameters); augmented_qr factors [J; sqrt(D)]
// and never squares the condition number of J.
enum class lm_linear_solver : std::uint8_t
{
    normal_ldlt,
    augmented_qr
};

struct lm_options
{
    lm_variant       variant       = lm_variant::nielsen;
    lm_linear_solver linear_solver = lm_linear_solver::normal_ldlt;

    bool   bold_acceptance                 = false;
    double bold_acceptance_exponent        = 2.0;
    bool   geodesic_acceleration           = true;
    double geodesic_acceleration_threshold = 0.75;
    double geodesic_acceleration_step      = 0.05;

    double initial_damping                     = 1e-4;
    double initial_rejection_multiplier        = 2.0;  // Nielsen's nu
    double damping_decrease_factor             = 9.0;
    double damping_increase_factor             = 11.0;
    double damping_floor                       = 1e-7;
    double nielsen_damping_floor               = 1e-15;
    double damping_ceiling                     = 1e12;
    double levenberg_marquardt_damping_ceiling = 1e7;

    double diagonal_scaling_floor = 1e-12;
    double roundoff_noise_factor  = 8.0;
};

// -- Ceres -------------------------------------------------------------------
enum class ceres_linear_solver : std::uint8_t
{
    dense_qr,
    dense_normal_cholesky,
    sparse_normal_cholesky,
    dense_schur,
    sparse_schur,
    iterative_schur,
    cgnr
};

enum class ceres_trust_region_strategy : std::uint8_t
{
    levenberg_marquardt,
    dogleg
};

struct ceres_options
{
    ceres_linear_solver         linear_solver = ceres_linear_solver::dense_qr;
    ceres_trust_region_strategy trust_region_strategy =
        ceres_trust_region_strategy::levenberg_marquardt;
    int    num_threads             = 1;
    double max_solver_time_seconds = 1e9;
};

// -- Ipopt -------------------------------------------------------------------
enum class ipopt_hessian_mode : std::uint8_t
{
    limited_memory,  // Ipopt's own L-BFGS quasi-Newton approximation
    exact            // use the problem's Hessian callback when present
};

struct ipopt_options
{
    ipopt_hessian_mode hessian_mode          = ipopt_hessian_mode::limited_memory;
    double             tol                   = 1e-8;
    double             acceptable_tol        = 1e-6;
    double             max_wall_time_seconds = 1e9;
    // Ipopt linear solver name (e.g. "mumps"); empty keeps Ipopt's default.
    std::string linear_solver;
};

// -- PETSc / TAO (also realizes the POUNDERS backend) ------------------------
enum class tao_algorithm : std::uint8_t
{
    automatic,  // dispatcher picks pounders (LS) / lmvm (objective)
    pounders,   // derivative-free least squares
    brgn,       // bounded regularized Gauss-Newton least squares
    nls,        // Newton line search
    ntr,        // Newton trust region
    ntl,        // Newton trust-region line search
    lmvm,       // limited-memory variable metric (matrix-free quasi-Newton)
    bqnls,      // bound-constrained quasi-Newton line search
    bnls        // bound-constrained Newton line search
};

struct petsc_tao_options
{
    tao_algorithm algorithm   = tao_algorithm::automatic;
    double        gatol       = 1e-8;   // absolute gradient tolerance
    double        grtol       = 1e-8;   // relative gradient tolerance
    bool          matrix_free = false;  // drive TAO with a Hessian-vector product
};

}  // namespace solverslib::api

#endif  // SOLVERS_BACKEND_OPTIONS_H_
