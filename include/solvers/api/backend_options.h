#pragma once

#include <string>

#include "include/detail/support.h"

// Backend-specific tuning for the problem-structure API. These mirror the knobs
// that the internal Ceres option builders expose, but as API-level enums
// and structs so third-party types never appear in a public header (redesign
// review section 10). The dispatcher maps them to the internal option objects.
namespace solverslib::api
{
// -- Ceres -------------------------------------------------------------------
enum class ceres_linear_solver
{
    dense_qr,
    dense_normal_cholesky,
    sparse_normal_cholesky,
    dense_schur,
    sparse_schur,
    iterative_schur,
    cgnr
};

enum class ceres_trust_region_strategy
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
enum class ipopt_hessian_mode
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
enum class tao_algorithm
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
