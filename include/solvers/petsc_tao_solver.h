#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "include/detail/eigen_support.h"
#include "include/detail/support.h"

namespace solverslib
{
class solver_options_petsc;

// Adapter around PETSc/TAO. TAO's algorithm family covers both general
// objective minimization (nls/ntr/ntl/lmvm/bnls/bqnls) and derivative-free /
// Gauss-Newton least squares (pounders/brgn), so this one adapter serves both
// the backend::petsc_tao and backend::pounders routes. PETSc's C types (Tao,
// Vec, Mat) never cross this header. Compiled against PETSc only when
// SOLVERS_HAS_PETSC is set; otherwise is_supported() is false.
class petsc_tao_solver
{
public:
    using objective_type = std::function<double(const vector_type&)>;
    using gradient_type  = std::function<void(const vector_type&, vector_type&)>;
    using hessian_type   = std::function<void(const vector_type&, matrix_type&)>;
    using residual_type  = std::function<void(const vector_type&, vector_type&)>;
    using jacobian_type  = std::function<void(const vector_type&, matrix_type&)>;

    // General objective mode: minimize f(x) with optional Hessian and bounds.
    SOLVER_API petsc_tao_solver(size_t num_parameters,
        objective_type                 objective,
        gradient_type                  gradient,
        hessian_type                   hessian      = nullptr,
        std::vector<double>            lower_bounds = {},
        std::vector<double>            upper_bounds = {});

    // Least-squares mode (POUNDERS / BRGN): minimize 0.5*||r(x)||^2.
    SOLVER_API petsc_tao_solver(size_t num_parameters,
        size_t                         num_residuals,
        residual_type                  residuals,
        jacobian_type                  jacobian     = nullptr,
        std::vector<double>            lower_bounds = {},
        std::vector<double>            upper_bounds = {});

    SOLVER_API static bool is_supported();

    // Returns true when TAO reports a converged termination reason.
    SOLVER_API bool solve(std::vector<double>& parameters, const solver_options_petsc& options);

private:
    bool                is_least_squares_;
    size_t              num_parameters_;
    size_t              num_residuals_;
    objective_type      objective_;
    gradient_type       gradient_;
    hessian_type        hessian_;
    residual_type       residuals_;
    jacobian_type       jacobian_;
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
};
}  // namespace solverslib
