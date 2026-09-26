#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "include/detail/eigen_support.h"
#include "include/detail/support.h"

namespace solverslib
{
class solver_options_ipopt;

// Adapter around Ipopt's interior-point NLP solver for general (optionally
// bound-constrained) scalar-objective optimization. Ipopt's own types
// (Ipopt::TNLP, SmartPtr, IpoptApplication) never cross this header - only
// std::function callbacks and std::vector<double>, mirroring ceres_solver /
// nlopt_solver. The implementation is compiled against Ipopt only when
// SOLVERS_HAS_IPOPT is set; otherwise is_supported() is false and solve() is a
// no-op the dispatcher never reaches.
class ipopt_solver
{
public:
    using objective_type = std::function<double(const vector_type&)>;
    using gradient_type  = std::function<void(const vector_type&, vector_type&)>;
    using hessian_type   = std::function<void(const vector_type&, matrix_type&)>;

    SOLVER_API ipopt_solver(size_t num_parameters,
        objective_type             objective,
        gradient_type              gradient,
        hessian_type               hessian      = nullptr,
        std::vector<double>        lower_bounds = {},
        std::vector<double>        upper_bounds = {});

    SOLVER_API static bool is_supported();

    // Returns true when Ipopt reports Solve_Succeeded or an acceptable point.
    SOLVER_API bool solve(std::vector<double>& parameters, const solver_options_ipopt& options);

private:
    size_t              num_parameters_;
    objective_type      objective_;
    gradient_type       gradient_;
    hessian_type        hessian_;
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
};
}  // namespace solverslib
