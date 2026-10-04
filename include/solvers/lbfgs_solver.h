#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "detail/native_result.h"
#include "detail/support.h"

namespace solverslib
{
namespace api::detail
{
class residual_evaluator;
class gradient_evaluator;
}  // namespace api::detail
class solver_options_bfgs;
class lbfgs_solver
{
    using size_type   = size_t;
    using scalar_type = double;

    using function_type = std::function<void(vector_type const&, vector_type&)>;
    using jacobian_type = std::function<void(vector_type const&, matrix_type&)>;

    using objective_type = std::function<double(vector_type const&)>;
    using gradient_type  = std::function<void(vector_type const&, vector_type&)>;

    function_type function_;
    jacobian_type jacobian_;
    size_type     num_parameters_;
    size_type     num_residuals_;

    objective_type objective_;
    gradient_type  gradient_;
    bool           scalar_mode_ = false;

    // Non-owning; when set, take precedence over the callbacks above.
    api::detail::residual_evaluator* evaluator_          = nullptr;
    api::detail::gradient_evaluator* gradient_evaluator_ = nullptr;

    native_result run(vector_type& parameters, const solver_options_bfgs& options) const;

public:
    SOLVER_API lbfgs_solver(size_type num_parameters,
        size_type                     num_residuals,
        function_type                 function,
        jacobian_type                 jacobian = nullptr);

    SOLVER_API lbfgs_solver(
        size_type num_parameters, objective_type objective, gradient_type gradient);

    // Evaluator-driven forms; the evaluator must outlive the solver.
    SOLVER_API explicit lbfgs_solver(api::detail::residual_evaluator& evaluator);
    SOLVER_API explicit lbfgs_solver(api::detail::gradient_evaluator& evaluator);

    SOLVER_API native_result solve(
        vector_type& parameters, const solver_options_bfgs& options) const;
};
}  // namespace solverslib
