#include "solvers/nlopt_solver.h"

#include <utility>

#include "detail/support.h"
#include "solver_options/solver_options_nlopt.h"

#if SOLVERS_HAS_NLOPT
#include <nlopt.h>
#endif

namespace solverslib
{
nlopt_solver::nlopt_solver(size_t num_parameters,
    objective_type                objective,
    gradient_type                 gradient,
    std::vector<double>           lower_bounds,
    std::vector<double>           upper_bounds)
    : num_parameters_(num_parameters), objective_(std::move(objective)),
      gradient_(std::move(gradient)), lower_bounds_(std::move(lower_bounds)),
      upper_bounds_(std::move(upper_bounds))
{
}

bool nlopt_solver::is_supported()
{
#if SOLVERS_HAS_NLOPT
    return true;
#else
    return false;
#endif
}

#if SOLVERS_HAS_NLOPT
namespace
{
struct nlopt_context
{
    nlopt_solver::objective_type* objective;
    nlopt_solver::gradient_type*  gradient;
};

double nlopt_objective_wrapper(
    unsigned n, const double* x, double* grad, void* data)
{
    auto* context = static_cast<nlopt_context*>(data);

    vector_type x_vec = vector_type::Map(const_cast<double*>(x), n);
    double f = (*context->objective)(x_vec);

    if (grad != nullptr && context->gradient != nullptr)
    {
        vector_type grad_vec = vector_type::Zero(n);
        (*context->gradient)(x_vec, grad_vec);
        std::copy(grad_vec.data(), grad_vec.data() + n, grad);
    }

    return f;
}
}  // namespace

bool nlopt_solver::solve(std::vector<double>& parameters,
    const solver_options_nlopt&      options)
{
    return solve_with_status(parameters, options).converged();
}

backend_solve_status nlopt_solver::solve_with_status(std::vector<double>& parameters,
    const solver_options_nlopt&         options)
{
    backend_solve_status status;
    status.outcome     = backend_outcome::failed;
    status.native_code = -1;
    status.message     = "NLopt solver not implemented";

    if (parameters.size() != num_parameters_)
    {
        status.message = "Parameter size mismatch";
        return status;
    }

    nlopt_context context{&objective_, &gradient_};

    nlopt_opt opt = nullptr;

    switch (options.algorithm())
    {
        case nlopt_algorithm_enum::LD_LBFGS:
            opt = nlopt_create(NLOPT_LD_LBFGS, num_parameters_);
            break;
        case nlopt_algorithm_enum::LD_MMA:
            opt = nlopt_create(NLOPT_LD_MMA, num_parameters_);
            break;
        case nlopt_algorithm_enum::LD_SLSQP:
            opt = nlopt_create(NLOPT_LD_SLSQP, num_parameters_);
            break;
        case nlopt_algorithm_enum::LD_CCSAQ:
            opt = nlopt_create(NLOPT_LD_CCSAQ, num_parameters_);
            break;
        case nlopt_algorithm_enum::LN_NELDERMEAD:
            opt = nlopt_create(NLOPT_LN_NELDERMEAD, num_parameters_);
            break;
        case nlopt_algorithm_enum::LN_SBPLX:
            opt = nlopt_create(NLOPT_LN_SBPLX, num_parameters_);
            break;
    }

    if (opt == nullptr)
    {
        status.message = "Failed to create NLopt optimizer";
        return status;
    }

    nlopt_set_min_objective(opt, nlopt_objective_wrapper, &context);

    if (!lower_bounds_.empty())
    {
        nlopt_set_lower_bounds(opt, lower_bounds_.data());
    }
    if (!upper_bounds_.empty())
    {
        nlopt_set_upper_bounds(opt, upper_bounds_.data());
    }

    nlopt_set_maxeval(opt, options.max_num_iterations());
    nlopt_set_xtol_rel(opt, options.xtol_rel());
    nlopt_set_ftol_rel(opt, options.ftol_rel());
    nlopt_set_maxtime(opt, options.max_time());

    double minf = 0.0;
    nlopt_result result = nlopt_optimize(opt, parameters.data(), &minf);

    status.native_code = static_cast<int>(result);

    if (result == NLOPT_SUCCESS || result == NLOPT_FTOL_REACHED ||
        result == NLOPT_XTOL_REACHED || result == NLOPT_STOPVAL_REACHED)
    {
        status.outcome = backend_outcome::converged;
        status.message = "Optimization succeeded";
    }
    else if (result == NLOPT_MAXEVAL_REACHED)
    {
        status.outcome = backend_outcome::budget_exhausted;
        status.message = "Maximum function evaluations reached";
    }
    else if (result == NLOPT_MAXTIME_REACHED)
    {
        status.outcome = backend_outcome::budget_exhausted;
        status.message = "Maximum time reached";
    }
    else
    {
        status.outcome = backend_outcome::failed;
        status.message = "NLopt optimization failed";
    }

    nlopt_destroy(opt);

    return status;
}

#else
bool nlopt_solver::solve(std::vector<double>& /* parameters */,
    const solver_options_nlopt& /* options */)
{
    return false;
}

backend_solve_status nlopt_solver::solve_with_status(std::vector<double>& /* parameters */,
    const solver_options_nlopt& /* options */)
{
    backend_solve_status status;
    status.outcome     = backend_outcome::failed;
    status.native_code = -1;
    status.message     = "NLopt backend not compiled in";
    return status;
}
#endif
}  // namespace solverslib
