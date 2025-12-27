#include "optimization_algorithm/nlopt_algorithms.h"

#if QUARISMA_MODULE_ENABLE_QUARISMA_nlopt
#include <nlopt.hpp>
#endif

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "MathModule.h"
#include "common/macros.h"
#include "optimization_algorithm/solver_options/solver_options_nlopt.h"
#include "terminals/matrix.h"
#include "terminals/vector.h"

namespace quarisma
{
#if QUARISMA_MODULE_ENABLE_QUARISMA_nlopt
// Function to map nlopt_algo_name enum to nlopt::algorithm enum
nlopt::algorithm getNloptEnum(nlopt_algo_name algo)
{
    switch (algo)
    {
    case nlopt_algo_name::AUGMENTED_LAGRANGIAN:
        return nlopt::LD_AUGLAG;
    case nlopt_algo_name::AUGMENTED_LAGRANGIAN_WITH_EQUALITY_CONSTRAINTS:
    case nlopt_algo_name::AUGMENTED_LAGRANGIAN_WITH_COBYLA:
    case nlopt_algo_name::AUGMENTED_LAGRANGIAN_WITH_BOBYQA:
        return nlopt::LD_AUGLAG_EQ;  // Adjust depending on the inner solver

    case nlopt_algo_name::METHOD_OF_MOVING_ASYMPTOTES:
        return nlopt::LD_MMA;
    case nlopt_algo_name::CONSTRAINED_OPTIMIZATION_BY_LINEAR_APPROXIMATIONS:
        return nlopt::LN_COBYLA;
    case nlopt_algo_name::SEQUENTIAL_LEAST_SQUARES_PROGRAMMING:
        return nlopt::LD_SLSQP;
    case nlopt_algo_name::BOUND_OPTIMIZATION_BY_QUADRATIC_APPROXIMATION:
        return nlopt::LN_BOBYQA;
    case nlopt_algo_name::LBFGS:
        return nlopt::LD_LBFGS;

    case nlopt_algo_name::IMPROVED_STOCHASTIC_RANKING_EVOLUTION_STRATEGY:
        return nlopt::GN_ISRES;
    case nlopt_algo_name::CONTROLLED_RANDOM_SEARCH_WITH_LOCAL_MUTATION:
        return nlopt::GN_CRS2_LM;
    case nlopt_algo_name::DIVIDING_RECTANGLES:
        return nlopt::GN_ORIG_DIRECT_L;

    case nlopt_algo_name::PRECONDITIONED_TRUNCATED_NEWTON_METHOD:
        return nlopt::LD_TNEWTON_PRECOND;
    case nlopt_algo_name::VARIABLE_METRIC_METHOD:
        return nlopt::LD_VAR1;

    default:
        throw std::invalid_argument("Unknown NLopt algorithm");
    }
}
#endif

nlopt_algorithms::nlopt_algorithms(
    ObjFunc func, ConFunc func_con, std::vector<double> lb, std::vector<double> ub)
    : objfun_(std::move(func)), confun_(std::move(func_con)), bl_(std::move(lb)), bu_(std::move(ub))
{
}

bool nlopt_algorithms::is_supported()
{
#if QUARISMA_MODULE_ENABLE_QUARISMA_nlopt
    return true;
#else
    return false;
#endif
};

void nlopt_algorithms::solve(std::vector<double>& parameters, const solver_options_nlopt& options)
{
    try
    {
#if QUARISMA_MODULE_ENABLE_QUARISMA_nlopt
        nlopt::opt optimizer(getNloptEnum(options.nloptal()), static_cast<int>(parameters.size()));

        optimizer.set_min_objective(&nlopt_algorithms::OBJFUN, this);

        optimizer.set_lower_bounds(bl_);
        optimizer.set_upper_bounds(bu_);
        if (confun_ != nullptr)
        {
            optimizer.add_inequality_constraint(&nlopt_algorithms::CONFUN, this);
        }

        optimizer.set_xtol_rel(options.parameter_tolerance());
        optimizer.set_ftol_rel(options.function_tolerance());

        double objf;

        nlopt::result result = optimizer.optimize(parameters, objf);

        if (result < 0)
        {
            throw std::runtime_error("NLOpt failed to find an optimal solution");
        }
#else
        QUARISMA_NOT_IMPLEMENTED("Not implemented");
#endif
    }
    catch (const std::exception& e)
    {
        // Handle or rethrow the exception as needed
        throw std::runtime_error(std::string("Optimization failed: ") + e.what());
    }
}

double nlopt_algorithms::OBJFUN(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
    auto* self = static_cast<nlopt_algorithms*>(data);

    vector<double> x_tmp(x);
    vector<double> grad_tmp(grad);

    double result = self->objfun_(x_tmp, grad_tmp);

    return result;
}

double nlopt_algorithms::CONFUN(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
    auto*          self = static_cast<nlopt_algorithms*>(data);
    vector<double> x_tmp(x);
    vector<double> grad_tmp(grad);

    double result = self->confun_(x_tmp, grad_tmp);

    return result;
}
}  // namespace quarisma
