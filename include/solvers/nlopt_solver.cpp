#include "nlopt_solver.h"

#if SOLVERS_HAS_NLOPT
#include <nlopt.hpp>
#endif

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "detail/support.h"
#include "detail/support.h"
#include "solver_options/solver_options_nlopt.h"

namespace solverslib
{
#if SOLVERS_HAS_NLOPT
// Function to map nlopt_algo_name_enum enum to nlopt::algorithm enum
nlopt::algorithm getNloptEnum(nlopt_algo_name_enum algo)
{
    switch (algo)
    {
    case nlopt_algo_name_enum::AUGMENTED_LAGRANGIAN:
        return nlopt::LD_AUGLAG;
    case nlopt_algo_name_enum::AUGMENTED_LAGRANGIAN_WITH_EQUALITY_CONSTRAINTS:
    case nlopt_algo_name_enum::AUGMENTED_LAGRANGIAN_WITH_COBYLA:
    case nlopt_algo_name_enum::AUGMENTED_LAGRANGIAN_WITH_BOBYQA:
        return nlopt::LD_AUGLAG_EQ;  // Adjust depending on the inner solver

    case nlopt_algo_name_enum::METHOD_OF_MOVING_ASYMPTOTES:
        return nlopt::LD_MMA;
    case nlopt_algo_name_enum::CONSTRAINED_OPTIMIZATION_BY_LINEAR_APPROXIMATIONS:
        return nlopt::LN_COBYLA;
    case nlopt_algo_name_enum::SEQUENTIAL_LEAST_SQUARES_PROGRAMMING:
        return nlopt::LD_SLSQP;
    case nlopt_algo_name_enum::BOUND_OPTIMIZATION_BY_QUADRATIC_APPROXIMATION:
        return nlopt::LN_BOBYQA;
    case nlopt_algo_name_enum::LBFGS:
        return nlopt::LD_LBFGS;

    case nlopt_algo_name_enum::IMPROVED_STOCHASTIC_RANKING_EVOLUTION_STRATEGY:
        return nlopt::GN_ISRES;
    case nlopt_algo_name_enum::CONTROLLED_RANDOM_SEARCH_WITH_LOCAL_MUTATION:
        return nlopt::GN_CRS2_LM;
    case nlopt_algo_name_enum::DIVIDING_RECTANGLES:
        return nlopt::GN_ORIG_DIRECT_L;

    case nlopt_algo_name_enum::PRECONDITIONED_TRUNCATED_NEWTON_METHOD:
        return nlopt::LD_TNEWTON_PRECOND;
    case nlopt_algo_name_enum::VARIABLE_METRIC_METHOD:
        return nlopt::LD_VAR1;

    default:
        throw std::invalid_argument("Unknown NLopt algorithm");
    }
}
#endif

nlopt_solver::nlopt_solver(
    size_t              num_parameters,
    size_t              num_residuals,
    ObjFunc             func,
    ObjFunc_aad         func_aad,
    std::vector<double> lb,
    std::vector<double> ub)
    : num_parameters_(num_parameters),
      num_residuals_(num_residuals),
      objfun_(std::move(func)),
      objfun_aad_(std::move(func_aad)),
      lower_bounds_(std::move(lb)),
      upper_bounds_(std::move(ub))
{
    if (objfun_aad_ == nullptr)
    {
        double bump = 1e-8;

        objfun_aad_ = [this, bump](Eigen::VectorXd const& x, Eigen::MatrixXd& dy_dx)
        {
            auto number_of_parameters = x.size();

            SOLVERS_CHECK(dy_dx.cols() == number_of_parameters);

            auto number_of_targets = dy_dx.rows();

            Eigen::VectorXd y_plus(number_of_targets);
            Eigen::VectorXd y_minus(number_of_targets);

            Eigen::VectorXd x_tmp(number_of_parameters);
            x_tmp = x;

            for (size_t i = 0; i < number_of_parameters; ++i)
            {
                x_tmp[i] += bump;

                objfun_(x_tmp, y_plus);

                x_tmp[i] -= 2 * bump;
                objfun_(x_tmp, y_minus);

                for (size_t j = 0; j < y_plus.size(); ++j)
                {
                    dy_dx(j, i) = 0.5 * (y_plus[j] - y_minus[j]) / bump;
                }

                x_tmp[i] = x[i];
            }
        };
    }
}

bool nlopt_solver::is_supported()
{
#if SOLVERS_HAS_NLOPT
    return true;
#else
    return false;
#endif
};

void nlopt_solver::solve(
    SOLVERS_UNUSED std::vector<double>&        parameters,
    SOLVERS_UNUSED const solver_options_nlopt& options)
{
#if SOLVERS_HAS_NLOPT
    nlopt::opt optimizer(getNloptEnum(options.nloptal()), static_cast<int>(parameters.size()));

    optimizer.set_min_objective(&nlopt_solver::OBJFUN, this);
    if (!upper_bounds_.empty() && !lower_bounds_.empty())
    {
        optimizer.set_lower_bounds(lower_bounds_);
        optimizer.set_upper_bounds(upper_bounds_);
    }
    /*if (confun_ != nullptr)
        {
            optimizer.add_inequality_constraint(&nlopt_solver::CONFUN, this);
        }*/

    optimizer.set_xtol_rel(options.parameter_tolerance());
    optimizer.set_ftol_rel(options.function_tolerance());

    double objf;

    nlopt::result result = optimizer.optimize(parameters, objf);

    if (result < 0)
    {
        throw std::runtime_error("NLOpt failed to find an optimal solution");
    }
#else
    SOLVERS_NOT_IMPLEMENTED("Not implemented");
#endif
}

double nlopt_solver::OBJFUN(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
    auto* self = static_cast<nlopt_solver*>(data);

    Eigen::VectorXd x_tmp = Eigen::Map<const Eigen::VectorXd>(x.data(), x.size());
    Eigen::VectorXd f_tmp(self->num_residuals_);

    self->objfun_(x_tmp, f_tmp);

    double result = f_tmp.squaredNorm();

    if (!grad.empty())
    {
        Eigen::VectorXd grad_tmp = Eigen::Map<const Eigen::VectorXd>(grad.data(), grad.size());

        Eigen::MatrixXd f_aad_tmp(self->num_residuals_, self->num_parameters_);
        self->objfun_aad_(x_tmp, f_aad_tmp);
        grad_tmp = 2. * (f_aad_tmp.transpose() * f_tmp);
        Eigen::Map<Eigen::VectorXd>(grad.data(), grad.size()) = grad_tmp;
    }

    return result;
}

double nlopt_solver::CONFUN(
    SOLVERS_UNUSED const std::vector<double>& x,
    SOLVERS_UNUSED std::vector<double>& grad,
    SOLVERS_UNUSED void*                data)
{
    //auto*          self = static_cast<nlopt_solver*>(data);
    //Eigen::VectorXd x_tmp = Eigen::Map<const Eigen::VectorXd>(x.data(), x.size());
    //Eigen::VectorXd grad_tmp = Eigen::Map<const Eigen::VectorXd>(grad.data(), grad.size());

    double result = 0.;
    //self->confun_(x_tmp, grad_tmp);

    return result;
}
}  // namespace solverslib
