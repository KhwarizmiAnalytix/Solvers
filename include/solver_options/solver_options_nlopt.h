#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "include/detail/support.h"
#include "include/detail/support.h"
#include "solver_options/solver_options.h"

namespace solverslib
{
enum class nlopt_algo_name_enum : int
{
    AUGMENTED_LAGRANGIAN                           = 1,
    AUGMENTED_LAGRANGIAN_WITH_EQUALITY_CONSTRAINTS = 2,
    AUGMENTED_LAGRANGIAN_WITH_COBYLA               = 3,
    AUGMENTED_LAGRANGIAN_WITH_BOBYQA               = 4,

    METHOD_OF_MOVING_ASYMPTOTES                       = 5,
    CONSTRAINED_OPTIMIZATION_BY_LINEAR_APPROXIMATIONS = 6,
    SEQUENTIAL_LEAST_SQUARES_PROGRAMMING              = 7,
    BOUND_OPTIMIZATION_BY_QUADRATIC_APPROXIMATION     = 8,
    LBFGS                                             = 9,

    IMPROVED_STOCHASTIC_RANKING_EVOLUTION_STRATEGY = 10,
    CONTROLLED_RANDOM_SEARCH_WITH_LOCAL_MUTATION   = 11,
    DIVIDING_RECTANGLES                            = 12,

    PRECONDITIONED_TRUNCATED_NEWTON_METHOD = 13,
    VARIABLE_METRIC_METHOD                 = 14
};

class MATH_VISIBILITY solver_options_nlopt : public solver_options
{
    friend class solver_options_nlopt_builder;

public:
    solver_options_nlopt(
        nlopt_algo_name_enum nloptal,
        int                  max_num_iterations,
        double               function_tolerance  = std::numeric_limits<double>::epsilon(),
        double               gradient_tolerance  = std::numeric_limits<double>::epsilon(),
        double               parameter_tolerance = std::numeric_limits<double>::epsilon(),
        bool                 verbose             = false)
        : solver_options(
              solver_enum::NLOPT,
              max_num_iterations,
              function_tolerance,
              gradient_tolerance,
              parameter_tolerance,
              verbose)
    {
        nloptal_ = nloptal;
    }

    // Getter and Setter for nlopt_algo_name_enum
    nlopt_algo_name_enum nloptal() const { return nloptal_; }
    void                 set_nloptal(nlopt_algo_name_enum nloptal) { nloptal_ = nloptal; }

private:
    solver_options_nlopt() : solver_options(solverslib::solver_enum::NLOPT) {};
    void initialize() const {};



    nlopt_algo_name_enum nloptal_ = nlopt_algo_name_enum::LBFGS;
};

/**
 * @brief Builder for solver_options_nlopt configuration
 *
 * Usage:
 *   auto options = solver_options_nlopt_builder()
 *       .with_max_iterations(100).with_function_tolerance(1e-6)
 *       .with_algorithm(nlopt_algo_name_enum::lbfgs_solver).build();
 */
class MATH_VISIBILITY solver_options_nlopt_builder
{
public:
    MATH_API solver_options_nlopt_builder()
        : options_(std::shared_ptr<solver_options_nlopt>(new solver_options_nlopt()))
    {
    }

    // Base solver options
    solverslib::solver_options_nlopt_builder& with_max_iterations(int val)
    {
        options_->set_max_num_iterations(val);
        return *this;
    }

    solverslib::solver_options_nlopt_builder& with_function_tolerance(double val)
    {
        options_->set_function_tolerance(val);
        return *this;
    }

    solverslib::solver_options_nlopt_builder& with_gradient_tolerance(double val)
    {
        options_->set_gradient_tolerance(val);
        return *this;
    }

    solverslib::solver_options_nlopt_builder& with_parameter_tolerance(double val)
    {
        options_->set_parameter_tolerance(val);
        return *this;
    }

    solverslib::solver_options_nlopt_builder& with_verbose(bool val = true)
    {
        options_->set_debug(val);
        return *this;
    }

    // NLopt-specific options
    solverslib::solver_options_nlopt_builder& with_algorithm(nlopt_algo_name_enum val)
    {
        options_->set_nloptal(val);
        return *this;
    }

    solverslib::solver_options_nlopt_builder& with_aad_jacobian(bool val)
    {
        options_->aad_jacobian_ = val;
        return *this;
    }

    solverslib::solver_options_nlopt_builder& with_log_file(const std::string& val)
    {
        options_->log_file_ = val;
        return *this;
    }
    // Build the final options
    MATH_API std::shared_ptr<const solver_options_nlopt> build() const { return options_; }

private:
    std::shared_ptr<solver_options_nlopt> options_;
};

}  // namespace solverslib
