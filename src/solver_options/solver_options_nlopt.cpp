#include "solver_options/solver_options_nlopt.h"

namespace solverslib
{
solver_options_nlopt::solver_options_nlopt() : solver_options(solver_enum::NLOPT) {}

nlopt_algorithm_enum solver_options_nlopt::algorithm() const
{
    return algorithm_;
}

double solver_options_nlopt::xtol_rel() const
{
    return xtol_rel_;
}

double solver_options_nlopt::ftol_rel() const
{
    return ftol_rel_;
}

double solver_options_nlopt::max_time() const
{
    return max_time_;
}

solver_options_nlopt_builder::solver_options_nlopt_builder()
    : options_(std::shared_ptr<solver_options_nlopt>(new solver_options_nlopt()))
{
}

solver_options_nlopt_builder& solver_options_nlopt_builder::with_max_iterations(int val)
{
    options_->max_num_iterations_ = val;
    return *this;
}

solver_options_nlopt_builder& solver_options_nlopt_builder::with_function_tolerance(double val)
{
    options_->ftol_rel_ = val;
    return *this;
}

solver_options_nlopt_builder& solver_options_nlopt_builder::with_gradient_tolerance(double val)
{
    return *this;
}

solver_options_nlopt_builder& solver_options_nlopt_builder::with_parameter_tolerance(double val)
{
    options_->xtol_rel_ = val;
    return *this;
}

solver_options_nlopt_builder& solver_options_nlopt_builder::with_verbose(bool val)
{
    options_->verbose_ = val;
    return *this;
}

solver_options_nlopt_builder& solver_options_nlopt_builder::with_algorithm(nlopt_algorithm_enum val)
{
    options_->algorithm_ = val;
    return *this;
}

solver_options_nlopt_builder& solver_options_nlopt_builder::with_xtol_rel(double val)
{
    options_->xtol_rel_ = val;
    return *this;
}

solver_options_nlopt_builder& solver_options_nlopt_builder::with_ftol_rel(double val)
{
    options_->ftol_rel_ = val;
    return *this;
}

solver_options_nlopt_builder& solver_options_nlopt_builder::with_max_time(double val)
{
    options_->max_time_ = val;
    return *this;
}

std::shared_ptr<const solver_options_nlopt> solver_options_nlopt_builder::build() const
{
    return options_;
}
}  // namespace solverslib
