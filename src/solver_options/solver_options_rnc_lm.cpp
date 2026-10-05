#include "solver_options/solver_options_rnc_lm.h"

namespace solverslib
{
solver_options_rnc_lm::solver_options_rnc_lm()
    : solver_options(solver_enum::LM)
{
}

int solver_options_rnc_lm::order() const
{
    return order_;
}

int solver_options_rnc_lm::max_curve_trials() const
{
    return max_curve_trials_;
}

double solver_options_rnc_lm::acceptance_threshold() const
{
    return acceptance_threshold_;
}

double solver_options_rnc_lm::contraction_min() const
{
    return contraction_min_;
}

double solver_options_rnc_lm::contraction_max() const
{
    return contraction_max_;
}

double solver_options_rnc_lm::initial_damping() const
{
    return initial_damping_;
}

double solver_options_rnc_lm::damping_floor() const
{
    return damping_floor_;
}

double solver_options_rnc_lm::damping_ceiling() const
{
    return damping_ceiling_;
}

double solver_options_rnc_lm::diagonal_scaling_floor() const
{
    return diagonal_scaling_floor_;
}

solver_options_rnc_lm_builder::solver_options_rnc_lm_builder()
    : options_(std::make_shared<solver_options_rnc_lm>())
{
}

solver_options_rnc_lm_builder& solver_options_rnc_lm_builder::with_max_iterations(int val)
{
    options_->max_num_iterations_ = val;
    return *this;
}

solver_options_rnc_lm_builder& solver_options_rnc_lm_builder::with_function_tolerance(double val)
{
    options_->function_tolerance_ = val;
    return *this;
}

solver_options_rnc_lm_builder& solver_options_rnc_lm_builder::with_gradient_tolerance(double val)
{
    options_->gradient_tolerance_ = val;
    return *this;
}

solver_options_rnc_lm_builder& solver_options_rnc_lm_builder::with_parameter_tolerance(double val)
{
    options_->parameter_tolerance_ = val;
    return *this;
}

solver_options_rnc_lm_builder& solver_options_rnc_lm_builder::with_verbose(bool val)
{
    options_->verbose_ = val;
    return *this;
}

solver_options_rnc_lm_builder& solver_options_rnc_lm_builder::with_log_file(
    const std::string& val)
{
    options_->log_file_ = val;
    return *this;
}

solver_options_rnc_lm_builder& solver_options_rnc_lm_builder::with_order(int val)
{
    const_cast<solver_options_rnc_lm&>(*options_).order_ = val;
    return *this;
}

solver_options_rnc_lm_builder& solver_options_rnc_lm_builder::with_max_curve_trials(int val)
{
    const_cast<solver_options_rnc_lm&>(*options_).max_curve_trials_ = val;
    return *this;
}

solver_options_rnc_lm_builder& solver_options_rnc_lm_builder::with_acceptance_threshold(
    double val)
{
    const_cast<solver_options_rnc_lm&>(*options_).acceptance_threshold_ = val;
    return *this;
}

solver_options_rnc_lm_builder& solver_options_rnc_lm_builder::with_contraction_min(double val)
{
    const_cast<solver_options_rnc_lm&>(*options_).contraction_min_ = val;
    return *this;
}

solver_options_rnc_lm_builder& solver_options_rnc_lm_builder::with_contraction_max(double val)
{
    const_cast<solver_options_rnc_lm&>(*options_).contraction_max_ = val;
    return *this;
}

solver_options_rnc_lm_builder& solver_options_rnc_lm_builder::with_initial_damping(double val)
{
    const_cast<solver_options_rnc_lm&>(*options_).initial_damping_ = val;
    return *this;
}

solver_options_rnc_lm_builder& solver_options_rnc_lm_builder::with_damping_floor(double val)
{
    const_cast<solver_options_rnc_lm&>(*options_).damping_floor_ = val;
    return *this;
}

solver_options_rnc_lm_builder& solver_options_rnc_lm_builder::with_damping_ceiling(double val)
{
    const_cast<solver_options_rnc_lm&>(*options_).damping_ceiling_ = val;
    return *this;
}

solver_options_rnc_lm_builder& solver_options_rnc_lm_builder::with_diagonal_scaling_floor(
    double val)
{
    const_cast<solver_options_rnc_lm&>(*options_).diagonal_scaling_floor_ = val;
    return *this;
}

std::shared_ptr<const solver_options_rnc_lm> solver_options_rnc_lm_builder::build() const
{
    return options_;
}
}  // namespace solverslib
