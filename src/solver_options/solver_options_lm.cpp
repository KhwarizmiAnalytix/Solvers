#include "solver_options/solver_options_lm.h"

namespace solverslib
{
solver_options_lm::solver_options_lm() : solver_options(solver_enum::LM) {}

bool   solver_options_lm::accept_uphill_step() const { return accept_uphill_step_; }
bool   solver_options_lm::use_geodesic() const { return use_geodesic_; }
double solver_options_lm::alpha() const { return alpha_; }
double solver_options_lm::lambda() const { return lambda_; }
double solver_options_lm::nu() const { return nu_; }
double solver_options_lm::lambda_down_fac() const { return lambda_down_fac_; }
double solver_options_lm::lambda_up_fac() const { return lambda_up_fac_; }
double solver_options_lm::epsilon() const { return epsilon_; }
double solver_options_lm::bump() const { return bump_; }
levenberg_marquardt_solver_enum solver_options_lm::type() const { return type_; }

solver_options_lm_builder::solver_options_lm_builder()
    : options_(std::shared_ptr<solver_options_lm>(new solver_options_lm()))
{
}

solver_options_lm_builder& solver_options_lm_builder::with_max_iterations(int val)
{
    options_->max_num_iterations_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_function_tolerance(double val)
{
    options_->function_tolerance_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_gradient_tolerance(double val)
{
    options_->gradient_tolerance_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_parameter_tolerance(double val)
{
    options_->parameter_tolerance_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_verbose(bool val)
{
    options_->verbose_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_accept_uphill_step(bool val)
{
    options_->accept_uphill_step_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_use_geodesic(bool val)
{
    options_->use_geodesic_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_alpha(double val)
{
    options_->alpha_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_lambda(double val)
{
    options_->lambda_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_nu(double val)
{
    options_->nu_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_lambda_down_fac(double val)
{
    options_->lambda_down_fac_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_lambda_up_fac(double val)
{
    options_->lambda_up_fac_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_epsilon(double val)
{
    options_->epsilon_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_bump(double val)
{
    options_->bump_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_type(levenberg_marquardt_solver_enum val)
{
    options_->type_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_log_file(const std::string& val)
{
    options_->log_file_ = val;
    return *this;
}

std::shared_ptr<const solver_options_lm> solver_options_lm_builder::build() const
{
    return options_;
}
}  // namespace solverslib
