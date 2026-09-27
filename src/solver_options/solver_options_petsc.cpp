#include "solver_options/solver_options_petsc.h"

namespace solverslib
{
solver_options_petsc::solver_options_petsc() : solver_options(solver_enum::PETSC_TAO) {}

tao_algorithm_enum solver_options_petsc::tao_type() const { return tao_type_; }
double             solver_options_petsc::gatol() const { return gatol_; }
double             solver_options_petsc::grtol() const { return grtol_; }
bool               solver_options_petsc::matrix_free() const { return matrix_free_; }

solver_options_petsc_builder::solver_options_petsc_builder()
    : options_(std::shared_ptr<solver_options_petsc>(new solver_options_petsc()))
{
}

solver_options_petsc_builder& solver_options_petsc_builder::with_tao_type(tao_algorithm_enum val)
{
    options_->tao_type_ = val;
    return *this;
}
solver_options_petsc_builder& solver_options_petsc_builder::with_max_iterations(int val)
{
    options_->max_num_iterations_ = val;
    return *this;
}
solver_options_petsc_builder& solver_options_petsc_builder::with_function_tolerance(double val)
{
    options_->function_tolerance_ = val;
    return *this;
}
solver_options_petsc_builder& solver_options_petsc_builder::with_gradient_tolerance(double val)
{
    options_->gradient_tolerance_ = val;
    return *this;
}
solver_options_petsc_builder& solver_options_petsc_builder::with_parameter_tolerance(double val)
{
    options_->parameter_tolerance_ = val;
    return *this;
}
solver_options_petsc_builder& solver_options_petsc_builder::with_verbose(bool val)
{
    options_->verbose_ = val;
    return *this;
}
solver_options_petsc_builder& solver_options_petsc_builder::with_gatol(double val)
{
    options_->gatol_ = val;
    return *this;
}
solver_options_petsc_builder& solver_options_petsc_builder::with_grtol(double val)
{
    options_->grtol_ = val;
    return *this;
}
solver_options_petsc_builder& solver_options_petsc_builder::with_matrix_free(bool val)
{
    options_->matrix_free_ = val;
    return *this;
}
solver_options_petsc_builder& solver_options_petsc_builder::with_log_file(const std::string& val)
{
    options_->log_file_ = val;
    return *this;
}
solver_options_petsc_builder& solver_options_petsc_builder::with_aad_jacobian(bool val)
{
    options_->aad_jacobian_ = val;
    return *this;
}

std::shared_ptr<const solver_options_petsc> solver_options_petsc_builder::build() const
{
    return options_;
}
}  // namespace solverslib
