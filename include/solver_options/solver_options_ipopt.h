#pragma once

#include <limits>
#include <memory>
#include <string>

#include "detail/support.h"
#include "solver_options/solver_options.h"

namespace solverslib
{
enum class ipopt_hessian_approximation_enum : int
{
    EXACT          = 0,
    LIMITED_MEMORY = 1
};

class SOLVER_VISIBILITY solver_options_ipopt : public solver_options
{
    friend class solver_options_ipopt_builder;

public:
    SOLVER_API double                           tol() const;
    SOLVER_API double                           acceptable_tol() const;
    SOLVER_API ipopt_hessian_approximation_enum hessian_approximation() const;
    SOLVER_API const std::string&               linear_solver() const;
    SOLVER_API double                           max_wall_time_seconds() const;

private:
    solver_options_ipopt();

    double                           tol_            = 1e-8;
    double                           acceptable_tol_ = 1e-6;
    ipopt_hessian_approximation_enum hessian_approximation_ =
        ipopt_hessian_approximation_enum::LIMITED_MEMORY;
    std::string linear_solver_         = "";
    double      max_wall_time_seconds_ = 1e9;
};

class SOLVER_VISIBILITY solver_options_ipopt_builder
{
public:
    SOLVER_API solver_options_ipopt_builder();

    SOLVER_API solver_options_ipopt_builder& with_max_iterations(int val);
    SOLVER_API solver_options_ipopt_builder& with_function_tolerance(double val);
    SOLVER_API solver_options_ipopt_builder& with_gradient_tolerance(double val);
    SOLVER_API solver_options_ipopt_builder& with_parameter_tolerance(double val);
    SOLVER_API solver_options_ipopt_builder& with_verbose(bool val = true);
    SOLVER_API solver_options_ipopt_builder& with_tol(double val);
    SOLVER_API solver_options_ipopt_builder& with_acceptable_tol(double val);
    SOLVER_API solver_options_ipopt_builder& with_hessian_approximation(
        ipopt_hessian_approximation_enum val);
    SOLVER_API solver_options_ipopt_builder& with_linear_solver(const std::string& val);
    SOLVER_API solver_options_ipopt_builder& with_max_wall_time_seconds(double val);
    SOLVER_API solver_options_ipopt_builder& with_log_file(const std::string& val);
    [[deprecated("Use problem.derivatives() to configure derivative computation")]]
    SOLVER_API solver_options_ipopt_builder& with_aad_jacobian(bool val = true);

    SOLVER_API std::shared_ptr<const solver_options_ipopt> build() const;

private:
    std::shared_ptr<solver_options_ipopt> options_;
};
}  // namespace solverslib
