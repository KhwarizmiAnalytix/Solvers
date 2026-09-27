#pragma once

#include <memory>
#include <string>

#include "detail/support.h"
#include "solver_options/solver_options.h"

namespace solverslib
{
enum class levenberg_marquardt_solver_enum : int
{
    LEVENBERG = 0,
    QUADRATIC = 1,
    NIELSEN   = 2
};

class SOLVER_VISIBILITY solver_options_lm : public solver_options
{
    friend class solver_options_lm_builder;

public:
    SOLVER_API bool   accept_uphill_step() const;
    SOLVER_API bool   use_geodesic() const;
    SOLVER_API double alpha() const;
    SOLVER_API double lambda() const;
    SOLVER_API double nu() const;
    SOLVER_API double lambda_down_fac() const;
    SOLVER_API double lambda_up_fac() const;
    SOLVER_API double epsilon() const;
    SOLVER_API double bump() const;
    SOLVER_API levenberg_marquardt_solver_enum type() const;

private:
    solver_options_lm();

    bool   accept_uphill_step_ = false;
    bool   use_geodesic_       = true;
    double alpha_              = 0.75;
    double lambda_             = 1e-4;
    double nu_                 = 2.0;
    double lambda_down_fac_    = 9.0;
    double lambda_up_fac_      = 11.0;
    double epsilon_            = 0.05;
    double bump_               = 0.00001;

    levenberg_marquardt_solver_enum type_ = levenberg_marquardt_solver_enum::NIELSEN;
};

class SOLVER_VISIBILITY solver_options_lm_builder
{
public:
    SOLVER_API solver_options_lm_builder();

    SOLVER_API solver_options_lm_builder& with_max_iterations(int val);
    SOLVER_API solver_options_lm_builder& with_function_tolerance(double val);
    SOLVER_API solver_options_lm_builder& with_gradient_tolerance(double val);
    SOLVER_API solver_options_lm_builder& with_parameter_tolerance(double val);
    SOLVER_API solver_options_lm_builder& with_verbose(bool val = true);
    SOLVER_API solver_options_lm_builder& with_accept_uphill_step(bool val = true);
    SOLVER_API solver_options_lm_builder& with_use_geodesic(bool val = true);
    SOLVER_API solver_options_lm_builder& with_alpha(double val);
    SOLVER_API solver_options_lm_builder& with_lambda(double val);
    SOLVER_API solver_options_lm_builder& with_nu(double val);
    SOLVER_API solver_options_lm_builder& with_lambda_down_fac(double val);
    SOLVER_API solver_options_lm_builder& with_lambda_up_fac(double val);
    SOLVER_API solver_options_lm_builder& with_epsilon(double val);
    SOLVER_API solver_options_lm_builder& with_bump(double val);
    SOLVER_API solver_options_lm_builder& with_type(levenberg_marquardt_solver_enum val);
    SOLVER_API solver_options_lm_builder& with_log_file(const std::string& val);
    SOLVER_API solver_options_lm_builder& with_aad_jacobian(bool val = true);

    SOLVER_API std::shared_ptr<const solver_options_lm> build() const;

private:
    std::shared_ptr<solver_options_lm> options_;
};

}  // namespace solverslib
