#pragma once

#include <limits>
#include <memory>
#include <string>

#include "detail/support.h"
#include "solver_options/solver_options.h"

namespace solverslib
{
enum class nlopt_algorithm_enum : int
{
    LD_LBFGS       = 0,
    LD_MMA         = 1,
    LD_SLSQP       = 2,
    LD_CCSAQ       = 3,
    LN_NELDERMEAD  = 4,
    LN_SBPLX       = 5,
};

class SOLVER_VISIBILITY solver_options_nlopt : public solver_options
{
    friend class solver_options_nlopt_builder;

public:
    SOLVER_API nlopt_algorithm_enum algorithm() const;
    SOLVER_API double               xtol_rel() const;
    SOLVER_API double               ftol_rel() const;
    SOLVER_API double               max_time() const;

private:
    solver_options_nlopt();

    nlopt_algorithm_enum algorithm_   = nlopt_algorithm_enum::LD_LBFGS;
    double               xtol_rel_    = 1e-4;
    double               ftol_rel_    = 1e-4;
    double               max_time_    = 1e9;
};

class solver_options_nlopt_builder
{
public:
    SOLVER_API solver_options_nlopt_builder();

    SOLVER_API solver_options_nlopt_builder& with_max_iterations(int val);
    SOLVER_API solver_options_nlopt_builder& with_function_tolerance(double val);
    SOLVER_API solver_options_nlopt_builder& with_gradient_tolerance(double val);
    SOLVER_API solver_options_nlopt_builder& with_parameter_tolerance(double val);
    SOLVER_API solver_options_nlopt_builder& with_verbose(bool val = true);
    SOLVER_API solver_options_nlopt_builder& with_algorithm(nlopt_algorithm_enum val);
    SOLVER_API solver_options_nlopt_builder& with_xtol_rel(double val);
    SOLVER_API solver_options_nlopt_builder& with_ftol_rel(double val);
    SOLVER_API solver_options_nlopt_builder& with_max_time(double val);
    SOLVER_API std::shared_ptr<const solver_options_nlopt> build() const;

private:
    std::shared_ptr<solver_options_nlopt> options_;
};
}  // namespace solverslib
