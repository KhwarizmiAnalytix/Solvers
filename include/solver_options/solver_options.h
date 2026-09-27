#pragma once

#include <limits>
#include <string>

#include "detail/support.h"

namespace solverslib
{
enum class solver_enum : int
{
    LM           = 0,
    LBFGS        = 1,
    CERES        = 3,
    GAUSS_NEWTON = 4,
    IPOPT        = 5,
    PETSC_TAO    = 6
};

class SOLVER_VISIBILITY solver_options
{
public:
    virtual ~solver_options() = default;

    SOLVER_API solver_enum        solver() const;
    SOLVER_API int                max_num_iterations() const;
    SOLVER_API double             function_tolerance() const;
    SOLVER_API double             gradient_tolerance() const;
    SOLVER_API double             parameter_tolerance() const;
    SOLVER_API bool               verbose() const noexcept;
    SOLVER_API bool               aad_jacobian() const noexcept;
    SOLVER_API const std::string& log_file() const;

protected:
    solver_options(solver_enum solver,
        int                    max_num_iterations,
        double                 function_tolerance,
        double                 gradient_tolerance,
        double                 parameter_tolerance,
        bool                   verbose,
        bool                   aad_jacobian = true);

    solver_options(solver_enum solver);
    solver_options(solver_enum solver, bool aad_jacobian);

    solver_enum solver_;
    int         max_num_iterations_ = 100;
    double      function_tolerance_  = std::numeric_limits<double>::epsilon();
    double      gradient_tolerance_  = 0.0;
    double      parameter_tolerance_ = 0.0;
    bool        verbose_             = false;
    bool        aad_jacobian_        = true;
    std::string log_file_;
};
}  // namespace solverslib
