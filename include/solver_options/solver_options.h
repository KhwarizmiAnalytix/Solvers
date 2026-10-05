#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include "detail/support.h"
#include "solvers/api/status.h"

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
    SOLVER_API const std::string& log_file() const;

protected:
    solver_options(solver_enum solver,
        int                    max_num_iterations,
        double                 function_tolerance,
        double                 gradient_tolerance,
        double                 parameter_tolerance,
        bool                   verbose);

    solver_options(solver_enum solver);

    solver_enum solver_;
    int         max_num_iterations_ = 100;
    double      function_tolerance_  = std::numeric_limits<double>::epsilon();
    double      gradient_tolerance_  = 0.0;
    double      parameter_tolerance_ = 0.0;
    bool        verbose_             = false;
    std::string log_file_;
};

// Forward declarations
class solver_options_lm;
class solver_options_rnc_lm;
class solver_options_ceres;
class solver_options_ipopt;
class solver_options_petsc;

// Minimal solve options - users choose which solver and configure it directly
struct solve_options
{
    api::algorithm algorithm = api::algorithm::automatic;
    api::backend   backend   = api::backend::automatic;
    api::derivative_mode derivatives = api::derivative_mode::automatic;

    // Defaults when building solver options
    int    max_iterations        = 100;
    int    max_function_evaluations = 0;
    double function_tolerance    = std::numeric_limits<double>::epsilon();
    double gradient_tolerance    = 0.0;
    double parameter_tolerance   = std::numeric_limits<double>::epsilon();
    bool   verbose               = false;

    // Optional backend-specific configurations
    std::optional<std::shared_ptr<const solver_options_lm>>     lm;
    std::optional<std::shared_ptr<const solver_options_rnc_lm>> rnc_lm;
    std::optional<std::shared_ptr<const solver_options_ceres>>  ceres;
    std::optional<std::shared_ptr<const solver_options_ipopt>>  ipopt;
    std::optional<std::shared_ptr<const solver_options_petsc>>  petsc_tao;
};
}  // namespace solverslib
