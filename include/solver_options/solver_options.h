#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "include/detail/support.h"
#include "include/solver_enum.h"

namespace solverslib
{
class MATH_VISIBILITY solver_options
{
protected:
    solver_options(
        solver_enum solver,
        int         max_num_iterations,
        double      function_tolerance,
        double      gradient_tolerance,
        double      parameter_tolerance,
        bool        verbose,
        bool        aad_jacobian = true)
        : solver_(solver),
          max_num_iterations_(max_num_iterations),
          function_tolerance_(function_tolerance),
          gradient_tolerance_(gradient_tolerance),
          parameter_tolerance_(parameter_tolerance),
          verbose_(verbose),
          aad_jacobian_(aad_jacobian)

    {
    }

public:
    virtual ~solver_options() = default;

    // Getters
    solver_enum        solver() const { return solver_; }
    int                max_num_iterations() const { return max_num_iterations_; }
    double             function_tolerance() const { return function_tolerance_; }
    double             gradient_tolerance() const { return gradient_tolerance_; }
    double             parameter_tolerance() const { return parameter_tolerance_; }
    bool               verbose() const noexcept { return verbose_; }
    bool               aad_jacobian() const noexcept { return aad_jacobian_; }
    const std::string& log_file() const { return log_file_; }
    // Setters
    void set_solver(solver_enum solver) { solver_ = solver; }
    void set_max_num_iterations(int max_num_iterations)
    {
        max_num_iterations_ = max_num_iterations;
    }
    void set_function_tolerance(double function_tolerance)
    {
        function_tolerance_ = function_tolerance;
    }
    void set_gradient_tolerance(double gradient_tolerance)
    {
        gradient_tolerance_ = gradient_tolerance;
    }
    void set_parameter_tolerance(double parameter_tolerance)
    {
        parameter_tolerance_ = parameter_tolerance;
    }
    void set_debug(bool verbose) { verbose_ = verbose; }

protected:
    solver_options(solver_enum solver) : solver_(solver) {}
    solver_options(solver_enum solver, bool aad_jacobian)
        : solver_(solver), aad_jacobian_(aad_jacobian)
    {
    }

    solver_enum solver_;
    int         max_num_iterations_;
    double      function_tolerance_  = std::numeric_limits<double>::epsilon();
    double      gradient_tolerance_  = 0.0;
    double      parameter_tolerance_ = 0.0;
    bool        verbose_             = false;
    bool        aad_jacobian_        = true;
    std::string log_file_            = "";  // Empty string means no file logging
};
}  // namespace solverslib
