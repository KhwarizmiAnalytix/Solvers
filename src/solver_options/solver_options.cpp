#include "solver_options/solver_options.h"

namespace solverslib
{
solver_options::solver_options(solver_enum solver,
    int                                    max_num_iterations,
    double                                 function_tolerance,
    double                                 gradient_tolerance,
    double                                 parameter_tolerance,
    bool                                   verbose)
    : solver_(solver), max_num_iterations_(max_num_iterations),
      function_tolerance_(function_tolerance), gradient_tolerance_(gradient_tolerance),
      parameter_tolerance_(parameter_tolerance), verbose_(verbose)
{
}

solver_options::solver_options(solver_enum solver) : solver_(solver) {}

solver_enum        solver_options::solver() const { return solver_; }
int                solver_options::max_num_iterations() const { return max_num_iterations_; }
double             solver_options::function_tolerance() const { return function_tolerance_; }
double             solver_options::gradient_tolerance() const { return gradient_tolerance_; }
double             solver_options::parameter_tolerance() const { return parameter_tolerance_; }
bool               solver_options::verbose() const noexcept { return verbose_; }
const std::string& solver_options::log_file() const { return log_file_; }
}  // namespace solverslib
