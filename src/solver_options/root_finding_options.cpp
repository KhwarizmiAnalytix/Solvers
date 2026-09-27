#include "solver_options/root_finding_options.h"

namespace solverslib
{
size_t root_finding_options::max_iterations() const { return max_iterations_; }
double root_finding_options::tolerance_function() const { return tolerance_function_; }
double root_finding_options::tolerance_parameter() const { return tolerance_parameter_; }
double root_finding_options::function_offset() const { return function_offset_; }

root_finding_options_builder::root_finding_options_builder() = default;

root_finding_options_builder& root_finding_options_builder::with_max_iterations(size_t val)
{
    options_.max_iterations_ = val;
    return *this;
}
root_finding_options_builder& root_finding_options_builder::with_tolerance_function(double val)
{
    options_.tolerance_function_ = val;
    return *this;
}
root_finding_options_builder& root_finding_options_builder::with_tolerance_parameter(double val)
{
    options_.tolerance_parameter_ = val;
    return *this;
}
root_finding_options_builder& root_finding_options_builder::with_function_offset(double val)
{
    options_.function_offset_ = val;
    return *this;
}

root_finding_options root_finding_options_builder::build() const { return options_; }
}  // namespace solverslib
