#ifndef SOLVERS_ROOT_FINDING_OPTIONS_H_
#define SOLVERS_ROOT_FINDING_OPTIONS_H_

#include <cstddef>
#include <limits>

#include "detail/support.h"

namespace solverslib
{
class SOLVER_VISIBILITY root_finding_options
{
    friend class root_finding_options_builder;

public:
    root_finding_options() = default;

    SOLVER_API size_t max_iterations() const;
    SOLVER_API double tolerance_function() const;
    SOLVER_API double tolerance_parameter() const;
    SOLVER_API double function_offset() const;

private:
    size_t max_iterations_      = 50;
    double tolerance_function_  = std::numeric_limits<double>::epsilon();
    double tolerance_parameter_ = std::numeric_limits<double>::epsilon();
    double function_offset_     = 0.0;
};

class SOLVER_VISIBILITY root_finding_options_builder
{
public:
    SOLVER_API root_finding_options_builder();

    SOLVER_API root_finding_options_builder& with_max_iterations(size_t val);
    SOLVER_API root_finding_options_builder& with_tolerance_function(double val);
    SOLVER_API root_finding_options_builder& with_tolerance_parameter(double val);
    SOLVER_API root_finding_options_builder& with_function_offset(double val);

    SOLVER_API root_finding_options build() const;

private:
    root_finding_options options_;
};

}  // namespace solverslib

#endif  // SOLVERS_ROOT_FINDING_OPTIONS_H_
