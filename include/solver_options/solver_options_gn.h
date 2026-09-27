#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "detail/support.h"
#include "solver_options/solver_options.h"

namespace solverslib
{
class SOLVER_VISIBILITY solver_options_gn : public solver_options
{
    friend class solver_options_gn_builder;

public:
    SOLVER_API double bump() const;
    SOLVER_API size_t max_line_search_iterations() const;
    SOLVER_API double line_search_backtracking_factor() const;
    SOLVER_API double line_search_sufficient_decrease() const;

private:
    solver_options_gn();

    double bump_                            = 0.00001;
    size_t max_line_search_iterations_      = 20;
    double line_search_backtracking_factor_ = 0.5;
    double line_search_sufficient_decrease_ = 1e-4;
};

class SOLVER_VISIBILITY solver_options_gn_builder
{
public:
    SOLVER_API solver_options_gn_builder();

    SOLVER_API solver_options_gn_builder& with_max_iterations(int val);
    SOLVER_API solver_options_gn_builder& with_function_tolerance(double val);
    SOLVER_API solver_options_gn_builder& with_gradient_tolerance(double val);
    SOLVER_API solver_options_gn_builder& with_parameter_tolerance(double val);
    SOLVER_API solver_options_gn_builder& with_verbose(bool val = true);
    SOLVER_API solver_options_gn_builder& with_bump(double val);
    SOLVER_API solver_options_gn_builder& with_max_line_search_iterations(size_t val);
    SOLVER_API solver_options_gn_builder& with_line_search_backtracking_factor(double val);
    SOLVER_API solver_options_gn_builder& with_line_search_sufficient_decrease(double val);
    SOLVER_API solver_options_gn_builder& with_log_file(const std::string& val);
    [[deprecated("Use problem.derivatives() to configure derivative computation")]]
    SOLVER_API solver_options_gn_builder& with_aad_jacobian(bool val = true);

    SOLVER_API std::shared_ptr<const solver_options_gn> build() const;

private:
    std::shared_ptr<solver_options_gn> options_;
};

}  // namespace solverslib
