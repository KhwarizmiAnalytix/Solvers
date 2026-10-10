#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <string>

#include "detail/support.h"
#include "solver_options/solver_options.h"

namespace solverslib
{
enum class lbfgs_line_search_method_type : int
{
    ARMIJO       = 1,
    WOLFE        = 2,
    STRONG_WOLFE = 3
};

enum class lbfgs_line_search_type : int
{
    BACKTRACKING   = 1,
    NOCEDAL_WRIGHT = 2,
    BRACKETING     = 3
};

class SOLVER_VISIBILITY solver_options_bfgs : public solver_options
{
    friend class solver_options_bfgs_builder;

public:
    SOLVER_API lbfgs_line_search_method_type method_type() const;
    SOLVER_API lbfgs_line_search_type        type() const;
    SOLVER_API size_t                        tau() const;
    SOLVER_API size_t                        max_iteration_linesearch() const;
    SOLVER_API double                        step_min() const;
    SOLVER_API double                        step_max() const;
    SOLVER_API double                        linesearch_tolerance() const;
    SOLVER_API double                        linesearch_wolfe() const;
    SOLVER_API double                        initial_step() const;
    SOLVER_API double                        backtracking_decrease() const;
    SOLVER_API double                        backtracking_increase() const;
    SOLVER_API double                        line_search_expansion() const;
    SOLVER_API double                        bump() const;
    SOLVER_API finite_difference_scale       difference_scale() const;

private:
    solver_options_bfgs();
    SOLVER_API void validate() const;

    lbfgs_line_search_method_type method_type_ = lbfgs_line_search_method_type::ARMIJO;
    lbfgs_line_search_type        type_        = lbfgs_line_search_type::NOCEDAL_WRIGHT;
    size_t                        tau_         = 6;
    size_t                        max_iteration_linesearch_ = 40;
    double                        step_min_                 = 0.;
    double                        step_max_                 = 1000000;
    double                        linesearch_tolerance_  = std::numeric_limits<double>::epsilon();
    double                        linesearch_wolfe_      = 0.9;
    double                        initial_step_          = 0.5;
    double                        backtracking_decrease_ = 0.5;
    double                        backtracking_increase_ = 2.1;
    double                        line_search_expansion_ = 5.0;
    double                        bump_                  = 0.000001;
    finite_difference_scale       difference_scale_      = finite_difference_scale::absolute;
};

class SOLVER_VISIBILITY solver_options_bfgs_builder
{
public:
    SOLVER_API solver_options_bfgs_builder();

    SOLVER_API solver_options_bfgs_builder& with_max_iterations(int val);
    SOLVER_API solver_options_bfgs_builder& with_function_tolerance(double val);
    SOLVER_API solver_options_bfgs_builder& with_gradient_tolerance(double val);
    SOLVER_API solver_options_bfgs_builder& with_parameter_tolerance(double val);
    SOLVER_API solver_options_bfgs_builder& with_verbose(bool val = true);
    SOLVER_API solver_options_bfgs_builder& with_method_type(lbfgs_line_search_method_type val);
    SOLVER_API solver_options_bfgs_builder& with_type(lbfgs_line_search_type val);
    SOLVER_API solver_options_bfgs_builder& with_tau(size_t val);
    SOLVER_API solver_options_bfgs_builder& with_max_iteration_linesearch(size_t val);
    SOLVER_API solver_options_bfgs_builder& with_step_min(double val);
    SOLVER_API solver_options_bfgs_builder& with_step_max(double val);
    SOLVER_API solver_options_bfgs_builder& with_linesearch_tolerance(double val);
    SOLVER_API solver_options_bfgs_builder& with_linesearch_wolfe(double val);
    SOLVER_API solver_options_bfgs_builder& with_initial_step(double val);
    SOLVER_API solver_options_bfgs_builder& with_backtracking_decrease(double val);
    SOLVER_API solver_options_bfgs_builder& with_backtracking_increase(double val);
    SOLVER_API solver_options_bfgs_builder& with_line_search_expansion(double val);
    SOLVER_API solver_options_bfgs_builder& with_bump(double val);
    SOLVER_API solver_options_bfgs_builder& with_difference_scale(finite_difference_scale val);
    SOLVER_API solver_options_bfgs_builder& with_log_file(const std::string& val);
    SOLVER_API std::shared_ptr<const solver_options_bfgs> build() const;

private:
    std::shared_ptr<solver_options_bfgs> options_;
};

}  // namespace solverslib
