#pragma once

#include <memory>

#include "detail/support.h"
#include "solver_options/solver_options.h"

namespace solverslib
{
class SOLVER_VISIBILITY solver_options_rnc_lm : public solver_options
{
    friend class solver_options_rnc_lm_builder;

public:
    SOLVER_API int    order() const;
    SOLVER_API int    max_curve_trials() const;
    SOLVER_API double acceptance_threshold() const;
    SOLVER_API double contraction_min() const;
    SOLVER_API double contraction_max() const;
    SOLVER_API double initial_damping() const;
    SOLVER_API double damping_floor() const;
    SOLVER_API double damping_ceiling() const;
    SOLVER_API double diagonal_scaling_floor() const;

    solver_options_rnc_lm();

    int    order_                  = 3;
    int    max_curve_trials_       = 4;
    double acceptance_threshold_   = 1e-4;
    double contraction_min_        = 0.3;
    double contraction_max_        = 0.5;
    double initial_damping_        = 1e-4;
    double damping_floor_          = 1e-15;
    double damping_ceiling_        = 1e12;
    double diagonal_scaling_floor_ = 1e-12;
};

class SOLVER_VISIBILITY solver_options_rnc_lm_builder
{
public:
    SOLVER_API solver_options_rnc_lm_builder();

    SOLVER_API solver_options_rnc_lm_builder& with_max_iterations(int val);
    SOLVER_API solver_options_rnc_lm_builder& with_function_tolerance(double val);
    SOLVER_API solver_options_rnc_lm_builder& with_gradient_tolerance(double val);
    SOLVER_API solver_options_rnc_lm_builder& with_parameter_tolerance(double val);
    SOLVER_API solver_options_rnc_lm_builder& with_verbose(bool val = true);
    SOLVER_API solver_options_rnc_lm_builder& with_log_file(const std::string& val);
    SOLVER_API solver_options_rnc_lm_builder& with_order(int val);
    SOLVER_API solver_options_rnc_lm_builder& with_max_curve_trials(int val);
    SOLVER_API solver_options_rnc_lm_builder& with_acceptance_threshold(double val);
    SOLVER_API solver_options_rnc_lm_builder& with_contraction_min(double val);
    SOLVER_API solver_options_rnc_lm_builder& with_contraction_max(double val);
    SOLVER_API solver_options_rnc_lm_builder& with_initial_damping(double val);
    SOLVER_API solver_options_rnc_lm_builder& with_damping_floor(double val);
    SOLVER_API solver_options_rnc_lm_builder& with_damping_ceiling(double val);
    SOLVER_API solver_options_rnc_lm_builder& with_diagonal_scaling_floor(double val);
    SOLVER_API std::shared_ptr<const solver_options_rnc_lm> build() const;

private:
    std::shared_ptr<solver_options_rnc_lm> options_;
};
}  // namespace solverslib
