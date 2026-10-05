#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "detail/support.h"
#include "solver_options/solver_options.h"

namespace solverslib
{
enum class levenberg_marquardt_solver_enum : int
{
    LEVENBERG_MARQUARDT     = 0,
    QUADRATIC_INTERPOLATION = 1,
    NIELSEN                 = 2
};

// How each damped step is computed; see damped_step_solver in eigen_support.h.
enum class levenberg_marquardt_linear_solver_enum : int
{
    NORMAL_LDLT  = 0,  // LDLT of J^T J + damping (default; fastest for m >> n)
    AUGMENTED_QR = 1   // QR of [J; sqrt(damping)]; robust for ill-conditioned J
};

class SOLVER_VISIBILITY solver_options_lm : public solver_options
{
    friend class solver_options_lm_builder;

public:
    // Transtrum & Sethna, arXiv:1201.5885, Sections 3–4.
    // Mathematical definitions and option mapping: docs/levenberg-marquardt.md.
    SOLVER_API bool   bold_acceptance() const;
    SOLVER_API double bold_acceptance_exponent() const;
    SOLVER_API bool   geodesic_acceleration() const;
    // alpha bounds ||a||/||v|| = 2||delta_theta_2||/||delta_theta_1||.
    SOLVER_API double                                 geodesic_acceleration_threshold() const;
    SOLVER_API double                                 geodesic_acceleration_step() const;
    SOLVER_API double                                 initial_damping() const;
    SOLVER_API double                                 initial_rejection_multiplier() const;
    SOLVER_API double                                 damping_decrease_factor() const;
    SOLVER_API double                                 damping_increase_factor() const;
    SOLVER_API double                                 damping_floor() const;
    SOLVER_API double                                 nielsen_damping_floor() const;
    SOLVER_API double                                 damping_ceiling() const;
    SOLVER_API double                                 levenberg_marquardt_damping_ceiling() const;
    SOLVER_API double                                 diagonal_scaling_floor() const;
    SOLVER_API double                                 roundoff_noise_factor() const;
    SOLVER_API double                                 finite_difference_step() const;
    SOLVER_API levenberg_marquardt_solver_enum        type() const;
    SOLVER_API levenberg_marquardt_linear_solver_enum linear_solver() const;

private:
    solver_options_lm();

    bool   bold_acceptance_                     = false;
    double bold_acceptance_exponent_            = 2.0;
    bool   geodesic_acceleration_               = true;
    double geodesic_acceleration_threshold_     = 0.75;
    double initial_damping_                     = 1e-4;
    double initial_rejection_multiplier_        = 2.0;
    double damping_decrease_factor_             = 9.0;
    double damping_increase_factor_             = 11.0;
    double damping_floor_                       = 1e-7;
    double nielsen_damping_floor_               = 1e-15;
    double damping_ceiling_                     = 1e12;
    double levenberg_marquardt_damping_ceiling_ = 1e7;
    double diagonal_scaling_floor_              = 1e-12;
    double roundoff_noise_factor_               = 8.0;
    double geodesic_acceleration_step_          = 0.05;
    double finite_difference_step_              = 0.00001;

    levenberg_marquardt_solver_enum        type_ = levenberg_marquardt_solver_enum::NIELSEN;
    levenberg_marquardt_linear_solver_enum linear_solver_ =
        levenberg_marquardt_linear_solver_enum::NORMAL_LDLT;
};

class SOLVER_VISIBILITY solver_options_lm_builder
{
public:
    SOLVER_API solver_options_lm_builder();

    SOLVER_API solver_options_lm_builder& with_max_iterations(int val);
    SOLVER_API solver_options_lm_builder& with_function_tolerance(double val);
    SOLVER_API solver_options_lm_builder& with_gradient_tolerance(double val);
    SOLVER_API solver_options_lm_builder& with_parameter_tolerance(double val);
    SOLVER_API solver_options_lm_builder& with_verbose(bool val = true);
    SOLVER_API solver_options_lm_builder& with_bold_acceptance(bool val = true);
    SOLVER_API solver_options_lm_builder& with_bold_acceptance_exponent(double val);
    SOLVER_API solver_options_lm_builder& with_geodesic_acceleration(bool val = true);
    SOLVER_API solver_options_lm_builder& with_geodesic_acceleration_threshold(double val);
    SOLVER_API solver_options_lm_builder& with_geodesic_acceleration_step(double val);
    SOLVER_API solver_options_lm_builder& with_initial_damping(double val);
    SOLVER_API solver_options_lm_builder& with_initial_rejection_multiplier(double val);
    SOLVER_API solver_options_lm_builder& with_damping_decrease_factor(double val);
    SOLVER_API solver_options_lm_builder& with_damping_increase_factor(double val);
    SOLVER_API solver_options_lm_builder& with_damping_floor(double val);
    SOLVER_API solver_options_lm_builder& with_nielsen_damping_floor(double val);
    SOLVER_API solver_options_lm_builder& with_damping_ceiling(double val);
    SOLVER_API solver_options_lm_builder& with_levenberg_marquardt_damping_ceiling(double val);
    SOLVER_API solver_options_lm_builder& with_diagonal_scaling_floor(double val);
    SOLVER_API solver_options_lm_builder& with_roundoff_noise_factor(double val);
    SOLVER_API solver_options_lm_builder& with_finite_difference_step(double val);
    SOLVER_API solver_options_lm_builder& with_type(levenberg_marquardt_solver_enum val);
    SOLVER_API solver_options_lm_builder& with_linear_solver(
        levenberg_marquardt_linear_solver_enum val);
    SOLVER_API solver_options_lm_builder& with_log_file(const std::string& val);
    SOLVER_API std::shared_ptr<const solver_options_lm> build() const;

private:
    std::shared_ptr<solver_options_lm> options_;
};

}  // namespace solverslib
