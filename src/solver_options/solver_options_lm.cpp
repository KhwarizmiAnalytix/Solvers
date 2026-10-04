#include "solver_options/solver_options_lm.h"

#include <cmath>
#include <stdexcept>

namespace solverslib
{
solver_options_lm::solver_options_lm() : solver_options(solver_enum::LM) {}

bool solver_options_lm::bold_acceptance() const
{
    return bold_acceptance_;
}
double solver_options_lm::bold_acceptance_exponent() const
{
    return bold_acceptance_exponent_;
}
bool solver_options_lm::geodesic_acceleration() const
{
    return geodesic_acceleration_;
}
double solver_options_lm::geodesic_acceleration_threshold() const
{
    return geodesic_acceleration_threshold_;
}
double solver_options_lm::geodesic_acceleration_step() const
{
    return geodesic_acceleration_step_;
}
double solver_options_lm::initial_damping() const
{
    return initial_damping_;
}
double solver_options_lm::initial_rejection_multiplier() const
{
    return initial_rejection_multiplier_;
}
double solver_options_lm::damping_decrease_factor() const
{
    return damping_decrease_factor_;
}
double solver_options_lm::damping_increase_factor() const
{
    return damping_increase_factor_;
}
double solver_options_lm::damping_floor() const
{
    return damping_floor_;
}
double solver_options_lm::nielsen_damping_floor() const
{
    return nielsen_damping_floor_;
}
double solver_options_lm::damping_ceiling() const
{
    return damping_ceiling_;
}
double solver_options_lm::levenberg_marquardt_damping_ceiling() const
{
    return levenberg_marquardt_damping_ceiling_;
}
double solver_options_lm::diagonal_scaling_floor() const
{
    return diagonal_scaling_floor_;
}
double solver_options_lm::roundoff_noise_factor() const
{
    return roundoff_noise_factor_;
}
double solver_options_lm::finite_difference_step() const
{
    return finite_difference_step_;
}
levenberg_marquardt_solver_enum solver_options_lm::type() const
{
    return type_;
}

solver_options_lm_builder::solver_options_lm_builder()
    : options_(std::shared_ptr<solver_options_lm>(new solver_options_lm()))
{
}

solver_options_lm_builder& solver_options_lm_builder::with_max_iterations(int val)
{
    options_->max_num_iterations_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_function_tolerance(double val)
{
    options_->function_tolerance_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_gradient_tolerance(double val)
{
    options_->gradient_tolerance_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_parameter_tolerance(double val)
{
    options_->parameter_tolerance_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_verbose(bool val)
{
    options_->verbose_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_bold_acceptance(bool val)
{
    options_->bold_acceptance_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_bold_acceptance_exponent(double val)
{
    options_->bold_acceptance_exponent_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_geodesic_acceleration(bool val)
{
    options_->geodesic_acceleration_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_geodesic_acceleration_threshold(
    double val)
{
    options_->geodesic_acceleration_threshold_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_geodesic_acceleration_step(double val)
{
    options_->geodesic_acceleration_step_ = val;
    return *this;
}

solver_options_lm_builder& solver_options_lm_builder::with_initial_damping(double val)
{
    options_->initial_damping_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_initial_rejection_multiplier(double val)
{
    options_->initial_rejection_multiplier_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_damping_decrease_factor(double val)
{
    options_->damping_decrease_factor_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_damping_increase_factor(double val)
{
    options_->damping_increase_factor_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_damping_floor(double val)
{
    options_->damping_floor_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_nielsen_damping_floor(double val)
{
    options_->nielsen_damping_floor_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_damping_ceiling(double val)
{
    options_->damping_ceiling_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_levenberg_marquardt_damping_ceiling(
    double val)
{
    options_->levenberg_marquardt_damping_ceiling_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_diagonal_scaling_floor(double val)
{
    options_->diagonal_scaling_floor_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_roundoff_noise_factor(double val)
{
    options_->roundoff_noise_factor_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_finite_difference_step(double val)
{
    options_->finite_difference_step_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_type(levenberg_marquardt_solver_enum val)
{
    options_->type_ = val;
    return *this;
}
solver_options_lm_builder& solver_options_lm_builder::with_log_file(const std::string& val)
{
    options_->log_file_ = val;
    return *this;
}

std::shared_ptr<const solver_options_lm> solver_options_lm_builder::build() const
{
    const auto positive    = [](double value) { return std::isfinite(value) && value > 0.; };
    const auto nonnegative = [](double value) { return std::isfinite(value) && value >= 0.; };
    if (options_->max_num_iterations_ < 0 || !nonnegative(options_->function_tolerance_) ||
        !nonnegative(options_->gradient_tolerance_) ||
        !nonnegative(options_->parameter_tolerance_) || !positive(options_->initial_damping_) ||
        !positive(options_->geodesic_acceleration_step_) ||
        !positive(options_->finite_difference_step_) ||
        !positive(options_->geodesic_acceleration_threshold_) ||
        !positive(options_->bold_acceptance_exponent_) ||
        !std::isfinite(options_->initial_rejection_multiplier_) ||
        options_->initial_rejection_multiplier_ <= 1. ||
        !std::isfinite(options_->damping_decrease_factor_) ||
        options_->damping_decrease_factor_ <= 1. ||
        !std::isfinite(options_->damping_increase_factor_) ||
        options_->damping_increase_factor_ <= 1. || !positive(options_->damping_floor_) ||
        !positive(options_->nielsen_damping_floor_) || !positive(options_->damping_ceiling_) ||
        !positive(options_->levenberg_marquardt_damping_ceiling_) ||
        !positive(options_->diagonal_scaling_floor_) ||
        !nonnegative(options_->roundoff_noise_factor_) ||
        options_->damping_floor_ >= options_->damping_ceiling_ ||
        options_->nielsen_damping_floor_ >= options_->damping_ceiling_ ||
        options_->damping_floor_ >= options_->levenberg_marquardt_damping_ceiling_ ||
        static_cast<int>(options_->type_) < 0 || static_cast<int>(options_->type_) > 2)
    {
        throw std::invalid_argument("Invalid Levenberg-Marquardt options");
    }
    return std::shared_ptr<const solver_options_lm>(new solver_options_lm(*options_));
}
}  // namespace solverslib
