#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

#include "include/detail/support.h"
#include "include/detail/support.h"
#include "solver_options/solver_options.h"

namespace solverslib
{
enum class levenberg_marquardt_solver_enum : int
{
    LEVENBERG = 0,
    QUADRATIC = 1,
    NIELSEN   = 2
};

class MATH_VISIBILITY solver_options_lm : public solver_options
{
    friend class solver_options_lm_builder;

public:
    solver_options_lm(
        int    max_num_iterations,
        double function_tolerance  = std::numeric_limits<double>::epsilon(),
        double gradient_tolerance  = 0.0,
        double parameter_tolerance = std::numeric_limits<double>::epsilon(),
        bool   verbose             = false)
        : solver_options(
              solver_enum::LM,
              max_num_iterations,
              function_tolerance,
              gradient_tolerance,
              parameter_tolerance,
              verbose)
    {
    }

    // Getters and Setters
    bool accept_uphill_step() const { return accept_uphill_step_; }
    void set_accept_uphill_step(bool accept) { accept_uphill_step_ = accept; }

    bool use_geodesic() const { return use_geodesic_; }
    void set_use_geodesic(bool use) { use_geodesic_ = use; }

    double alpha() const { return alpha_; }
    void   set_alpha(double alpha) { alpha_ = alpha; }

    double lambda() const { return lambda_; }
    void   set_lambda(double lambda) { lambda_ = lambda; }

    double nu() const { return nu_; }
    void   set_nu(double nu) { nu_ = nu; }

    double lambda_down_fac() const { return lambda_down_fac_; }
    void   set_lambda_down_fac(double fac) { lambda_down_fac_ = fac; }

    double lambda_up_fac() const { return lambda_up_fac_; }
    void   set_lambda_up_fac(double fac) { lambda_up_fac_ = fac; }

    double epsilon() const { return epsilon_; }
    void   set_epsilon(double epsilon) { epsilon_ = epsilon; }

    double bump() const { return bump_; }
    void   set_bump(double bump) { bump_ = bump; }

    levenberg_marquardt_solver_enum type() const { return type_; }
    void                            set_type(levenberg_marquardt_solver_enum type) { type_ = type; }

    void set_log_file(const std::string& log_file) { log_file_ = log_file; }

private:
    solver_options_lm() : solver_options(solverslib::solver_enum::LM) {};

    void initialize() const {};



    bool   accept_uphill_step_ = false;
    bool   use_geodesic_       = true;
    double alpha_              = 0.75;
    double lambda_             = 1e-4;
    double nu_                 = 2.0;
    double lambda_down_fac_    = 9.0;
    double lambda_up_fac_      = 11.0;
    double epsilon_            = 0.05;
    double bump_               = 0.00001;

    levenberg_marquardt_solver_enum type_ = levenberg_marquardt_solver_enum::NIELSEN;
};

/**
 * @brief Builder for solver_options_lm configuration
 *
 * Usage:
 *   auto options = solver_options_lm_builder()
 *       .with_max_iterations(100).with_function_tolerance(1e-6)
 *       .with_lambda(1e-3).with_type(levenberg_marquardt_solver_enum::NIELSEN).build();
 */
class MATH_VISIBILITY solver_options_lm_builder
{
public:
    solver_options_lm_builder() : options_(std::shared_ptr<solver_options_lm>(new solver_options_lm()))
    {
    }

    // Base solver options
    solverslib::solver_options_lm_builder& with_max_iterations(int val)
    {
        options_->set_max_num_iterations(val);
        return *this;
    }

    solverslib::solver_options_lm_builder& with_function_tolerance(double val)
    {
        options_->set_function_tolerance(val);
        return *this;
    }

    solverslib::solver_options_lm_builder& with_gradient_tolerance(double val)
    {
        options_->set_gradient_tolerance(val);
        return *this;
    }

    solverslib::solver_options_lm_builder& with_parameter_tolerance(double val)
    {
        options_->set_parameter_tolerance(val);
        return *this;
    }

    solverslib::solver_options_lm_builder& with_verbose(bool val = true)
    {
        options_->set_debug(val);
        return *this;
    }

    // LM-specific options
    solverslib::solver_options_lm_builder& with_accept_uphill_step(bool val = true)
    {
        options_->set_accept_uphill_step(val);
        return *this;
    }

    solverslib::solver_options_lm_builder& with_use_geodesic(bool val = true)
    {
        options_->set_use_geodesic(val);
        return *this;
    }

    solverslib::solver_options_lm_builder& with_alpha(double val)
    {
        options_->set_alpha(val);
        return *this;
    }

    solverslib::solver_options_lm_builder& with_lambda(double val)
    {
        options_->set_lambda(val);
        return *this;
    }

    solverslib::solver_options_lm_builder& with_nu(double val)
    {
        options_->set_nu(val);
        return *this;
    }

    solverslib::solver_options_lm_builder& with_lambda_down_fac(double val)
    {
        options_->set_lambda_down_fac(val);
        return *this;
    }

    solverslib::solver_options_lm_builder& with_lambda_up_fac(double val)
    {
        options_->set_lambda_up_fac(val);
        return *this;
    }

    solverslib::solver_options_lm_builder& with_epsilon(double val)
    {
        options_->set_epsilon(val);
        return *this;
    }

    solverslib::solver_options_lm_builder& with_bump(double val)
    {
        options_->set_bump(val);
        return *this;
    }

    solverslib::solver_options_lm_builder& with_type(levenberg_marquardt_solver_enum val)
    {
        options_->set_type(val);
        return *this;
    }

    solverslib::solver_options_lm_builder& with_log_file(const std::string& val)
    {
        options_->set_log_file(val);
        return *this;
    }

    solverslib::solver_options_lm_builder& with_aad_jacobian(bool val = true)
    {
        options_->aad_jacobian_ = val;
        return *this;
    }

    // Build the final options
    std::shared_ptr<const solver_options_lm> build() const { return options_; }

private:
    std::shared_ptr<solver_options_lm> options_;
};

}  // namespace solverslib
