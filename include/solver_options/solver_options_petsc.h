#pragma once

#include <limits>
#include <memory>
#include <string>

#include "include/detail/support.h"
#include "solver_options/solver_options.h"

namespace solverslib
{
// PETSc/TAO exposes a family of algorithms selected by a short type string.
// POUNDERS (derivative-free least squares) and the Newton/quasi-Newton methods
// are all TAO algorithms, so a single adapter/option class covers them; the
// enum maps to TAO's "-tao_type" values inside the adapter.
enum class tao_algorithm_enum : int
{
    POUNDERS = 0,  // "pounders": model-based derivative-free least squares (BRGN family)
    BRGN     = 1,  // "brgn": bounded regularized Gauss-Newton least squares
    NLS      = 2,  // "nls": Newton line search (general objective)
    NTR      = 3,  // "ntr": Newton trust region
    NTL      = 4,  // "ntl": Newton trust-region line search
    LMVM     = 5,  // "lmvm": limited-memory variable metric (matrix-free quasi-Newton)
    BQNLS    = 6,  // "bqnls": bound-constrained quasi-Newton line search
    BNLS     = 7   // "bnls": bound-constrained Newton line search
};

class SOLVER_VISIBILITY solver_options_petsc : public solver_options
{
    friend class solver_options_petsc_builder;

public:
    solver_options_petsc(tao_algorithm_enum tao_type,
        int                                 max_num_iterations,
        double                              function_tolerance  = 1e-8,
        double                              gradient_tolerance  = 1e-8,
        double                              parameter_tolerance = 1e-8,
        bool                                verbose             = false)
        : solver_options(solver_enum::PETSC_TAO,
              max_num_iterations,
              function_tolerance,
              gradient_tolerance,
              parameter_tolerance,
              verbose),
          tao_type_(tao_type)
    {
    }

    tao_algorithm_enum tao_type() const { return tao_type_; }
    void               set_tao_type(tao_algorithm_enum value) { tao_type_ = value; }

    // Absolute/relative gradient tolerances forwarded to TaoSetTolerances.
    double gatol() const { return gatol_; }
    void   set_gatol(double value) { gatol_ = value; }

    double grtol() const { return grtol_; }
    void   set_grtol(double value) { grtol_ = value; }

    // When true, the adapter drives TAO with a matrix-free Hessian-vector
    // product instead of an assembled Hessian (large-scale / Newton-Krylov).
    bool matrix_free() const { return matrix_free_; }
    void set_matrix_free(bool value) { matrix_free_ = value; }

    void set_log_file(const std::string& log_file) { log_file_ = log_file; }

private:
    solver_options_petsc() : solver_options(solverslib::solver_enum::PETSC_TAO) {}
    void initialize() const {}

    tao_algorithm_enum tao_type_    = tao_algorithm_enum::LMVM;
    double             gatol_       = 1e-8;
    double             grtol_       = 1e-8;
    bool               matrix_free_ = false;
};

class SOLVER_VISIBILITY solver_options_petsc_builder
{
public:
    solver_options_petsc_builder()
        : options_(std::shared_ptr<solver_options_petsc>(new solver_options_petsc()))
    {
    }

    solver_options_petsc_builder& with_tao_type(tao_algorithm_enum val)
    {
        options_->set_tao_type(val);
        return *this;
    }
    solver_options_petsc_builder& with_max_iterations(int val)
    {
        options_->set_max_num_iterations(val);
        return *this;
    }
    solver_options_petsc_builder& with_function_tolerance(double val)
    {
        options_->set_function_tolerance(val);
        return *this;
    }
    solver_options_petsc_builder& with_gradient_tolerance(double val)
    {
        options_->set_gradient_tolerance(val);
        return *this;
    }
    solver_options_petsc_builder& with_parameter_tolerance(double val)
    {
        options_->set_parameter_tolerance(val);
        return *this;
    }
    solver_options_petsc_builder& with_verbose(bool val = true)
    {
        options_->set_debug(val);
        return *this;
    }
    solver_options_petsc_builder& with_gatol(double val)
    {
        options_->set_gatol(val);
        return *this;
    }
    solver_options_petsc_builder& with_grtol(double val)
    {
        options_->set_grtol(val);
        return *this;
    }
    solver_options_petsc_builder& with_matrix_free(bool val = true)
    {
        options_->set_matrix_free(val);
        return *this;
    }
    solver_options_petsc_builder& with_log_file(const std::string& val)
    {
        options_->set_log_file(val);
        return *this;
    }

    std::shared_ptr<const solver_options_petsc> build() const { return options_; }

private:
    std::shared_ptr<solver_options_petsc> options_;
};
}  // namespace solverslib
