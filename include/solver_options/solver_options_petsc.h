#pragma once

#include <limits>
#include <memory>
#include <string>

#include "detail/support.h"
#include "solver_options/solver_options.h"

namespace solverslib
{
enum class tao_algorithm_enum : int
{
    POUNDERS = 0,
    BRGN     = 1,
    NLS      = 2,
    NTR      = 3,
    NTL      = 4,
    LMVM     = 5,
    BQNLS    = 6,
    BNLS     = 7
};

class SOLVER_VISIBILITY solver_options_petsc : public solver_options
{
    friend class solver_options_petsc_builder;

public:
    SOLVER_API tao_algorithm_enum tao_type() const;
    SOLVER_API double             gatol() const;
    SOLVER_API double             grtol() const;
    SOLVER_API bool               matrix_free() const;

private:
    solver_options_petsc();

    tao_algorithm_enum tao_type_    = tao_algorithm_enum::LMVM;
    double             gatol_       = 1e-8;
    double             grtol_       = 1e-8;
    bool               matrix_free_ = false;
};

class SOLVER_VISIBILITY solver_options_petsc_builder
{
public:
    SOLVER_API solver_options_petsc_builder();

    SOLVER_API solver_options_petsc_builder& with_tao_type(tao_algorithm_enum val);
    SOLVER_API solver_options_petsc_builder& with_max_iterations(int val);
    SOLVER_API solver_options_petsc_builder& with_function_tolerance(double val);
    SOLVER_API solver_options_petsc_builder& with_gradient_tolerance(double val);
    SOLVER_API solver_options_petsc_builder& with_parameter_tolerance(double val);
    SOLVER_API solver_options_petsc_builder& with_verbose(bool val = true);
    SOLVER_API solver_options_petsc_builder& with_gatol(double val);
    SOLVER_API solver_options_petsc_builder& with_grtol(double val);
    SOLVER_API solver_options_petsc_builder& with_matrix_free(bool val = true);
    SOLVER_API solver_options_petsc_builder& with_log_file(const std::string& val);
    SOLVER_API solver_options_petsc_builder& with_aad_jacobian(bool val = true);

    SOLVER_API std::shared_ptr<const solver_options_petsc> build() const;

private:
    std::shared_ptr<solver_options_petsc> options_;
};
}  // namespace solverslib
