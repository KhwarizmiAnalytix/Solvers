#include "solvers/petsc_tao_solver.h"

#include <utility>

#include "detail/support.h"
#include "solver_options/solver_options_petsc.h"

#if SOLVERS_HAS_PETSC
#include <petsctao.h>
#endif

namespace solverslib
{
namespace
{
#if SOLVERS_HAS_PETSC
const char* tao_type_string(tao_algorithm_enum type)
{
    switch (type)
    {
    case tao_algorithm_enum::POUNDERS:
        return TAOPOUNDERS;
    case tao_algorithm_enum::BRGN:
        return TAOBRGN;
    case tao_algorithm_enum::NLS:
        return TAONLS;
    case tao_algorithm_enum::NTR:
        return TAONTR;
    case tao_algorithm_enum::NTL:
        return TAONTL;
    case tao_algorithm_enum::LMVM:
        return TAOLMVM;
    case tao_algorithm_enum::BQNLS:
        return TAOBQNLS;
    case tao_algorithm_enum::BNLS:
        return TAOBNLS;
    }
    return TAOLMVM;
}

// Context handed to the TAO C trampolines via void*.
struct tao_context
{
    size_t                            n;
    size_t                            m;
    petsc_tao_solver::objective_type* objective;
    petsc_tao_solver::gradient_type*  gradient;
    petsc_tao_solver::hessian_type*   hessian;
    petsc_tao_solver::residual_type*  residuals;
    petsc_tao_solver::jacobian_type*  jacobian;
};

vector_type vec_to_eigen(Vec v, size_t n)
{
    const PetscScalar* a = nullptr;
    VecGetArrayRead(v, &a);
    vector_type x = to_vector_type(a, n);
    VecRestoreArrayRead(v, &a);
    return x;
}

PetscErrorCode obj_grad_tramp(Tao /*tao*/, Vec x, PetscReal* f, Vec g, void* ctx)
{
    auto*       c  = static_cast<tao_context*>(ctx);
    vector_type xe = vec_to_eigen(x, c->n);
    *f             = (*c->objective)(xe);
    vector_type ge = make_vector(c->n);
    (*c->gradient)(xe, ge);
    PetscScalar* ga = nullptr;
    VecGetArray(g, &ga);
    copy_into(ga, c->n, ge);
    VecRestoreArray(g, &ga);
    return 0;
}

PetscErrorCode hessian_tramp(Tao /*tao*/, Vec x, Mat H, Mat /*Hpre*/, void* ctx)
{
    auto*       c  = static_cast<tao_context*>(ctx);
    vector_type xe = vec_to_eigen(x, c->n);
    matrix_type he = make_matrix(c->n, c->n);
    (*c->hessian)(xe, he);
    for (size_t i = 0; i < c->n; ++i)
    {
        for (size_t j = 0; j < c->n; ++j)
        {
            MatSetValue(
                H, static_cast<PetscInt>(i), static_cast<PetscInt>(j), he(i, j), INSERT_VALUES);
        }
    }
    MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);
    return 0;
}

PetscErrorCode residual_tramp(Tao /*tao*/, Vec x, Vec r, void* ctx)
{
    auto*       c  = static_cast<tao_context*>(ctx);
    vector_type xe = vec_to_eigen(x, c->n);
    vector_type re = make_vector(c->m);
    (*c->residuals)(xe, re);
    PetscScalar* ra = nullptr;
    VecGetArray(r, &ra);
    copy_into(ra, c->m, re);
    VecRestoreArray(r, &ra);
    return 0;
}

PetscErrorCode residual_jac_tramp(Tao /*tao*/, Vec x, Mat J, Mat /*Jpre*/, void* ctx)
{
    auto*       c  = static_cast<tao_context*>(ctx);
    vector_type xe = vec_to_eigen(x, c->n);
    matrix_type je = make_matrix(c->m, c->n);
    (*c->jacobian)(xe, je);
    for (size_t i = 0; i < c->m; ++i)
    {
        for (size_t j = 0; j < c->n; ++j)
        {
            MatSetValue(
                J, static_cast<PetscInt>(i), static_cast<PetscInt>(j), je(i, j), INSERT_VALUES);
        }
    }
    MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
    return 0;
}
#endif  // SOLVERS_HAS_PETSC
}  // namespace

petsc_tao_solver::petsc_tao_solver(size_t num_parameters,
    objective_type                        objective,
    gradient_type                         gradient,
    hessian_type                          hessian,
    std::vector<double>                   lower_bounds,
    std::vector<double>                   upper_bounds)
    : is_least_squares_(false), num_parameters_(num_parameters), num_residuals_(0),
      objective_(std::move(objective)), gradient_(std::move(gradient)),
      hessian_(std::move(hessian)), lower_bounds_(std::move(lower_bounds)),
      upper_bounds_(std::move(upper_bounds))
{
}

petsc_tao_solver::petsc_tao_solver(size_t num_parameters,
    size_t                                num_residuals,
    residual_type                         residuals,
    jacobian_type                         jacobian,
    std::vector<double>                   lower_bounds,
    std::vector<double>                   upper_bounds)
    : is_least_squares_(true), num_parameters_(num_parameters), num_residuals_(num_residuals),
      residuals_(std::move(residuals)), jacobian_(std::move(jacobian)),
      lower_bounds_(std::move(lower_bounds)), upper_bounds_(std::move(upper_bounds))
{
}

bool petsc_tao_solver::is_supported()
{
#if SOLVERS_HAS_PETSC
    return true;
#else
    return false;
#endif
}

bool petsc_tao_solver::solve(SOLVERS_UNUSED std::vector<double>& parameters,
    SOLVERS_UNUSED const solver_options_petsc&                   options)
{
#if SOLVERS_HAS_PETSC
    if (!PetscInitializeCalled)
    {
        PetscInitializeNoArguments();
    }

    const auto  n = static_cast<PetscInt>(num_parameters_);
    tao_context ctx{num_parameters_,
        num_residuals_,
        &objective_,
        &gradient_,
        &hessian_,
        &residuals_,
        &jacobian_};

    Vec x;
    VecCreateSeq(PETSC_COMM_SELF, n, &x);
    {
        PetscScalar* xa = nullptr;
        VecGetArray(x, &xa);
        for (size_t i = 0; i < num_parameters_; ++i)
        {
            xa[i] = parameters[i];
        }
        VecRestoreArray(x, &xa);
    }

    Tao tao;
    TaoCreate(PETSC_COMM_SELF, &tao);
    TaoSetType(tao, tao_type_string(options.tao_type()));
    TaoSetSolution(tao, x);

    Vec res = nullptr;
    Mat H   = nullptr;
    Mat J   = nullptr;
    if (is_least_squares_)
    {
        VecCreateSeq(PETSC_COMM_SELF, static_cast<PetscInt>(num_residuals_), &res);
        TaoSetResidualRoutine(tao, res, residual_tramp, &ctx);
        if (jacobian_)
        {
            MatCreateSeqDense(
                PETSC_COMM_SELF, static_cast<PetscInt>(num_residuals_), n, nullptr, &J);
            TaoSetJacobianResidualRoutine(tao, J, J, residual_jac_tramp, &ctx);
        }
    }
    else
    {
        TaoSetObjectiveAndGradient(tao, nullptr, obj_grad_tramp, &ctx);
        if (hessian_ && !options.matrix_free())
        {
            MatCreateSeqDense(PETSC_COMM_SELF, n, n, nullptr, &H);
            TaoSetHessian(tao, H, H, hessian_tramp, &ctx);
        }
    }

    if (!lower_bounds_.empty() && !upper_bounds_.empty())
    {
        Vec xl;
        Vec xu;
        VecDuplicate(x, &xl);
        VecDuplicate(x, &xu);
        PetscScalar* la = nullptr;
        PetscScalar* ua = nullptr;
        VecGetArray(xl, &la);
        VecGetArray(xu, &ua);
        for (size_t i = 0; i < num_parameters_; ++i)
        {
            la[i] = lower_bounds_[i];
            ua[i] = upper_bounds_[i];
        }
        VecRestoreArray(xl, &la);
        VecRestoreArray(xu, &ua);
        TaoSetVariableBounds(tao, xl, xu);
        VecDestroy(&xl);
        VecDestroy(&xu);
    }

    TaoSetTolerances(tao, options.gatol(), options.grtol(), 0.0);
    TaoSetMaximumIterations(tao, options.max_num_iterations());
    TaoSetFromOptions(tao);

    TaoSolve(tao);

    TaoConvergedReason reason;
    TaoGetConvergedReason(tao, &reason);

    {
        const PetscScalar* xa = nullptr;
        VecGetArrayRead(x, &xa);
        for (size_t i = 0; i < num_parameters_; ++i)
        {
            parameters[i] = PetscRealPart(xa[i]);
        }
        VecRestoreArrayRead(x, &xa);
    }

    TaoDestroy(&tao);
    VecDestroy(&x);
    if (res != nullptr)
    {
        VecDestroy(&res);
    }
    if (H != nullptr)
    {
        MatDestroy(&H);
    }
    if (J != nullptr)
    {
        MatDestroy(&J);
    }

    return reason > 0;  // positive TaoConvergedReason codes indicate convergence
#else
    return false;  // dispatcher gates on is_supported() before ever calling this
#endif
}
}  // namespace solverslib
