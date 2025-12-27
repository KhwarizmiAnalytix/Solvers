#include "optimization_algorithm/solver_output.h"

#include "util/logger.h"

namespace quarisma
{
namespace
{
template <typename T>
inline double l2_norm(T const& h)
{
    return sqrt(accumulate(sqr(h)));
}
template <typename T>
inline double l_max_norm(T const& h)
{
    return sqrt(max(sqr(h)));
}
}  // namespace

solver_output::solver_output(size_type m)
    : x2_(0), errors_(m, 0), iterations_(0), status_(solver_convergence_enum::NOT_CONVERGED)
{
}

void solver_output::update(
    bool                       x2_converged,
    bool                       parameters_converged,
    bool                       gradient_converged,
    size_type                  iteration,
    const vector<scalar_type>& y_p)
{
    iterations_ = iteration;
    if (x2_converged)
    {
        status_ = solver_convergence_enum::X2_CONVERGED;
    }
    else if (parameters_converged)
    {
        status_ = solver_convergence_enum::PARAMETERS_CONVERGED;
    }
    else if (gradient_converged)
    {
        status_ = solver_convergence_enum::GRADIENT_CONVERGED;
    }
    else
    {
        status_ = solver_convergence_enum::NOT_CONVERGED;
    }

    errors_.assign(y_p.begin(), y_p.end());
    x2_ = l2_norm(y_p);
}

void solver_output::print() const
{
    QUARISMA_LOGF(INFO, "============ final results ============");
    switch (status_)
    {
    case quarisma::solver_convergence_enum::GRADIENT_CONVERGED:
        QUARISMA_LOGF(INFO, "convergence: GRADIENT_CONVERGED");
        break;
    case quarisma::solver_convergence_enum::PARAMETERS_CONVERGED:
        QUARISMA_LOGF(INFO, "convergence: PARAMETERS_CONVERGED");
        break;
    case quarisma::solver_convergence_enum::X2_CONVERGED:
        QUARISMA_LOGF(INFO, "convergence: X2_CONVERGED");
        break;
    case quarisma::solver_convergence_enum::NOT_CONVERGED:
        QUARISMA_LOGF(INFO, "convergence: NOT_CONVERGED");
        break;
    }
    QUARISMA_LOGF(INFO, "number of iterations %d", static_cast<int>(iterations_));

    QUARISMA_LOGF(INFO, "difference = %.2e", x2_);
}

}  // namespace quarisma