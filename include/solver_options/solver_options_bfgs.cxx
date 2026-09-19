#include "solver_options/solver_options_bfgs.h"

#include "detail/support.h"

namespace quarisma
{
void solver_options_bfgs::validate() const
{
    SOLVERS_CHECK(
        linesearch_tolerance_ > 0 && linesearch_tolerance_ < 0.5,
        "'linesearch_tolerance_' must satisfy 0 < linesearch_tolerance_ < 0.5");

    SOLVERS_CHECK(
        linesearch_wolfe_ > linesearch_tolerance_ && linesearch_wolfe_ < 1.,
        "'linesearch_wolfe_' must satisfy linesearch_tolerance_ < linesearch_wolfe_ < 1");
}

//-----------------------------------------------------------------------------


}  // namespace quarisma
