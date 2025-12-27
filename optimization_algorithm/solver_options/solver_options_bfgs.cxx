#include "optimization_algorithm/solver_options/solver_options_bfgs.h"

#include "common/serialization_macros.h"
#include "util/exception.h"

namespace quarisma
{
void solver_options_bfgs::validate() const
{
    QUARISMA_CHECK(
        linesearch_tolerance_ > 0 && linesearch_tolerance_ < 0.5,
        "'linesearch_tolerance_' must satisfy 0 < linesearch_tolerance_ < 0.5");

    QUARISMA_CHECK(
        linesearch_wolfe_ > linesearch_tolerance_ && linesearch_wolfe_ < 1.,
        "'linesearch_wolfe_' must satisfy linesearch_tolerance_ < linesearch_wolfe_ < 1");
}

//-----------------------------------------------------------------------------
QUARISMA_SERIALIZATION_METHODES(solver_options_bfgs);
QUARISMA_REGISTER_DERIVED_SERIALIZATION(solver_options_bfgs);
}  // namespace quarisma
