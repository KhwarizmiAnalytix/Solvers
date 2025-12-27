// NOLINT(clang-tidy): Skip clang-tidy for this file due to memory issues with Ceres templates
#include "optimization_algorithm/solver_options/solver_options_ceres.h"

#include "common/serialization_macros.h"

namespace quarisma
{
//-----------------------------------------------------------------------------
QUARISMA_SERIALIZATION_METHODES(solver_options_ceres);
QUARISMA_REGISTER_DERIVED_SERIALIZATION(solver_options_ceres);
}  // namespace quarisma
