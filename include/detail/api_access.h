#ifndef SOLVERS_API_ACCESS_H_
#define SOLVERS_API_ACCESS_H_

#include "detail/eigen_support.h"

namespace solverslib
{
class solver_options;

namespace api
{
struct least_squares_problem;
struct optimization_problem;
struct solver_result;

solver_result solve(const least_squares_problem& problem,
    const vector_type&                           initial_guess,
    const solver_options&                        options);

solver_result solve(const optimization_problem& problem,
    const vector_type&                          initial_guess,
    const solver_options&                       options);
}  // namespace api
}  // namespace solverslib

// Grants api::solve access to a solver's private constructors and methods.
#define SOLVERS_FRIEND_API_SOLVE                                                                   \
    friend ::solverslib::api::solver_result(::solverslib::api::solve)(                             \
        const ::solverslib::api::least_squares_problem&,                                           \
        const ::solverslib::vector_type&,                                                          \
        const ::solverslib::solver_options&);                                                      \
    friend ::solverslib::api::solver_result(::solverslib::api::solve)(                             \
        const ::solverslib::api::optimization_problem&,                                            \
        const ::solverslib::vector_type&,                                                          \
        const ::solverslib::solver_options&);

#endif  // SOLVERS_API_ACCESS_H_
