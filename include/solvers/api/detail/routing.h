#ifndef SOLVERS_ROUTING_H_
#define SOLVERS_ROUTING_H_

#include <cstdint>
#include <optional>
#include <string>

#include "solvers/api/options.h"
#include "solvers/api/problem.h"
#include "solvers/api/status.h"

// Internal: the single table that drives backend/algorithm selection,
// capability validation and the "why was this rejected" messages. Not part of
// the stable API; exposed only so tests can inject backend availability.
//
// Design note: PyTorch's dispatcher resolves a call by looking a key set up in
// a table rather than branching through code; Ceres validates Solver::Options
// in one compatibility function. This is the same idea at a much smaller
// scale: a compile-time array, no runtime registry.
namespace solverslib::api::detail
{
enum class capability : std::uint16_t
{
    none                  = 0,
    bounds                = 1u << 0,
    nonlinear_constraints = 1u << 1,
    hessian_vector        = 1u << 2,
    derivative_free       = 1u << 3,  // runs with no Jacobian/gradient source
    native_ad             = 1u << 4   // consumes provider_factory directly
};

constexpr capability operator|(capability a, capability b)
{
    return static_cast<capability>(static_cast<std::uint16_t>(a) | static_cast<std::uint16_t>(b));
}
constexpr capability operator&(capability a, capability b)
{
    return static_cast<capability>(static_cast<std::uint16_t>(a) & static_cast<std::uint16_t>(b));
}
constexpr bool has_all(capability set, capability required)
{
    return (set & required) == required;
}

enum class problem_kind : std::uint8_t
{
    least_squares,
    objective
};

// Which build-time feature a route needs.
enum class requirement : std::uint8_t
{
    always,
    ceres,
    petsc_tao,
    ipopt
};

// Test seam: which optional backends count as built.
struct backend_availability
{
    bool ceres     = false;
    bool petsc_tao = false;
    bool ipopt     = false;

    bool operator()(requirement need) const noexcept;

    // What this binary was actually compiled with.
    static backend_availability current();
    // Everything built; used by tests to exercise every route in any build.
    static backend_availability all();
    // Nothing optional built.
    static backend_availability none();
};

struct route
{
    api::backend   backend;
    api::algorithm algorithm;
    problem_kind   kind;
    capability     supports;
    requirement    needs;
    int            priority;  // higher wins among eligible routes
    bool           large_scale;
    // Chosen automatically only when the problem has no derivative source
    // (the algorithm ignores derivatives, e.g. POUNDERS).
    bool derivative_free_only;
    // Chosen automatically only when the problem has some derivative source.
    bool needs_derivative_source;
};

struct route_decision
{
    const route*  chosen  = nullptr;  // null when no route can serve the request
    solver_status failure = solver_status::unsupported_capability;
    std::string   message;
};

capability required_capabilities(const problem_traits& traits);

// Resolve a request to one table row. A row that is not built is chosen only
// when no built row can serve the request, so automatic mode never picks an
// unbuilt backend while a built one would do; the executor then reports
// `backend_unavailable`.
route_decision select_route(const problem_traits& traits,
    const solve_options&                          options,
    const backend_availability&                   availability = backend_availability::current());

struct validation_error
{
    solver_status status;
    std::string   message;
};

// Option sanity independent of any backend (budgets, tolerances, finiteness).
std::optional<validation_error> validate_options(
    const solve_options& options, const vector_type& x);

// One bounds check shared by both problem kinds. Returns a message on error.
std::optional<std::string> validate_bounds(const api::bounds& bounds, std::size_t n);

const char* to_string(capability single_capability);
}  // namespace solverslib::api::detail

#endif  // SOLVERS_ROUTING_H_
