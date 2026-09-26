#include "solvers/ipopt_solver.h"

#include <utility>

#include "detail/support.h"
#include "solver_options/solver_options_ipopt.h"

#if SOLVERS_HAS_IPOPT
#include <IpIpoptApplication.hpp>
#include <IpTNLP.hpp>
#endif

namespace solverslib
{
ipopt_solver::ipopt_solver(size_t num_parameters,
    objective_type                 objective,
    gradient_type                  gradient,
    hessian_type                   hessian,
    std::vector<double>            lower_bounds,
    std::vector<double>            upper_bounds)
    : num_parameters_(num_parameters), objective_(std::move(objective)),
      gradient_(std::move(gradient)), hessian_(std::move(hessian)),
      lower_bounds_(std::move(lower_bounds)), upper_bounds_(std::move(upper_bounds))
{
}

bool ipopt_solver::is_supported()
{
#if SOLVERS_HAS_IPOPT
    return true;
#else
    return false;
#endif
}

#if SOLVERS_HAS_IPOPT
namespace
{
// Minimal bound-constrained TNLP (no general nonlinear constraints: m = 0).
// The redesign's constraints struct only carries counts, not callbacks, so the
// adapter models box constraints exactly and leaves general constraints for a
// later problem-struct extension.
class solvers_tnlp : public Ipopt::TNLP
{
public:
    solvers_tnlp(size_t n,
        ipopt_solver::objective_type&   objective,
        ipopt_solver::gradient_type&    gradient,
        ipopt_solver::hessian_type&     hessian,
        std::vector<double>&            lower,
        std::vector<double>&            upper,
        std::vector<double>&            x0,
        bool                            use_exact_hessian)
        : n_(static_cast<Ipopt::Index>(n)), objective_(objective), gradient_(gradient),
          hessian_(hessian), lower_(lower), upper_(upper), x_(x0),
          use_exact_hessian_(use_exact_hessian)
    {
    }

    bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m, Ipopt::Index& nnz_jac_g,
        Ipopt::Index& nnz_h_lag, IndexStyleEnum& index_style) override
    {
        n           = n_;
        m           = 0;
        nnz_jac_g   = 0;
        nnz_h_lag   = use_exact_hessian_ ? (n_ * (n_ + 1)) / 2 : 0;
        index_style = TNLP::C_STYLE;
        return true;
    }

    bool get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
        Ipopt::Index /*m*/, Ipopt::Number* /*g_l*/, Ipopt::Number* /*g_u*/) override
    {
        const double inf = 2e19;  // Ipopt treats |value| >= nlp_upper_bound_inf as infinity
        for (Ipopt::Index i = 0; i < n; ++i)
        {
            x_l[i] = lower_.empty() ? -inf : lower_[static_cast<size_t>(i)];
            x_u[i] = upper_.empty() ? inf : upper_[static_cast<size_t>(i)];
        }
        return true;
    }

    bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x, bool /*init_z*/,
        Ipopt::Number* /*z_L*/, Ipopt::Number* /*z_U*/, Ipopt::Index /*m*/, bool /*init_lambda*/,
        Ipopt::Number* /*lambda*/) override
    {
        if (init_x)
        {
            for (Ipopt::Index i = 0; i < n; ++i)
            {
                x[i] = x_[static_cast<size_t>(i)];
            }
        }
        return true;
    }

    bool eval_f(Ipopt::Index n, const Ipopt::Number* x, bool /*new_x*/,
        Ipopt::Number& obj_value) override
    {
        obj_value = objective_(to_vector_type(x, static_cast<size_t>(n)));
        return std::isfinite(obj_value);
    }

    bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool /*new_x*/,
        Ipopt::Number* grad_f) override
    {
        vector_type g = make_vector(static_cast<size_t>(n));
        gradient_(to_vector_type(x, static_cast<size_t>(n)), g);
        copy_into(grad_f, static_cast<size_t>(n), g);
        return true;
    }

    bool eval_g(Ipopt::Index /*n*/, const Ipopt::Number* /*x*/, bool /*new_x*/, Ipopt::Index /*m*/,
        Ipopt::Number* /*g*/) override
    {
        return true;  // no constraints
    }

    bool eval_jac_g(Ipopt::Index /*n*/, const Ipopt::Number* /*x*/, bool /*new_x*/,
        Ipopt::Index /*m*/, Ipopt::Index /*nele_jac*/, Ipopt::Index* /*iRow*/,
        Ipopt::Index* /*jCol*/, Ipopt::Number* /*values*/) override
    {
        return true;  // no constraints
    }

    bool eval_h(Ipopt::Index n, const Ipopt::Number* x, bool /*new_x*/, Ipopt::Number obj_factor,
        Ipopt::Index /*m*/, const Ipopt::Number* /*lambda*/, bool /*new_lambda*/,
        Ipopt::Index /*nele_hess*/, Ipopt::Index* iRow, Ipopt::Index* jCol,
        Ipopt::Number* values) override
    {
        if (!use_exact_hessian_)
        {
            return false;
        }
        if (values == nullptr)
        {
            // Lower-triangular sparsity pattern of a dense Hessian.
            Ipopt::Index idx = 0;
            for (Ipopt::Index r = 0; r < n; ++r)
            {
                for (Ipopt::Index c = 0; c <= r; ++c)
                {
                    iRow[idx] = r;
                    jCol[idx] = c;
                    ++idx;
                }
            }
            return true;
        }
        matrix_type h = make_matrix(static_cast<size_t>(n), static_cast<size_t>(n));
        hessian_(to_vector_type(x, static_cast<size_t>(n)), h);
        Ipopt::Index idx = 0;
        for (Ipopt::Index r = 0; r < n; ++r)
        {
            for (Ipopt::Index c = 0; c <= r; ++c)
            {
                values[idx++] = obj_factor * h(r, c);
            }
        }
        return true;
    }

    void finalize_solution(Ipopt::SolverReturn /*status*/, Ipopt::Index n, const Ipopt::Number* x,
        const Ipopt::Number* /*z_L*/, const Ipopt::Number* /*z_U*/, Ipopt::Index /*m*/,
        const Ipopt::Number* /*g*/, const Ipopt::Number* /*lambda*/, Ipopt::Number /*obj_value*/,
        const Ipopt::IpoptData* /*ip_data*/,
        Ipopt::IpoptCalculatedQuantities* /*ip_cq*/) override
    {
        for (Ipopt::Index i = 0; i < n; ++i)
        {
            x_[static_cast<size_t>(i)] = x[i];
        }
    }

private:
    Ipopt::Index                  n_;
    ipopt_solver::objective_type& objective_;
    ipopt_solver::gradient_type&  gradient_;
    ipopt_solver::hessian_type&   hessian_;
    std::vector<double>&          lower_;
    std::vector<double>&          upper_;
    std::vector<double>&          x_;
    bool                          use_exact_hessian_;
};
}  // namespace
#endif  // SOLVERS_HAS_IPOPT

bool ipopt_solver::solve(SOLVERS_UNUSED std::vector<double>& parameters,
    SOLVERS_UNUSED const solver_options_ipopt&                options)
{
#if SOLVERS_HAS_IPOPT
    const bool use_exact_hessian =
        options.hessian_approximation() == ipopt_hessian_approximation_enum::EXACT &&
        static_cast<bool>(hessian_);

    Ipopt::SmartPtr<Ipopt::TNLP> nlp = new solvers_tnlp(num_parameters_, objective_, gradient_,
        hessian_, lower_bounds_, upper_bounds_, parameters, use_exact_hessian);

    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();
    app->Options()->SetNumericValue("tol", options.tol());
    app->Options()->SetNumericValue("acceptable_tol", options.acceptable_tol());
    app->Options()->SetIntegerValue("max_iter", options.max_num_iterations());
    app->Options()->SetNumericValue("max_wall_time", options.max_wall_time_seconds());
    app->Options()->SetStringValue("hessian_approximation",
        use_exact_hessian ? "exact" : "limited-memory");
    if (!options.linear_solver().empty())
    {
        app->Options()->SetStringValue("linear_solver", options.linear_solver());
    }
    app->Options()->SetIntegerValue("print_level", options.verbose() ? 5 : 0);

    if (app->Initialize() != Ipopt::Solve_Succeeded)
    {
        return false;
    }
    const Ipopt::ApplicationReturnStatus status = app->OptimizeTNLP(nlp);
    return status == Ipopt::Solve_Succeeded || status == Ipopt::Solved_To_Acceptable_Level;
#else
    return false;  // dispatcher gates on is_supported() before ever calling this
#endif
}
}  // namespace solverslib
