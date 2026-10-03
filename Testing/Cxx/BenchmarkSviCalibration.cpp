// Cross-backend calibration benchmark for the raw-SVI model in:
// T. Ferhati, "Robust Calibration For SVI Model Arbitrage Free" (2020),
// DOI 10.2139/ssrn.3543766.
//
// The fixture is the 1.01-year EURO STOXX 50 slice from Tables 3.1 and 3.2.
// The paper does not print the forward, so it is recovered from all quoted
// calls and puts by least-squares put-call parity. Every backend receives the
// paper's initial guess and parameter bounds through the same smooth bounded
// parameterization. The paper's nonlinear g(k) constraint is reported as a
// post-calibration diagnostic because the unified solver API currently exposes
// box constraints only.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "solvers/api/solve.h"

namespace
{
using clock_type = std::chrono::steady_clock;
using namespace solverslib;
using namespace solverslib::api;

constexpr std::size_t kNumParameters = 5;
constexpr double      kVarianceScale = 10000.0;
constexpr std::size_t kWarmupRounds  = 3;
constexpr std::size_t kSamples       = 11;
constexpr std::size_t kBatchSize     = 5;

struct svi_parameters
{
    double a;
    double b;
    double rho;
    double m;
    double sigma;
};

struct paper_quote
{
    double strike;
    double call;
    double put;
    double market_total_variance;
    double published_svi_variance;
};

// Tables 3.1 and 3.2, 2019-04-05 to 2020-04-06 (T = 1.01 years).
constexpr std::array<paper_quote, 13> kPaperQuotes{{
    {2068.48, 1268.59, 7.25, 0.06249, 0.06361},
    {2413.23, 936.83, 21.56, 0.05000, 0.04935},
    {2757.98, 621.99, 52.79, 0.03780, 0.03720},
    {3016.54, 407.05, 97.39, 0.02964, 0.02932},
    {3585.37, 77.18, 338.53, 0.01662, 0.01674},
    {3964.59, 15.10, 657.11, 0.01501, 0.01470},
    {4481.71, 1.87, 1162.98, 0.01694, 0.01700},
    {4998.83, 0.35, 1680.55, 0.02018, 0.02037},
    {5688.33, 0.05, 2372.38, 0.02462, 0.02479},
    {6033.07, 0.02, 2718.41, 0.02678, 0.02688},
    {6377.82, 0.01, 3064.46, 0.02892, 0.02887},
    {6722.57, 0.01, 3410.52, 0.03100, 0.03077},
    {6894.94, 0.00, 3583.55, 0.03207, 0.03169},
}};

// Section 3.5.2 reports this rounded tuple in (a, b, rho, m, sigma) order.
constexpr svi_parameters kPublishedParameters{0.01, 0.07, 0.43, 0.11, 0.12};

double sigmoid(double x)
{
    if (x >= 0.0)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }
    const double exp_x = std::exp(x);
    return exp_x / (1.0 + exp_x);
}

double logit(double probability)
{
    return std::log(probability / (1.0 - probability));
}

double svi_total_variance(double log_moneyness, const svi_parameters& parameters)
{
    const double centered = log_moneyness - parameters.m;
    return parameters.a +
           parameters.b * (parameters.rho * centered +
                              std::sqrt(centered * centered + parameters.sigma * parameters.sigma));
}

double density_factor(double log_moneyness, const svi_parameters& parameters)
{
    const double centered = log_moneyness - parameters.m;
    const double root     = std::sqrt(centered * centered + parameters.sigma * parameters.sigma);
    const double variance = svi_total_variance(log_moneyness, parameters);
    const double first    = parameters.b * (parameters.rho + centered / root);
    const double second = parameters.b * parameters.sigma * parameters.sigma / (root * root * root);

    const double leading = 1.0 - log_moneyness * first / (2.0 * variance);
    return leading * leading - 0.25 * first * first * (1.0 / variance + 0.25) + 0.5 * second;
}

class svi_calibration_problem
{
public:
    svi_calibration_problem() : forward_(infer_forward())
    {
        log_moneyness_.reserve(kPaperQuotes.size());
        market_variance_.reserve(kPaperQuotes.size());
        published_variance_.reserve(kPaperQuotes.size());
        for (const auto& quote : kPaperQuotes)
        {
            log_moneyness_.push_back(std::log(quote.strike / forward_));
            market_variance_.push_back(quote.market_total_variance);
            published_variance_.push_back(quote.published_svi_variance);
        }

        const auto [minimum_k, maximum_k] =
            std::minmax_element(log_moneyness_.begin(), log_moneyness_.end());
        lower_ = {1e-5, 0.001, -1.0, 2.0 * *minimum_k, 0.01};
        upper_ = {*std::max_element(market_variance_.begin(), market_variance_.end()),
            1.0,
            1.0,
            2.0 * *maximum_k,
            1.0};
    }

    std::size_t num_quotes() const noexcept { return log_moneyness_.size(); }
    double      forward() const noexcept { return forward_; }

    vector_type initial_guess() const
    {
        // Equation 3.17: (min(w_market)/2, 0.1, -0.5, 0.1, 0.1).
        return encode({0.5 * *std::min_element(market_variance_.begin(), market_variance_.end()),
            0.1,
            -0.5,
            0.1,
            0.1});
    }

    svi_parameters decode(const vector_type& unconstrained) const
    {
        return {decode_component(unconstrained[0], 0),
            decode_component(unconstrained[1], 1),
            decode_component(unconstrained[2], 2),
            decode_component(unconstrained[3], 3),
            decode_component(unconstrained[4], 4)};
    }

    void residuals(const vector_type& unconstrained, vector_type& residuals) const
    {
        const svi_parameters parameters = decode(unconstrained);
        for (std::size_t i = 0; i < num_quotes(); ++i)
        {
            residuals[static_cast<index_type>(i)] =
                kVarianceScale *
                (svi_total_variance(log_moneyness_[i], parameters) - market_variance_[i]);
        }
    }

    void jacobian(const vector_type& unconstrained, matrix_type& jacobian) const
    {
        const svi_parameters               parameters = decode(unconstrained);
        std::array<double, kNumParameters> chain{};
        for (std::size_t i = 0; i < kNumParameters; ++i)
        {
            const double unit = sigmoid(unconstrained[static_cast<index_type>(i)]);
            chain[i]          = (upper_[i] - lower_[i]) * unit * (1.0 - unit);
        }

        for (std::size_t i = 0; i < num_quotes(); ++i)
        {
            const double centered = log_moneyness_[i] - parameters.m;
            const double root =
                std::sqrt(centered * centered + parameters.sigma * parameters.sigma);
            const index_type row = static_cast<index_type>(i);

            jacobian(row, 0) = kVarianceScale * chain[0];
            jacobian(row, 1) = kVarianceScale * (parameters.rho * centered + root) * chain[1];
            jacobian(row, 2) = kVarianceScale * parameters.b * centered * chain[2];
            jacobian(row, 3) =
                kVarianceScale * parameters.b * (-parameters.rho - centered / root) * chain[3];
            jacobian(row, 4) = kVarianceScale * parameters.b * parameters.sigma / root * chain[4];
        }
    }

    double objective(const vector_type& unconstrained) const
    {
        vector_type values = make_vector(num_quotes());
        residuals(unconstrained, values);
        return 0.5 * values.squaredNorm();
    }

    void gradient(const vector_type& unconstrained, vector_type& gradient) const
    {
        vector_type values      = make_vector(num_quotes());
        matrix_type derivatives = make_matrix(num_quotes(), kNumParameters);
        residuals(unconstrained, values);
        jacobian(unconstrained, derivatives);
        gradient = derivatives.transpose() * values;
    }

    double rmse_variance_bps(const vector_type& unconstrained) const
    {
        vector_type values = make_vector(num_quotes());
        residuals(unconstrained, values);
        return values.norm() / std::sqrt(static_cast<double>(num_quotes()));
    }

    double max_variance_error_bps(const vector_type& unconstrained) const
    {
        vector_type values = make_vector(num_quotes());
        residuals(unconstrained, values);
        return values.cwiseAbs().maxCoeff();
    }

    double rmse_variance_bps(const svi_parameters& parameters) const
    {
        double squared_error = 0.0;
        for (std::size_t i = 0; i < num_quotes(); ++i)
        {
            const double error =
                kVarianceScale *
                (svi_total_variance(log_moneyness_[i], parameters) - market_variance_[i]);
            squared_error += error * error;
        }
        return std::sqrt(squared_error / static_cast<double>(num_quotes()));
    }

    double published_curve_rmse_bps() const
    {
        double squared_error = 0.0;
        for (std::size_t i = 0; i < num_quotes(); ++i)
        {
            const double error = kVarianceScale * (published_variance_[i] - market_variance_[i]);
            squared_error += error * error;
        }
        return std::sqrt(squared_error / static_cast<double>(num_quotes()));
    }

    static double minimum_density_factor(const svi_parameters& parameters)
    {
        constexpr int    intervals = 40000;
        constexpr double lower_k   = -10.0;
        constexpr double upper_k   = 10.0;
        double           minimum   = std::numeric_limits<double>::infinity();
        for (int i = 0; i <= intervals; ++i)
        {
            const double k = lower_k + (upper_k - lower_k) * static_cast<double>(i) /
                                           static_cast<double>(intervals);
            minimum        = std::min(minimum, density_factor(k, parameters));
        }
        return minimum;
    }

    void validate_analytic_jacobian() const
    {
        const vector_type point    = initial_guess();
        matrix_type       analytic = make_matrix(num_quotes(), kNumParameters);
        jacobian(point, analytic);

        double maximum_error = 0.0;
        for (std::size_t column = 0; column < kNumParameters; ++column)
        {
            vector_type  plus  = point;
            vector_type  minus = point;
            const double step  = 1e-6;
            plus[static_cast<index_type>(column)] += step;
            minus[static_cast<index_type>(column)] -= step;
            vector_type plus_values  = make_vector(num_quotes());
            vector_type minus_values = make_vector(num_quotes());
            residuals(plus, plus_values);
            residuals(minus, minus_values);
            const vector_type numerical = (plus_values - minus_values) / (2.0 * step);
            maximum_error               = std::max(maximum_error,
                (analytic.col(static_cast<index_type>(column)) - numerical).cwiseAbs().maxCoeff());
        }
        if (maximum_error > 1e-4)
        {
            throw std::runtime_error("raw-SVI analytic Jacobian validation failed");
        }
    }

private:
    static double infer_forward()
    {
        // Regress C-P = alpha + beta*K. Then discount=-beta and F=alpha/discount.
        double mean_strike = 0.0;
        double mean_parity = 0.0;
        for (const auto& quote : kPaperQuotes)
        {
            mean_strike += quote.strike;
            mean_parity += quote.call - quote.put;
        }
        mean_strike /= static_cast<double>(kPaperQuotes.size());
        mean_parity /= static_cast<double>(kPaperQuotes.size());

        double covariance      = 0.0;
        double strike_variance = 0.0;
        for (const auto& quote : kPaperQuotes)
        {
            const double centered_strike = quote.strike - mean_strike;
            covariance += centered_strike * (quote.call - quote.put - mean_parity);
            strike_variance += centered_strike * centered_strike;
        }
        const double beta     = covariance / strike_variance;
        const double alpha    = mean_parity - beta * mean_strike;
        const double discount = -beta;
        return alpha / discount;
    }

    double decode_component(double unconstrained, std::size_t component) const
    {
        return lower_[component] + (upper_[component] - lower_[component]) * sigmoid(unconstrained);
    }

    vector_type encode(const svi_parameters& parameters) const
    {
        const std::array<double, kNumParameters> raw{
            parameters.a, parameters.b, parameters.rho, parameters.m, parameters.sigma};
        vector_type encoded = make_vector(kNumParameters);
        for (std::size_t i = 0; i < kNumParameters; ++i)
        {
            const double unit                   = (raw[i] - lower_[i]) / (upper_[i] - lower_[i]);
            encoded[static_cast<index_type>(i)] = logit(unit);
        }
        return encoded;
    }

    double                             forward_ = 0.0;
    std::array<double, kNumParameters> lower_{};
    std::array<double, kNumParameters> upper_{};
    std::vector<double>                log_moneyness_;
    std::vector<double>                market_variance_;
    std::vector<double>                published_variance_;
};

least_squares_problem make_least_squares_problem(
    const std::shared_ptr<const svi_calibration_problem>& calibration, bool with_jacobian)
{
    least_squares_problem problem;
    problem.num_parameters = kNumParameters;
    problem.num_residuals  = calibration->num_quotes();
    problem.residuals      = [calibration](const vector_type& x, vector_type& residuals)
    { calibration->residuals(x, residuals); };
    if (with_jacobian)
    {
        problem.jacobian = [calibration](const vector_type& x, matrix_type& jacobian)
        { calibration->jacobian(x, jacobian); };
    }
    return problem;
}

optimization_problem make_objective_problem(
    const std::shared_ptr<const svi_calibration_problem>& calibration)
{
    optimization_problem problem;
    problem.num_parameters = kNumParameters;
    problem.objective = [calibration](const vector_type& x) { return calibration->objective(x); };
    problem.gradient  = [calibration](const vector_type& x, vector_type& gradient)
    { calibration->gradient(x, gradient); };
    return problem;
}

struct benchmark_case
{
    std::string                    name;
    std::string                    backend_name;
    std::function<solver_result()> solve;
};

struct benchmark_row
{
    solver_result  result;
    bool           available     = false;
    double         median_us     = 0.0;
    double         rmse_bps      = std::numeric_limits<double>::quiet_NaN();
    double         max_error_bps = std::numeric_limits<double>::quiet_NaN();
    double         minimum_g     = std::numeric_limits<double>::quiet_NaN();
    svi_parameters parameters{};
};

std::vector<benchmark_case> make_cases(
    const std::shared_ptr<const svi_calibration_problem>& calibration)
{
    const least_squares_problem analytical_ls    = make_least_squares_problem(calibration, true);
    const least_squares_problem residual_only_ls = make_least_squares_problem(calibration, false);
    const optimization_problem  objective        = make_objective_problem(calibration);
    const vector_type           initial          = calibration->initial_guess();

    auto least_squares_case = [analytical_ls, initial](backend     selected_backend,
                                  algorithm                        selected_algorithm,
                                  std::optional<petsc_tao_options> petsc = std::nullopt)
    {
        solve_options options;
        options.backend            = selected_backend;
        options.algorithm          = selected_algorithm;
        options.derivatives        = derivative_mode::supplied;
        options.max_iterations     = 1000;
        options.function_tolerance = 1e-12;
        options.gradient_tolerance =
            selected_algorithm == algorithm::levenberg_marquardt ? 1e-3 : 1e-8;
        options.parameter_tolerance = 1e-12;
        options.petsc_tao           = std::move(petsc);
        return [analytical_ls, initial, options]()
        { return solve(analytical_ls, initial, options); };
    };

    auto objective_case = [objective, initial](backend         selected_backend,
                              algorithm                        selected_algorithm,
                              std::optional<ipopt_options>     ipopt = std::nullopt,
                              std::optional<petsc_tao_options> petsc = std::nullopt)
    {
        solve_options options;
        options.backend             = selected_backend;
        options.algorithm           = selected_algorithm;
        options.derivatives         = derivative_mode::supplied;
        options.max_iterations      = 1000;
        options.function_tolerance  = 1e-12;
        options.gradient_tolerance  = 1e-8;
        options.parameter_tolerance = 1e-12;
        options.ipopt               = std::move(ipopt);
        options.petsc_tao           = std::move(petsc);
        return [objective, initial, options]() { return solve(objective, initial, options); };
    };

    solve_options pounders_options;
    pounders_options.backend        = backend::pounders;
    pounders_options.algorithm      = algorithm::pounders;
    pounders_options.max_iterations = 1000;
    pounders_options.petsc_tao =
        petsc_tao_options{.algorithm = tao_algorithm::pounders, .gatol = 1e-8, .grtol = 1e-8};

    return {
        {"Native LM",
            "native",
            least_squares_case(backend::native, algorithm::levenberg_marquardt)},
        {"Native GN", "native", least_squares_case(backend::native, algorithm::gauss_newton)},
        {"Native L-BFGS", "native", objective_case(backend::native, algorithm::lbfgs)},
        {"Ceres LM", "ceres", least_squares_case(backend::ceres, algorithm::automatic)},
        {"Ipopt L-BFGS",
            "ipopt",
            objective_case(backend::ipopt,
                algorithm::automatic,
                ipopt_options{.hessian_mode = ipopt_hessian_mode::limited_memory,
                    .tol                    = 1e-8,
                    .acceptable_tol         = 1e-6})},
        {"TAO BRGN",
            "petsc_tao",
            least_squares_case(backend::petsc_tao,
                algorithm::automatic,
                petsc_tao_options{.algorithm = tao_algorithm::brgn, .gatol = 1e-8, .grtol = 1e-8})},
        {"TAO LMVM",
            "petsc_tao",
            objective_case(backend::petsc_tao,
                algorithm::automatic,
                std::nullopt,
                petsc_tao_options{.algorithm = tao_algorithm::lmvm, .gatol = 1e-3, .grtol = 1e-8})},
        {"POUNDERS",
            "pounders",
            [residual_only_ls, initial, pounders_options]()
            { return solve(residual_only_ls, initial, pounders_options); }},
    };
}

std::vector<benchmark_row> measure(std::vector<benchmark_case>& cases,
    const std::shared_ptr<const svi_calibration_problem>&       calibration)
{
    std::vector<benchmark_row>       rows(cases.size());
    std::vector<std::vector<double>> samples(cases.size());
    for (std::size_t i = 0; i < cases.size(); ++i)
    {
        rows[i].result    = cases[i].solve();
        rows[i].available = rows[i].result.status != solver_status::backend_unavailable;
        samples[i].reserve(kSamples);
    }

    for (std::size_t round = 0; round < kWarmupRounds; ++round)
    {
        for (std::size_t position = 0; position < cases.size(); ++position)
        {
            const std::size_t index = (round + position) % cases.size();
            if (rows[index].available)
            {
                rows[index].result = cases[index].solve();
            }
        }
    }

    for (std::size_t sample = 0; sample < kSamples; ++sample)
    {
        for (std::size_t position = 0; position < cases.size(); ++position)
        {
            const std::size_t index = (sample + position) % cases.size();
            if (!rows[index].available)
            {
                continue;
            }

            const auto start = clock_type::now();
            for (std::size_t run = 0; run < kBatchSize; ++run)
            {
                rows[index].result = cases[index].solve();
            }
            const auto   end = clock_type::now();
            const double elapsed_us =
                std::chrono::duration<double, std::micro>(end - start).count();
            samples[index].push_back(elapsed_us / static_cast<double>(kBatchSize));
        }
    }

    for (std::size_t i = 0; i < cases.size(); ++i)
    {
        if (!rows[i].available)
        {
            continue;
        }
        std::sort(samples[i].begin(), samples[i].end());
        rows[i].median_us = samples[i][samples[i].size() / 2];

        if (rows[i].result.parameters.size() == static_cast<index_type>(kNumParameters))
        {
            rows[i].rmse_bps      = calibration->rmse_variance_bps(rows[i].result.parameters);
            rows[i].max_error_bps = calibration->max_variance_error_bps(rows[i].result.parameters);
            rows[i].parameters    = calibration->decode(rows[i].result.parameters);
            rows[i].minimum_g = svi_calibration_problem::minimum_density_factor(rows[i].parameters);
        }
    }
    return rows;
}

void print_results(const svi_calibration_problem& calibration,
    const std::vector<benchmark_case>&            cases,
    const std::vector<benchmark_row>&             rows)
{
    std::cout << "\nFerhati (2020) EURO STOXX 50 raw-SVI fixture\n"
              << "Quotes: " << calibration.num_quotes()
              << ", T=1.01Y, parity-implied F=" << std::fixed << std::setprecision(4)
              << calibration.forward() << '\n'
              << "Published Table 3.2 curve RMSE: " << std::setprecision(5)
              << calibration.published_curve_rmse_bps() << " variance bp\n"
              << "Published rounded parameters evaluated on the recovered log-moneyness: "
              << calibration.rmse_variance_bps(kPublishedParameters) << " variance bp RMSE, "
              << "min g[-10,10]="
              << svi_calibration_problem::minimum_density_factor(kPublishedParameters) << '\n'
              << "The published tuple has rho=+0.43; the tabulated left skew calibrates to "
                 "rho<0.\n\n";

    std::cout << std::left << std::setw(18) << "Solver" << std::setw(12) << "Backend"
              << std::setw(17) << "Status" << std::setw(9) << "Iters" << std::setw(15)
              << "Median (us)" << std::setw(14) << "RMSE (bp)" << std::setw(14) << "Max err (bp)"
              << std::setw(14) << "Min g" << '\n';
    std::cout << std::string(113, '-') << '\n';

    for (std::size_t i = 0; i < cases.size(); ++i)
    {
        std::cout << std::left << std::setw(18) << cases[i].name << std::setw(12)
                  << cases[i].backend_name;
        if (!rows[i].available)
        {
            std::cout << std::setw(17) << "not compiled" << '\n';
            continue;
        }

        std::cout << std::setw(17) << to_string(rows[i].result.status);
        if (rows[i].result.iterations == 0 && rows[i].result.backend != backend::native &&
            rows[i].result.backend != backend::ceres)
        {
            std::cout << std::setw(9) << "-";
        }
        else
        {
            std::cout << std::setw(9) << rows[i].result.iterations;
        }
        std::cout << std::setw(15) << std::fixed << std::setprecision(2) << rows[i].median_us
                  << std::setw(14) << std::setprecision(5) << rows[i].rmse_bps << std::setw(14)
                  << rows[i].max_error_bps << std::setw(14) << rows[i].minimum_g << '\n';
    }

    std::cout << "\nCalibrated raw-SVI parameters (a, b, rho, m, sigma)\n";
    std::cout << std::left << std::setw(18) << "Solver" << std::setw(13) << "a" << std::setw(13)
              << "b" << std::setw(13) << "rho" << std::setw(13) << "m" << std::setw(13) << "sigma"
              << '\n';
    std::cout << std::string(83, '-') << '\n';
    std::cout << std::left << std::setw(18) << "Paper (rounded)" << std::setw(13) << std::fixed
              << std::setprecision(7) << kPublishedParameters.a << std::setw(13)
              << kPublishedParameters.b << std::setw(13) << kPublishedParameters.rho
              << std::setw(13) << kPublishedParameters.m << std::setw(13)
              << kPublishedParameters.sigma << '\n';
    for (std::size_t i = 0; i < cases.size(); ++i)
    {
        if (!rows[i].available ||
            rows[i].result.parameters.size() != static_cast<index_type>(kNumParameters))
        {
            continue;
        }
        const auto& p = rows[i].parameters;
        std::cout << std::left << std::setw(18) << cases[i].name << std::setw(13) << std::fixed
                  << std::setprecision(7) << p.a << std::setw(13) << p.b << std::setw(13) << p.rho
                  << std::setw(13) << p.m << std::setw(13) << p.sigma << '\n';
    }
}

}  // namespace

int main()
{
    auto calibration = std::make_shared<svi_calibration_problem>();
    calibration->validate_analytic_jacobian();
    auto       cases = make_cases(calibration);
    const auto rows  = measure(cases, calibration);
    print_results(*calibration, cases, rows);
    return 0;
}
