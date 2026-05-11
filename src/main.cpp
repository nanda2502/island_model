#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <omp.h>

#include "config/default_grid.hpp"
#include "config/parameter_grid.hpp"
#include "dynamics/simulator.hpp"
#include "io/write_equilibrium_csv.hpp"
#include "io/write_run_parameters_csv.hpp"
#include "io/write_time_bookkeeping_csv.hpp"
#include "model/lattice.hpp"
#include "model/payoff_landscape.hpp"
#include "state_space/reachable_states.hpp"

namespace im = island_model;

namespace {

[[nodiscard]] std::filesystem::path make_temp_results_dir() {
    const auto output_root = std::filesystem::current_path().parent_path();
    const auto timestamp =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();

    for (int attempt = 0; attempt < 1000; ++attempt) {
        auto dir = output_root / (
            "island_model_results_tmp_"
            + std::to_string(timestamp)
            + "_"
            + std::to_string(attempt));
        if (std::filesystem::create_directory(dir)) {
            return dir;
        }
    }

    throw std::runtime_error("Failed to create temporary results directory");
}

[[nodiscard]] std::filesystem::path run_fragment_path(const std::filesystem::path& temp_dir,
                                                      const std::string& stem,
                                                      std::size_t run_index) {
    return temp_dir / (stem + "_" + std::to_string(run_index) + ".csv");
}

[[nodiscard]] std::ofstream open_output_file(const std::filesystem::path& path) {
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open " + path.string());
    }
    return file;
}

void append_file_contents(std::ostream& output, const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open temporary result " + path.string());
    }

    output << input.rdbuf();
    if (!output) {
        throw std::runtime_error("Failed while writing combined result from " + path.string());
    }
    if (input.bad()) {
        throw std::runtime_error("Failed while reading temporary result " + path.string());
    }
}

void write_equilibrium_failure_row(std::ostream& out,
                                   const im::RunConfig& cfg,
                                   std::size_t run_index) {
    out << run_index << ','
        << cfg.run_id << ','
        << 0 << ','
        << 0 << ','
        << 0.0 << ','
        << 0 << ','
        << 0 << ','
        << 0.0 << ','
        << 0.0 << ','
        << 0.0 << ','
        << 0.0 << ','
        << 0.0 << ','
        << 0.0 << '\n';
}

void write_trait_failure_row(std::ostream& out,
                             const im::RunConfig& cfg,
                             std::size_t run_index) {
    out << run_index << ','
        << cfg.run_id << ','
        << 0 << ','
        << 0 << ','
        << 0 << ','
        << 0 << ','
        << 1 << ','
        << 0.0 << '\n';
}

void write_equilibrium_summary_failure_row(std::ostream& out,
                                           const im::RunConfig& cfg,
                                           std::size_t run_index) {
    const im::EquilibriumRunMetadata meta{
        .run_index = run_index,
        .run_id = cfg.run_id,
        .converged = 0,
        .steps_to_equilibrium = 0,
        .final_distance = -1.0
    };
    im::EquilibriumSummaryCsvWriter::write_row(
        out,
        meta,
        im::PopulationBookkeepingSnapshot{});
}

void write_run_parameters_failure_row(std::ostream& out,
                                      const im::RunConfig& cfg,
                                      std::size_t run_index) {
    const im::EquilibriumRunMetadata meta{
        .run_index = run_index,
        .run_id = cfg.run_id,
        .seed = cfg.seed,
        .columns = cfg.columns,
        .layers = cfg.layers,
        .cross_column_depth = cfg.cross_column_depth,
        .island_count = cfg.island_count,
        .strictness = cfg.strictness,
        .m = cfg.m,
        .rho = cfg.rho,
        .mu = cfg.mu,
        .alpha = cfg.alpha,
        .beta = cfg.beta,
        .lambda = cfg.lambda,
        .gamma = cfg.gamma,
        .eta = cfg.eta,
        .delta = cfg.delta,
        .sigma_b = cfg.sigma_b,
        .sigma_nu = cfg.sigma_nu,
        .k = cfg.k,
        .converged = 0,
        .steps_to_equilibrium = 0,
        .final_distance = -1.0
    };
    im::RunParametersCsvWriter::write_row(out, meta);
}

[[nodiscard]] bool dense_payoff_cache_enabled() {
    const char* value = std::getenv("ISLAND_MODEL_DENSE_PAYOFF_CACHE");
    if (value == nullptr) {
        return true;
    }

    const std::string flag(value);
    return flag != "0" && flag != "false" && flag != "FALSE"
        && flag != "off" && flag != "OFF";
}

[[nodiscard]] std::uint64_t parse_positive_u64_env(const char* name) {
    if (const char* value = std::getenv(name)) {
        char* end = nullptr;
        const auto parsed = std::strtoull(value, &end, 10);
        if (end != value && *end == '\0' && parsed > 0) {
            return static_cast<std::uint64_t>(parsed);
        }

        std::cerr << "Warning: ignoring invalid " << name << "="
                  << value << '\n';
    }

    return 0;
}

[[nodiscard]] double parse_memory_fraction() {
    if (const char* value = std::getenv("ISLAND_MODEL_MEMORY_FRACTION")) {
        char* end = nullptr;
        const double parsed = std::strtod(value, &end);
        if (end != value && *end == '\0' && parsed > 0.0 && parsed <= 1.0) {
            return parsed;
        }

        std::cerr << "Warning: ignoring invalid ISLAND_MODEL_MEMORY_FRACTION="
                  << value << '\n';
    }

    return 0.80;
}

[[nodiscard]] std::uint64_t configured_memory_budget_bytes() {
    constexpr std::uint64_t mib = 1024ULL * 1024ULL;
    constexpr std::uint64_t gib = 1024ULL * mib;

    if (const auto memory_gib = parse_positive_u64_env("ISLAND_MODEL_MEMORY_GIB")) {
        return memory_gib * gib;
    }
    if (const auto slurm_mem_mb = parse_positive_u64_env("SLURM_MEM_PER_NODE")) {
        return slurm_mem_mb * mib;
    }
    if (const auto slurm_mem_per_cpu_mb = parse_positive_u64_env("SLURM_MEM_PER_CPU")) {
        const auto threads = static_cast<std::uint64_t>(std::max(1, omp_get_max_threads()));
        return slurm_mem_per_cpu_mb * threads * mib;
    }

    return 384ULL * gib;
}

[[nodiscard]] std::uint64_t saturating_multiply(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return a * b;
}

[[nodiscard]] std::uint64_t estimate_run_memory_bytes(const im::RunConfig& cfg,
                                                      std::size_t state_count,
                                                      bool dense_payoff_cache) {
    constexpr std::uint64_t double_bytes = sizeof(double);
    constexpr std::uint64_t overhead_bytes = 1024ULL * 1024ULL * 1024ULL;

    const auto island_count = static_cast<std::uint64_t>(cfg.island_count);
    const auto states = static_cast<std::uint64_t>(state_count);
    const auto population_bytes =
        saturating_multiply(saturating_multiply(island_count, states), double_bytes);

    // Current, migrated, next, and Visibility::repertoire_weights can coexist.
    std::uint64_t estimate = saturating_multiply(population_bytes, 4);

    if (dense_payoff_cache) {
        estimate += population_bytes; // expression-bias denominator table.
        if (cfg.lambda != 0.0) {
            estimate += population_bytes; // payoff-sum table.
        }
    }

    return estimate + overhead_bytes;
}

[[nodiscard]] int configured_parallel_run_count(std::uint64_t max_estimated_run_memory_bytes,
                                                std::size_t run_count) {
    const int max_threads = std::max(1, omp_get_max_threads());

    if (const char* value = std::getenv("ISLAND_MODEL_MAX_PARALLEL_RUNS")) {
        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end != value && *end == '\0' && parsed > 0) {
            return std::min({static_cast<int>(parsed), max_threads, static_cast<int>(run_count)});
        }

        std::cerr << "Warning: ignoring invalid ISLAND_MODEL_MAX_PARALLEL_RUNS="
                  << value << '\n';
    }

    if (max_estimated_run_memory_bytes == 0) {
        return 1;
    }

    const double usable_memory =
        static_cast<double>(configured_memory_budget_bytes()) * parse_memory_fraction();
    const auto memory_limited_runs =
        static_cast<int>(std::max(1.0, usable_memory / static_cast<double>(max_estimated_run_memory_bytes)));

    return std::max(
        1,
        std::min({max_threads, memory_limited_runs, static_cast<int>(run_count)}));
}

[[nodiscard]] double gib(std::uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const bool logging_enabled =
            argc > 1 && std::string(argv[1]) == "1";
        const bool write_equilibrium_distribution =
            argc <= 2 || std::string(argv[2]) != "0";

        const auto grid = im::make_default_grid();
        const auto runs = im::make_parameter_combinations(grid);

        struct LatticeConfigKey {
            std::size_t columns{0};
            std::size_t layers{0};
            std::size_t cross_column_depth{0};
            double strictness{1.0};

            [[nodiscard]] bool operator==(const LatticeConfigKey& other) const noexcept {
                return columns == other.columns
                    && layers == other.layers
                    && cross_column_depth == other.cross_column_depth
                    && strictness == other.strictness;
            }
        };

        struct LatticeConfigKeyHash {
            [[nodiscard]] std::size_t operator()(const LatticeConfigKey& key) const noexcept {
                return std::hash<std::size_t>{}(key.columns)
                    ^ (std::hash<std::size_t>{}(key.layers) << 1U)
                    ^ (std::hash<std::size_t>{}(key.cross_column_depth) << 2U)
                    ^ (std::hash<double>{}(key.strictness) << 3U);
            }
        };

        using reachable_states_ptr = std::shared_ptr<const im::ReachableStates>;

        std::unordered_map<LatticeConfigKey, reachable_states_ptr, LatticeConfigKeyHash> reachable_states_cache;
        reachable_states_cache.reserve(runs.size());
        std::size_t max_repertoire_count = 0;
        std::uint64_t max_estimated_run_memory_bytes = 0;
        const bool dense_payoff_cache = dense_payoff_cache_enabled();

        for (const auto& cfg : runs) {
            const LatticeConfigKey key{
                .columns = cfg.columns,
                .layers = cfg.layers,
                .cross_column_depth = cfg.cross_column_depth,
                .strictness = cfg.strictness
            };
            if (reachable_states_cache.contains(key)) {
                continue;
            }

            const im::Lattice lattice(cfg.columns, cfg.layers, cfg.cross_column_depth);

            if (logging_enabled) {
                std::cout << "[lattice " << cfg.columns << "x" << cfg.layers
                          << ", cross-column depth " << cfg.cross_column_depth
                          << ", strictness " << cfg.strictness
                          << "] preprocessing: building reachable states" << '\n';
            }

            const auto log_reachable_states_progress =
                [&](std::size_t processed_states,
                    std::size_t discovered_states,
                    std::size_t frontier_size) {
                    std::cout << "[lattice " << cfg.columns << "x" << cfg.layers
                              << ", cross-column depth " << cfg.cross_column_depth
                              << ", strictness " << cfg.strictness
                              << "] reachable states: processed " << processed_states
                              << ", discovered " << discovered_states
                              << ", frontier " << frontier_size << '\n';
                };

            auto states = std::make_shared<im::ReachableStates>(
                lattice,
                cfg.strictness,
                logging_enabled ? 100 : 0,
                logging_enabled
                    ? im::ReachableStates::build_progress_callback_type(log_reachable_states_progress)
                    : im::ReachableStates::build_progress_callback_type{}
            );
            max_repertoire_count = std::max(max_repertoire_count, states->size());

            reachable_states_cache.emplace(
                key,
                std::move(states));
        }

        for (const auto& cfg : runs) {
            const LatticeConfigKey key{
                .columns = cfg.columns,
                .layers = cfg.layers,
                .cross_column_depth = cfg.cross_column_depth,
                .strictness = cfg.strictness
            };
            const auto states_it = reachable_states_cache.find(key);
            if (states_it == reachable_states_cache.end()) {
                throw std::logic_error("Missing reachable states cache entry for memory estimate");
            }

            max_estimated_run_memory_bytes = std::max(
                max_estimated_run_memory_bytes,
                estimate_run_memory_bytes(cfg, states_it->second->size(), dense_payoff_cache));
        }

        const auto temp_results_dir = make_temp_results_dir();
        const bool parallelize_single_run_by_island =
            runs.size() == 1 && runs.front().m == 0.0;
        const int parallel_run_count =
            configured_parallel_run_count(max_estimated_run_memory_bytes, runs.size());
        const bool parallelize_runs =
            !parallelize_single_run_by_island && parallel_run_count > 1;

        std::cout << "Max reachable repertoires: " << max_repertoire_count
                  << "; estimated peak/run: " << gib(max_estimated_run_memory_bytes) << " GiB"
                  << "; memory budget: " << gib(configured_memory_budget_bytes()) << " GiB"
                  << "; dense payoff cache: " << (dense_payoff_cache ? "on" : "off")
                  << "; parallel runs: "
                  << (parallelize_runs ? parallel_run_count : 1)
                  << '\n';

        #pragma omp parallel for schedule(dynamic) if(parallelize_runs) num_threads(parallel_run_count)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(runs.size()); ++i) {
            const auto run_index = static_cast<std::size_t>(i);
            const auto& cfg = runs[run_index];
            const auto equilibrium_path =
                run_fragment_path(temp_results_dir, "equilibrium", run_index);
            const auto equilibrium_summary_path =
                run_fragment_path(temp_results_dir, "equilibrium_summary", run_index);
            const auto trait_path =
                run_fragment_path(temp_results_dir, "trait_equilibrium", run_index);
            const auto bookkeeping_path =
                run_fragment_path(temp_results_dir, "time_bookkeeping", run_index);
            const auto run_parameters_path =
                run_fragment_path(temp_results_dir, "run_parameters", run_index);

            try {
                const im::Lattice lattice(cfg.columns, cfg.layers, cfg.cross_column_depth);
                const LatticeConfigKey lattice_key{
                    .columns = cfg.columns,
                    .layers = cfg.layers,
                    .cross_column_depth = cfg.cross_column_depth,
                    .strictness = cfg.strictness
                };
                const auto states_it = reachable_states_cache.find(lattice_key);
                if (states_it == reachable_states_cache.end()) {
                    throw std::logic_error("Missing reachable states cache entry for lattice");
                }
                const reachable_states_ptr& states = states_it->second;

                im::DynamicsParams dynamics{
                    .m = cfg.m,
                    .rho = cfg.rho,
                    .mu = cfg.mu,
                    .alpha = cfg.alpha,
                    .beta = cfg.beta,
                    .lambda = cfg.lambda,
                    .gamma = cfg.gamma,
                    .eta = cfg.eta
                };

                im::PayoffParams payoff_params{
                    .delta = cfg.delta,
                    .sigma_b = cfg.sigma_b,
                    .sigma_nu = cfg.sigma_nu,
                    .k = cfg.k
                };

                std::mt19937_64 rng(cfg.seed);

                if (logging_enabled) {
                    #pragma omp critical
                    {
                        std::cout << "[run " << cfg.run_id << "] preprocessing: building payoff landscape" << '\n';
                    }
                }
                im::PayoffLandscape payoff(lattice, cfg.island_count, payoff_params, rng);

                im::Simulator simulator(states, payoff, dynamics);

                auto state = simulator.initial_state_all_empty();

                const std::size_t max_steps = 50000;
                const double tolerance = 1e-10;
                const std::size_t bookkeeping_interval = 50;

                const auto log_progress = [&](std::size_t step) {
                    #pragma omp critical
                    {
                        std::cout << "[run " << cfg.run_id << "] step " << step << '\n';
                    }
                };

                const auto eq = simulator.run_to_equilibrium(
                    std::move(state),
                    max_steps,
                    tolerance,
                    logging_enabled ? 100 : 0,
                    logging_enabled ? log_progress : std::function<void(std::size_t)>{},
                    bookkeeping_interval
                );

                const im::EquilibriumRunMetadata meta{
                    .run_index = run_index,
                    .run_id = cfg.run_id,
                    .seed = cfg.seed,
                    .columns = cfg.columns,
                    .layers = cfg.layers,
                    .cross_column_depth = cfg.cross_column_depth,
                    .island_count = cfg.island_count,
                    .strictness = cfg.strictness,
                    .m = cfg.m,
                    .rho = cfg.rho,
                    .mu = cfg.mu,
                    .alpha = cfg.alpha,
                    .beta = cfg.beta,
                    .lambda = cfg.lambda,
                    .gamma = cfg.gamma,
                    .eta = cfg.eta,
                    .delta = cfg.delta,
                    .sigma_b = cfg.sigma_b,
                    .sigma_nu = cfg.sigma_nu,
                    .k = cfg.k,
                    .converged = static_cast<std::uint32_t>(eq.converged ? 1 : 0),
                    .steps_to_equilibrium = eq.steps,
                    .final_distance = eq.final_distance
                };

                auto run_parameters_out = open_output_file(run_parameters_path);
                im::RunParametersCsvWriter::write_row(run_parameters_out, meta);

                if (write_equilibrium_distribution) {
                    auto out = open_output_file(equilibrium_path);
                    im::EquilibriumCsvWriter::write_rows(
                        out,
                        meta,
                        eq.state,
                        *states,
                        payoff,
                        1e-6
                    );
                }

                auto equilibrium_summary_out = open_output_file(equilibrium_summary_path);
                im::EquilibriumSummaryCsvWriter::write_row(
                    equilibrium_summary_out,
                    meta,
                    eq.bookkeeping.back()
                );

                auto trait_out = open_output_file(trait_path);
                im::TraitEquilibriumCsvWriter::write_rows(
                    trait_out,
                    meta,
                    eq.state,
                    *states
                );

                auto bookkeeping_out = open_output_file(bookkeeping_path);
                im::TimeBookkeepingCsvWriter::write_rows(
                    bookkeeping_out,
                    meta,
                    eq.bookkeeping
                );
            } catch (const std::exception& e) {
                try {
                    auto run_parameters_out = open_output_file(run_parameters_path);
                    write_run_parameters_failure_row(run_parameters_out, cfg, run_index);

                    if (write_equilibrium_distribution) {
                        auto out = open_output_file(equilibrium_path);
                        write_equilibrium_failure_row(out, cfg, run_index);
                    }

                    auto equilibrium_summary_out = open_output_file(equilibrium_summary_path);
                    write_equilibrium_summary_failure_row(equilibrium_summary_out, cfg, run_index);

                    auto trait_out = open_output_file(trait_path);
                    write_trait_failure_row(trait_out, cfg, run_index);

                    auto bookkeeping_out = open_output_file(bookkeeping_path);
                } catch (const std::exception& write_error) {
                    #pragma omp critical
                    {
                        std::cerr << "Run " << cfg.run_id
                                  << " failed to write temporary result: "
                                  << write_error.what() << '\n';
                    }
                }

                #pragma omp critical
                {
                    std::cerr << "Run " << cfg.run_id << " failed: " << e.what() << '\n';
                }
            }
        }

        std::ofstream run_parameters_file("../run_parameters.csv");
        if (!run_parameters_file) {
            throw std::runtime_error("Failed to open run_parameters.csv");
        }

        im::RunParametersCsvWriter::write_header(run_parameters_file);
        for (std::size_t i = 0; i < runs.size(); ++i) {
            append_file_contents(run_parameters_file, run_fragment_path(temp_results_dir, "run_parameters", i));
        }

        if (write_equilibrium_distribution) {
            std::ofstream file("../equilibrium_distribution.csv");
            if (!file) {
                throw std::runtime_error("Failed to open equilibrium_distribution.csv");
            }

            im::EquilibriumCsvWriter::write_header(file);
            for (std::size_t i = 0; i < runs.size(); ++i) {
                append_file_contents(file, run_fragment_path(temp_results_dir, "equilibrium", i));
            }
        }

        std::ofstream equilibrium_summary_file("../equilibrium_summary.csv");
        if (!equilibrium_summary_file) {
            throw std::runtime_error("Failed to open equilibrium_summary.csv");
        }

        im::EquilibriumSummaryCsvWriter::write_header(equilibrium_summary_file);
        for (std::size_t i = 0; i < runs.size(); ++i) {
            append_file_contents(
                equilibrium_summary_file,
                run_fragment_path(temp_results_dir, "equilibrium_summary", i));
        }

        std::ofstream trait_file("../equilibrium_trait_distribution.csv");
        if (!trait_file) {
            throw std::runtime_error("Failed to open equilibrium_trait_distribution.csv");
        }

        im::TraitEquilibriumCsvWriter::write_header(trait_file);
        for (std::size_t i = 0; i < runs.size(); ++i) {
            append_file_contents(trait_file, run_fragment_path(temp_results_dir, "trait_equilibrium", i));
        }

        std::ofstream bookkeeping_file("../time_bookkeeping.csv");
        if (!bookkeeping_file) {
            throw std::runtime_error("Failed to open time_bookkeeping.csv");
        }

        im::TimeBookkeepingCsvWriter::write_header(bookkeeping_file);
        for (std::size_t i = 0; i < runs.size(); ++i) {
            append_file_contents(bookkeeping_file, run_fragment_path(temp_results_dir, "time_bookkeeping", i));
        }

        std::error_code remove_error;
        std::filesystem::remove_all(temp_results_dir, remove_error);
        if (remove_error) {
            std::cerr << "Warning: failed to remove temporary results directory "
                      << temp_results_dir << ": " << remove_error.message() << '\n';
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }
}
