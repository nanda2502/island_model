#include <cstdint>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "config/default_grid.hpp"
#include "config/parameter_grid.hpp"
#include "dynamics/simulator.hpp"
#include "io/write_equilibrium_csv.hpp"
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

void write_equilibrium_failure_row(std::ostream& out, const im::RunConfig& cfg) {
    out << cfg.run_id << ','
        << cfg.seed << ','
        << cfg.columns << ','
        << cfg.layers << ','
        << cfg.cross_column_depth << ','
        << cfg.island_count << ','
        << cfg.m << ','
        << cfg.rho << ','
        << cfg.mu << ','
        << cfg.alpha << ','
        << cfg.beta << ','
        << cfg.lambda << ','
        << cfg.gamma << ','
        << cfg.eta << ','
        << cfg.delta << ','
        << cfg.sigma_b << ','
        << cfg.sigma_nu << ','
        << cfg.k << ','
        << 0 << ','
        << 0 << ','
        << -1.0 << ','
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

void write_trait_failure_row(std::ostream& out, const im::RunConfig& cfg) {
    out << cfg.run_id << ','
        << cfg.seed << ','
        << cfg.columns << ','
        << cfg.layers << ','
        << cfg.cross_column_depth << ','
        << cfg.island_count << ','
        << cfg.m << ','
        << cfg.rho << ','
        << cfg.mu << ','
        << cfg.alpha << ','
        << cfg.beta << ','
        << cfg.lambda << ','
        << cfg.gamma << ','
        << cfg.eta << ','
        << cfg.delta << ','
        << cfg.sigma_b << ','
        << cfg.sigma_nu << ','
        << cfg.k << ','
        << 0 << ','
        << 0 << ','
        << -1.0 << ','
        << 0 << ','
        << 0 << ','
        << 0 << ','
        << 0 << ','
        << 1 << ','
        << 0.0 << '\n';
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const bool logging_enabled =
            argc > 1 && std::string(argv[1]) == "1";

        const auto grid = im::make_default_grid();
        const auto runs = im::make_parameter_combinations(grid);

        struct LatticeConfigKey {
            std::size_t columns{0};
            std::size_t layers{0};
            std::size_t cross_column_depth{0};

            [[nodiscard]] bool operator==(const LatticeConfigKey& other) const noexcept {
                return columns == other.columns
                    && layers == other.layers
                    && cross_column_depth == other.cross_column_depth;
            }
        };

        struct LatticeConfigKeyHash {
            [[nodiscard]] std::size_t operator()(const LatticeConfigKey& key) const noexcept {
                return std::hash<std::size_t>{}(key.columns)
                    ^ (std::hash<std::size_t>{}(key.layers) << 1U)
                    ^ (std::hash<std::size_t>{}(key.cross_column_depth) << 2U);
            }
        };

        using reachable_states_ptr = std::shared_ptr<const im::ReachableStates>;

        std::unordered_map<LatticeConfigKey, reachable_states_ptr, LatticeConfigKeyHash> reachable_states_cache;
        reachable_states_cache.reserve(runs.size());

        for (const auto& cfg : runs) {
            const LatticeConfigKey key{
                .columns = cfg.columns,
                .layers = cfg.layers,
                .cross_column_depth = cfg.cross_column_depth
            };
            if (reachable_states_cache.contains(key)) {
                continue;
            }

            const im::Lattice lattice(cfg.columns, cfg.layers, cfg.cross_column_depth);

            if (logging_enabled) {
                std::cout << "[lattice " << cfg.columns << "x" << cfg.layers
                          << ", cross-column depth " << cfg.cross_column_depth
                          << "] preprocessing: building reachable states" << '\n';
            }

            const auto log_reachable_states_progress =
                [&](std::size_t processed_states,
                    std::size_t discovered_states,
                    std::size_t frontier_size) {
                    std::cout << "[lattice " << cfg.columns << "x" << cfg.layers
                              << ", cross-column depth " << cfg.cross_column_depth
                              << "] reachable states: processed " << processed_states
                              << ", discovered " << discovered_states
                              << ", frontier " << frontier_size << '\n';
                };

            reachable_states_cache.emplace(
                key,
                std::make_shared<im::ReachableStates>(
                    lattice,
                    logging_enabled ? 100 : 0,
                    logging_enabled
                        ? im::ReachableStates::build_progress_callback_type(log_reachable_states_progress)
                        : im::ReachableStates::build_progress_callback_type{}
                ));
        }

        const auto temp_results_dir = make_temp_results_dir();
        const bool parallelize_single_run_by_island =
            runs.size() == 1 && runs.front().m == 0.0;

        #pragma omp parallel for schedule(dynamic) if(!parallelize_single_run_by_island)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(runs.size()); ++i) {
            const auto run_index = static_cast<std::size_t>(i);
            const auto& cfg = runs[run_index];
            const auto equilibrium_path =
                run_fragment_path(temp_results_dir, "equilibrium", run_index);
            const auto trait_path =
                run_fragment_path(temp_results_dir, "trait_equilibrium", run_index);
            const auto bookkeeping_path =
                run_fragment_path(temp_results_dir, "time_bookkeeping", run_index);

            try {
                const im::Lattice lattice(cfg.columns, cfg.layers, cfg.cross_column_depth);
                const LatticeConfigKey lattice_key{
                    .columns = cfg.columns,
                    .layers = cfg.layers,
                    .cross_column_depth = cfg.cross_column_depth
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
                    .run_id = cfg.run_id,
                    .seed = cfg.seed,
                    .columns = cfg.columns,
                    .layers = cfg.layers,
                    .cross_column_depth = cfg.cross_column_depth,
                    .island_count = cfg.island_count,
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

                auto out = open_output_file(equilibrium_path);
                im::EquilibriumCsvWriter::write_rows(
                    out,
                    meta,
                    eq.state,
                    *states,
                    payoff,
                    1e-6
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
                    auto out = open_output_file(equilibrium_path);
                    write_equilibrium_failure_row(out, cfg);

                    auto trait_out = open_output_file(trait_path);
                    write_trait_failure_row(trait_out, cfg);

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

        std::ofstream file("../equilibrium_distribution.csv");
        if (!file) {
            throw std::runtime_error("Failed to open equilibrium_distribution.csv");
        }

        im::EquilibriumCsvWriter::write_header(file);
        for (std::size_t i = 0; i < runs.size(); ++i) {
            append_file_contents(file, run_fragment_path(temp_results_dir, "equilibrium", i));
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
