#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
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

int main(int argc, char* argv[]) {
    try {
        const bool logging_enabled =
            argc > 1 && std::string(argv[1]) == "1";

        const auto grid = im::make_test_grid();
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

        std::vector<std::string> csv_rows(runs.size());
        std::vector<std::string> trait_csv_rows(runs.size());
        std::vector<std::string> bookkeeping_rows(runs.size());
        const bool parallelize_single_run_by_island =
            runs.size() == 1 && runs.front().m == 0.0;

        #pragma omp parallel for schedule(dynamic) if(!parallelize_single_run_by_island)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(runs.size()); ++i) {
            const auto& cfg = runs[static_cast<std::size_t>(i)];

            std::ostringstream out;

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
                    .gamma = cfg.gamma
                    ,
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

                im::EquilibriumCsvWriter::write_rows(
                    out,
                    meta,
                    eq.state,
                    *states,
                    payoff,
                    1e-6
                );

                std::ostringstream trait_out;
                im::TraitEquilibriumCsvWriter::write_rows(
                    trait_out,
                    meta,
                    eq.state,
                    *states
                );
                trait_csv_rows[static_cast<std::size_t>(i)] = trait_out.str();

                std::ostringstream bookkeeping_out;
                im::TimeBookkeepingCsvWriter::write_rows(
                    bookkeeping_out,
                    meta,
                    eq.bookkeeping
                );
                bookkeeping_rows[static_cast<std::size_t>(i)] = bookkeeping_out.str();
            } catch (const std::exception& e) {
                std::ostringstream err;
                err << cfg.run_id << ','
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

                out.str({});
                out.clear();
                out << err.str();

                std::ostringstream trait_err;
                trait_err << cfg.run_id << ','
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
                trait_csv_rows[static_cast<std::size_t>(i)] = trait_err.str();

                #pragma omp critical
                {
                    std::cerr << "Run " << cfg.run_id << " failed: " << e.what() << '\n';
                }
            }

            csv_rows[static_cast<std::size_t>(i)] = out.str();
        }

        std::ofstream file("../equilibrium_distribution.csv");
        if (!file) {
            throw std::runtime_error("Failed to open equilibrium_distribution.csv");
        }

        im::EquilibriumCsvWriter::write_header(file);
        for (const auto& rows : csv_rows) {
            file << rows;
        }

        std::ofstream trait_file("../equilibrium_trait_distribution.csv");
        if (!trait_file) {
            throw std::runtime_error("Failed to open equilibrium_trait_distribution.csv");
        }

        im::TraitEquilibriumCsvWriter::write_header(trait_file);
        for (const auto& rows : trait_csv_rows) {
            trait_file << rows;
        }

        std::ofstream bookkeeping_file("../time_bookkeeping.csv");
        if (!bookkeeping_file) {
            throw std::runtime_error("Failed to open time_bookkeeping.csv");
        }

        im::TimeBookkeepingCsvWriter::write_header(bookkeeping_file);
        for (const auto& rows : bookkeeping_rows) {
            bookkeeping_file << rows;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }
}
