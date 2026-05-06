cmake_minimum_required(VERSION 3.26)
project(island_model)

enable_testing()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "C:/msys64/mingw64")

file(GLOB_RECURSE ISLAND_MODEL_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/*.hpp"
)

add_library(island_model_headers INTERFACE)
target_include_directories(island_model_headers INTERFACE "${CMAKE_SOURCE_DIR}/src")
target_compile_options(island_model_headers INTERFACE -Wall -Wextra -O2 -fopenmp)
target_link_options(island_model_headers INTERFACE -fopenmp)
target_sources(island_model_headers INTERFACE ${ISLAND_MODEL_HEADERS})

add_executable(island_model "${CMAKE_SOURCE_DIR}/src/main.cpp")
target_link_libraries(island_model PRIVATE island_model_headers)

set(HEADER_SMOKE_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/header_smoke.cpp")
file(WRITE "${HEADER_SMOKE_SOURCE}" [=[
#include "analysis/convergence.hpp"
#include "analysis/metrics.hpp"
#include "analysis/time_bookkeeping.hpp"
#include "config/default_grid.hpp"
#include "config/parameter_grid.hpp"
#include "dynamics/migration.hpp"
#include "dynamics/simulator.hpp"
#include "dynamics/transition_kernel.hpp"
#include "dynamics/visibility.hpp"
#include "dynamics/weights.hpp"
#include "io/write_equilibrium_csv.hpp"
#include "io/write_time_bookkeeping_csv.hpp"
#include "model/lattice.hpp"
#include "model/payoff_landscape.hpp"
#include "model/prerequisites.hpp"
#include "state/population_state.hpp"
#include "state_space/reachable_states.hpp"
#include "state_space/repertoire.hpp"

#include <random>
#include <sstream>

int main() {
    namespace im = island_model;

    im::Lattice lattice(3, 2);
    im::ReachableStates states(lattice);
    std::mt19937_64 rng(123);
    im::PayoffParams payoff_params{.delta = 1.0, .sigma_b = 0.5, .sigma_nu = 0.1, .k = 0.25};
    im::PayoffLandscape payoff(lattice, 2, payoff_params, rng);
    im::DynamicsParams dynamics{.m = 0.1, .rho = 0.05, .mu = 0.2, .alpha = 1.0, .beta = 1.0, .lambda = 0.0, .gamma = 1.0, .eta = 0.0};
    im::Simulator simulator(states, payoff, dynamics);

    auto initial = simulator.initial_state_all_empty();
    auto next = simulator.step_state_only(initial);
    auto summary = im::RepertoireSummaries::summarize(states.empty_state(), 0, states, payoff);
    std::ostringstream out;
    im::EquilibriumCsvWriter::write_header(out);

    const im::EquilibriumRunMetadata meta{
        .run_id = 1,
        .seed = 123,
        .columns = lattice.columns(),
        .layers = lattice.layers(),
        .cross_column_depth = lattice.cross_column_depth(),
        .island_count = payoff.island_count(),
        .m = dynamics.m,
        .rho = dynamics.rho,
        .mu = dynamics.mu,
        .alpha = dynamics.alpha,
        .beta = dynamics.beta,
        .lambda = dynamics.lambda,
        .gamma = dynamics.gamma,
        .eta = dynamics.eta,
        .delta = payoff_params.delta,
        .sigma_b = payoff_params.sigma_b,
        .sigma_nu = payoff_params.sigma_nu,
        .k = payoff_params.k,
        .converged = 0,
        .steps_to_equilibrium = 0,
        .final_distance = im::Convergence::max_island_l1(initial, next)
    };

    im::EquilibriumCsvWriter::write_rows(out, meta, next, states, payoff);

    return summary.repertoire_size > lattice.trait_count() ? 1 : 0;
}
]=])

add_executable(island_model_header_smoke "${HEADER_SMOKE_SOURCE}")
target_link_libraries(island_model_header_smoke PRIVATE island_model_headers)

add_executable(
    divergence_test
    "${CMAKE_SOURCE_DIR}/src/analysis/divergence_test.cpp"
)
target_link_libraries(divergence_test PRIVATE island_model_headers)
add_test(NAME divergence_test COMMAND divergence_test)
