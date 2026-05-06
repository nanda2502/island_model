#include "metrics.hpp"

#include "../model/lattice.hpp"
#include "../model/payoff_landscape.hpp"
#include "../state/population_state.hpp"
#include "../state_space/reachable_states.hpp"

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

namespace {

void expect_true(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

bool approx_equal(double a, double b, double tolerance = 1e-12) {
    return std::abs(a - b) <= tolerance;
}

StateId state_id_for_traits(const island_model::ReachableStates& states,
                            const std::vector<TraitId>& traits) {
    island_model::Repertoire repertoire(states.lattice().trait_count());
    for (const TraitId trait : traits) {
        repertoire.add(trait);
    }
    return states.state_id(repertoire);
}

void set_distribution(island_model::PopulationState& population,
                      IslandId island,
                      const std::vector<std::pair<StateId, double>>& masses) {
    for (const auto& [state, mass] : masses) {
        population(island, state) = mass;
    }
    population.normalize_island(island);
}

void test_identical_islands_have_zero_divergence() {
    namespace im = island_model;

    const im::Lattice lattice(2, 1);
    const im::ReachableStates states(lattice);
    im::PopulationState population(2, states);

    const auto trait0 = state_id_for_traits(states, {TraitId{0}});
    set_distribution(population, IslandId{0}, {{trait0, 1.0}});
    set_distribution(population, IslandId{1}, {{trait0, 1.0}});

    std::mt19937_64 rng(123);
    const im::PayoffLandscape payoff(
        lattice,
        2,
        im::PayoffParams{.delta = 1.0, .sigma_b = 0.3, .sigma_nu = 0.2, .k = 0.4},
        rng);

    const auto summary =
        im::DifferentiationMetrics::adaptive_divergence(population, states, payoff);

    expect_true(summary.pair_count == 1, "identical islands should produce one pair");
    expect_true(approx_equal(summary.mean_distance, 0.0),
                "identical islands should have zero adaptive divergence");
}

void test_disjoint_profiles_have_unit_divergence() {
    namespace im = island_model;

    const im::Lattice lattice(2, 1);
    const im::ReachableStates states(lattice);
    im::PopulationState population(2, states);

    const auto trait0 = state_id_for_traits(states, {TraitId{0}});
    const auto trait1 = state_id_for_traits(states, {TraitId{1}});
    set_distribution(population, IslandId{0}, {{trait0, 1.0}});
    set_distribution(population, IslandId{1}, {{trait1, 1.0}});

    std::mt19937_64 rng(456);
    const im::PayoffLandscape payoff(
        lattice,
        2,
        im::PayoffParams{.delta = 1.0, .sigma_b = 0.5, .sigma_nu = 0.1, .k = 0.2},
        rng);

    const auto summary =
        im::DifferentiationMetrics::adaptive_divergence(population, states, payoff);

    expect_true(summary.pair_count == 1, "disjoint islands should produce one pair");
    expect_true(approx_equal(summary.mean_distance, 1.0),
                "disjoint trait profiles should have unit adaptive divergence");
}

void test_payoff_weighted_jaccard_matches_manual_value() {
    namespace im = island_model;

    const im::Lattice lattice(2, 1);
    const im::ReachableStates states(lattice);
    im::PopulationState population(2, states);

    const auto trait0 = state_id_for_traits(states, {TraitId{0}});
    const auto trait1 = state_id_for_traits(states, {TraitId{1}});
    set_distribution(population, IslandId{0}, {{trait0, 1.0}});
    set_distribution(population, IslandId{1}, {{trait0, 0.5}, {trait1, 0.5}});

    std::mt19937_64 rng(789);
    const im::PayoffLandscape payoff(
        lattice,
        2,
        im::PayoffParams{.delta = 1.0, .sigma_b = 0.7, .sigma_nu = 0.1, .k = 0.6},
        rng);

    const auto summary =
        im::DifferentiationMetrics::adaptive_divergence(population, states, payoff);

    const double w0 = 0.5 * (payoff(IslandId{0}, TraitId{0}) + payoff(IslandId{1}, TraitId{0}));
    const double w1 = 0.5 * (payoff(IslandId{0}, TraitId{1}) + payoff(IslandId{1}, TraitId{1}));
    const double expected =
        1.0 - ((w0 * 0.5 + w1 * 0.0) / (w0 * 1.0 + w1 * 0.5));

    expect_true(summary.pair_count == 1, "manual-check population should produce one pair");
    expect_true(approx_equal(summary.mean_distance, expected),
                "adaptive divergence should match the payoff-weighted Jaccard formula");
}

void test_cultural_jaccard_matches_manual_value() {
    namespace im = island_model;

    const im::Lattice lattice(2, 1);
    const im::ReachableStates states(lattice);
    im::PopulationState population(2, states);

    const auto trait0 = state_id_for_traits(states, {TraitId{0}});
    const auto trait1 = state_id_for_traits(states, {TraitId{1}});
    set_distribution(population, IslandId{0}, {{trait0, 1.0}});
    set_distribution(population, IslandId{1}, {{trait0, 0.5}, {trait1, 0.5}});

    const auto summary =
        im::DifferentiationMetrics::cultural_divergence(population, states);

    const double expected = 1.0 - ((0.5 + 0.0) / (1.0 + 0.5));

    expect_true(summary.pair_count == 1, "manual-check population should produce one pair");
    expect_true(approx_equal(summary.mean_distance, expected),
                "cultural divergence should match the unweighted Jaccard formula");
}

void test_all_empty_profiles_return_zero_distance() {
    namespace im = island_model;

    const im::Lattice lattice(2, 1);
    const im::ReachableStates states(lattice);
    im::PopulationState population(2, states);
    population.set_all_empty(states);

    std::mt19937_64 rng(321);
    const im::PayoffLandscape payoff(
        lattice,
        2,
        im::PayoffParams{.delta = 1.0, .sigma_b = 0.2, .sigma_nu = 0.2, .k = 0.5},
        rng);

    const auto summary =
        im::DifferentiationMetrics::adaptive_divergence(population, states, payoff);

    expect_true(summary.pair_count == 1, "all-empty population should still produce one pair");
    expect_true(approx_equal(summary.mean_distance, 0.0),
                "all-empty profiles should return zero divergence");
}

} // namespace

int main() {
    try {
        test_identical_islands_have_zero_divergence();
        test_disjoint_profiles_have_unit_divergence();
        test_payoff_weighted_jaccard_matches_manual_value();
        test_cultural_jaccard_matches_manual_value();
        test_all_empty_profiles_return_zero_distance();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }

    std::cout << "All divergence tests passed.\n";
    return 0;
}
