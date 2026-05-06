#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../model/payoff_landscape.hpp"
#include "../state/population_state.hpp"
#include "../state_space/reachable_states.hpp"

namespace island_model {

struct RepertoireSummary {
    std::size_t repertoire_size{0};
    std::size_t max_layer{0};
    double mean_layer{0.0};
    double repertoire_payoff_sum{0.0};
    double repertoire_payoff_mean{0.0};
    double repertoire_payoff_max{0.0};
};

struct TraitFrequencyMatrix {
    std::size_t island_count{0};
    std::size_t trait_count{0};
    std::vector<double> values{};

    [[nodiscard]] double operator()(IslandId island, TraitId trait) const {
        const auto island_idx = static_cast<std::size_t>(island);
        const auto trait_idx = static_cast<std::size_t>(trait);
        if (island_idx >= island_count || trait_idx >= trait_count) {
            throw std::out_of_range("TraitFrequencyMatrix: invalid island or trait id");
        }
        return values[island_idx * trait_count + trait_idx];
    }
};

struct DivergenceSummary {
    double mean_distance{0.0};
    std::size_t pair_count{0};
};

class RepertoireSummaries {
public:
    using reachable_states_type = ReachableStates;
    using payoff_landscape_type = PayoffLandscape;

    static RepertoireSummary summarize(StateId state_id,
                                       IslandId island,
                                       const reachable_states_type& states,
                                       const payoff_landscape_type& payoff) {
        const auto& repertoire = states.repertoire(state_id);
        const auto& lattice = states.lattice();

        RepertoireSummary out{};
        double max_payoff = -std::numeric_limits<double>::infinity();
        double layer_sum = 0.0;

        for (TraitId trait = 0; trait < payoff.trait_count(); ++trait) {
            if (!repertoire.contains(trait)) {
                continue;
            }

            const double value = payoff(island, trait);
            const auto pos = lattice.pos(trait);
            ++out.repertoire_size;
            out.repertoire_payoff_sum += value;
            layer_sum += static_cast<double>(pos.layer);
            out.max_layer = std::max(
                out.max_layer,
                static_cast<std::size_t>(pos.layer));

            if (value > max_payoff) {
                max_payoff = value;
            }
        }

        if (out.repertoire_size > 0) {
            out.mean_layer = layer_sum / static_cast<double>(out.repertoire_size);
            out.repertoire_payoff_mean =
                out.repertoire_payoff_sum / static_cast<double>(out.repertoire_size);
            out.repertoire_payoff_max = max_payoff;
        }

        return out;
    }
};

class DifferentiationMetrics {
public:
    using reachable_states_type = ReachableStates;
    using population_state_type = PopulationState;

    static TraitFrequencyMatrix trait_frequencies(const population_state_type& population,
                                                  const reachable_states_type& states) {
        const std::size_t island_count = population.island_count();
        const std::size_t state_count = population.state_count();
        const std::size_t trait_count = states.lattice().trait_count();

        TraitFrequencyMatrix out{
            .island_count = island_count,
            .trait_count = trait_count,
            .values = std::vector<double>(island_count * trait_count, 0.0)
        };

        if (island_count == 0 || state_count == 0 || trait_count == 0) {
            return out;
        }

        for (IslandId island = 0; island < island_count; ++island) {
            for (StateId state = 0; state < state_count; ++state) {
                const double state_frequency = population(island, state);
                if (state_frequency <= 0.0) {
                    continue;
                }

                for (const TraitId trait : states.present_traits(state)) {
                    out.values[static_cast<std::size_t>(island) * trait_count
                               + static_cast<std::size_t>(trait)] += state_frequency;
                }
            }
        }

        return out;
    }

    static DivergenceSummary adaptive_divergence(
        const population_state_type& population,
        const reachable_states_type& states,
        const PayoffLandscape& payoff) {
        if (population.state_count() != states.size()) {
            throw std::invalid_argument("DifferentiationMetrics::adaptive_divergence: population/state-space size mismatch");
        }
        if (population.island_count() != payoff.island_count()) {
            throw std::invalid_argument("DifferentiationMetrics::adaptive_divergence: population/payoff island count mismatch");
        }
        if (states.lattice() != payoff.lattice()) {
            throw std::invalid_argument("DifferentiationMetrics::adaptive_divergence: lattice mismatch");
        }

        const std::size_t island_count = population.island_count();
        if (island_count < 2) {
            return {};
        }

        const auto frequencies = trait_frequencies(population, states);
        DivergenceSummary out{};

        for (IslandId i = 0; i < island_count; ++i) {
            for (IslandId j = i + 1; j < island_count; ++j) {
                out.mean_distance += payoff_weighted_jaccard_distance(
                    frequencies,
                    i,
                    j,
                    payoff);
                ++out.pair_count;
            }
        }

        if (out.pair_count > 0) {
            out.mean_distance /= static_cast<double>(out.pair_count);
        }

        return out;
    }

    static DivergenceSummary cultural_divergence(
        const population_state_type& population,
        const reachable_states_type& states) {
        if (population.state_count() != states.size()) {
            throw std::invalid_argument("DifferentiationMetrics::cultural_divergence: population/state-space size mismatch");
        }

        const std::size_t island_count = population.island_count();
        if (island_count < 2) {
            return {};
        }

        const auto frequencies = trait_frequencies(population, states);
        DivergenceSummary out{};

        for (IslandId i = 0; i < island_count; ++i) {
            for (IslandId j = i + 1; j < island_count; ++j) {
                out.mean_distance += jaccard_distance(frequencies, i, j);
                ++out.pair_count;
            }
        }

        if (out.pair_count > 0) {
            out.mean_distance /= static_cast<double>(out.pair_count);
        }

        return out;
    }

private:
    static double payoff_weighted_jaccard_distance(const TraitFrequencyMatrix& frequencies,
                                                   IslandId island_a,
                                                   IslandId island_b,
                                                   const PayoffLandscape& payoff) {
        double numerator = 0.0;
        double denominator = 0.0;

        for (TraitId trait = 0; trait < frequencies.trait_count; ++trait) {
            const double q_a = frequencies(island_a, trait);
            const double q_b = frequencies(island_b, trait);
            const double weight = 0.5 * (payoff(island_a, trait) + payoff(island_b, trait));

            numerator += weight * std::min(q_a, q_b);
            denominator += weight * std::max(q_a, q_b);
        }

        if (denominator <= 0.0) {
            return 0.0;
        }

        return 1.0 - (numerator / denominator);
    }

    static double jaccard_distance(const TraitFrequencyMatrix& frequencies,
                                   IslandId island_a,
                                   IslandId island_b) {
        double numerator = 0.0;
        double denominator = 0.0;

        for (TraitId trait = 0; trait < frequencies.trait_count; ++trait) {
            const double q_a = frequencies(island_a, trait);
            const double q_b = frequencies(island_b, trait);

            numerator += std::min(q_a, q_b);
            denominator += std::max(q_a, q_b);
        }

        if (denominator <= 0.0) {
            return 0.0;
        }

        return 1.0 - (numerator / denominator);
    }
};

} // namespace island_model
