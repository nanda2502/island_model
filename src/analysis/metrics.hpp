#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
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

struct RepertoireDifferentiationSummary {
    double within_distance{0.0};
    double total_distance{0.0};
    double f_rep{0.0};
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

    static double repertoire_jaccard_distance(const Repertoire& a,
                                              const Repertoire& b) {
        if (a.trait_count() != b.trait_count()) {
            throw std::invalid_argument("DifferentiationMetrics::repertoire_jaccard_distance: trait-count mismatch");
        }

        const auto& a_words = a.words();
        const auto& b_words = b.words();
        if (a_words.size() != b_words.size()) {
            throw std::invalid_argument("DifferentiationMetrics::repertoire_jaccard_distance: bitset word-count mismatch");
        }

        std::size_t intersection_size = 0;
        std::size_t union_size = 0;

        for (std::size_t i = 0; i < a_words.size(); ++i) {
            const std::uint64_t aw = a_words[i];
            const std::uint64_t bw = b_words[i];
            intersection_size += static_cast<std::size_t>(std::popcount(aw & bw));
            union_size += static_cast<std::size_t>(std::popcount(aw | bw));
        }

        if (union_size == 0) {
            return 0.0;
        }

        return 1.0 - (static_cast<double>(intersection_size) / static_cast<double>(union_size));
    }

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

    static RepertoireDifferentiationSummary repertoire_differentiation(
        const population_state_type& population,
        const reachable_states_type& states) {
        if (population.state_count() != states.size()) {
            throw std::invalid_argument("DifferentiationMetrics::repertoire_differentiation: population/state-space size mismatch");
        }

        const std::size_t island_count = population.island_count();
        const std::size_t state_count = population.state_count();
        if (island_count == 0 || state_count == 0) {
            return {};
        }

        const auto distance_cache = build_repertoire_distance_cache(states);

        RepertoireDifferentiationSummary out{};
        for (IslandId island = 0; island < island_count; ++island) {
            std::vector<double> island_distribution(state_count, 0.0);
            for (StateId state = 0; state < state_count; ++state) {
                island_distribution[static_cast<std::size_t>(state)] = population(island, state);
            }
            out.within_distance += weighted_average_distance(
                island_distribution,
                distance_cache,
                state_count);
        }
        out.within_distance /= static_cast<double>(island_count);

        std::vector<double> metapop_distribution(state_count, 0.0);
        for (StateId state = 0; state < state_count; ++state) {
            double avg = 0.0;
            for (IslandId island = 0; island < island_count; ++island) {
                avg += population(island, state);
            }
            metapop_distribution[static_cast<std::size_t>(state)] =
                avg / static_cast<double>(island_count);
        }

        out.total_distance = weighted_average_distance(
            metapop_distribution,
            distance_cache,
            state_count);

        if (out.total_distance <= 0.0) {
            out.f_rep = 0.0;
            return out;
        }

        out.f_rep = 1.0 - (out.within_distance / out.total_distance);
        constexpr double tolerance = 1e-12;
        if (out.f_rep < 0.0 && std::abs(out.f_rep) <= tolerance) {
            out.f_rep = 0.0;
        }
        if (out.f_rep > 1.0 && std::abs(out.f_rep - 1.0) <= tolerance) {
            out.f_rep = 1.0;
        }
        out.f_rep = std::clamp(out.f_rep, 0.0, 1.0);

        return out;
    }

private:
    static std::vector<double> build_repertoire_distance_cache(const reachable_states_type& states) {
        const std::size_t state_count = states.size();
        std::vector<double> cache(state_count * state_count, 0.0);

        for (StateId r1 = 0; r1 < state_count; ++r1) {
            cache[static_cast<std::size_t>(r1) * state_count + static_cast<std::size_t>(r1)] = 0.0;
            for (StateId r2 = r1 + 1; r2 < state_count; ++r2) {
                const double dist = repertoire_jaccard_distance(
                    states.repertoire(r1),
                    states.repertoire(r2));
                cache[static_cast<std::size_t>(r1) * state_count + static_cast<std::size_t>(r2)] = dist;
                cache[static_cast<std::size_t>(r2) * state_count + static_cast<std::size_t>(r1)] = dist;
            }
        }

        return cache;
    }

    static double weighted_average_distance(const std::vector<double>& distribution,
                                            const std::vector<double>& distance_cache,
                                            std::size_t state_count) {
        double out = 0.0;
        for (StateId r1 = 0; r1 < state_count; ++r1) {
            const double p1 = distribution[static_cast<std::size_t>(r1)];
            if (p1 <= 0.0) {
                continue;
            }

            for (StateId r2 = 0; r2 < state_count; ++r2) {
                const double p2 = distribution[static_cast<std::size_t>(r2)];
                if (p2 <= 0.0) {
                    continue;
                }

                const double d = distance_cache[
                    static_cast<std::size_t>(r1) * state_count + static_cast<std::size_t>(r2)];
                out += p1 * p2 * d;
            }
        }

        return out;
    }
};

} // namespace island_model
