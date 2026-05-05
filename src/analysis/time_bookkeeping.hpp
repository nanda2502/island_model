#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "../model/payoff_landscape.hpp"
#include "../state/population_state.hpp"
#include "../state_space/reachable_states.hpp"
#include "metrics.hpp"

namespace island_model {

struct PopulationBookkeepingSnapshot {
    std::size_t step{0};
    double f_rep{0.0};
    double within_distance{0.0};
    double total_distance{0.0};
    double mean_payoff{0.0};
    double adj_payoff{0.0};
    double mean_max_depth{0.0};
    double mean_depth{0.0};
    double eff_column{0.0};
    double top_col_mass{0.0};
    double mean_rep_size{0.0};
    double empty_rep_size{0.0};
};

class PopulationBookkeeping {
public:
    using reachable_states_type = ReachableStates;
    using population_state_type = PopulationState;
    using payoff_landscape_type = PayoffLandscape;

    static PopulationBookkeepingSnapshot summarize(std::size_t step,
                                                   const population_state_type& population,
                                                   const reachable_states_type& states,
                                                   const payoff_landscape_type& payoff) {
        validate(population, states, payoff);

        const auto& lattice = states.lattice();
        const double total_population_mass =
            static_cast<double>(population.island_count());

        PopulationBookkeepingSnapshot out{.step = step};
        if (total_population_mass <= 0.0) {
            return out;
        }

        const auto differentiation =
            DifferentiationMetrics::repertoire_differentiation(population, states);
        out.f_rep = differentiation.f_rep;
        out.within_distance = differentiation.within_distance;
        out.total_distance = differentiation.total_distance;

        std::vector<double> column_mass(lattice.columns(), 0.0);
        double total_column_mass = 0.0;

        for (IslandId island = 0; island < population.island_count(); ++island) {
            for (StateId state_id = 0; state_id < population.state_count(); ++state_id) {
                const double frequency = population(island, state_id);
                if (frequency <= 0.0) {
                    continue;
                }

                const auto summary =
                    RepertoireSummaries::summarize(state_id, island, states, payoff);

                out.mean_payoff += frequency * summary.repertoire_payoff_sum;
                out.adj_payoff += frequency * summary.repertoire_payoff_mean;
                out.mean_max_depth += frequency * static_cast<double>(summary.max_layer);
                out.mean_depth += frequency * summary.mean_layer;
                out.mean_rep_size += frequency * static_cast<double>(summary.repertoire_size);

                if (summary.repertoire_size == 0) {
                    out.empty_rep_size += frequency;
                }

                const auto& repertoire = states.repertoire(state_id);
                for (TraitId trait = 0; trait < lattice.trait_count(); ++trait) {
                    if (!repertoire.contains(trait)) {
                        continue;
                    }

                    const auto column =
                        static_cast<std::size_t>(lattice.pos(trait).column);
                    column_mass[column] += frequency;
                    total_column_mass += frequency;
                }
            }
        }

        out.mean_payoff /= total_population_mass;
        out.adj_payoff /= total_population_mass;
        out.mean_max_depth /= total_population_mass;
        out.mean_depth /= total_population_mass;
        out.mean_rep_size /= total_population_mass;
        out.empty_rep_size /= total_population_mass;

        if (total_column_mass <= 0.0) {
            return out;
        }

        double entropy = 0.0;
        for (const double mass : column_mass) {
            if (mass <= 0.0) {
                continue;
            }

            const double proportion = mass / total_column_mass;
            out.top_col_mass = std::max(out.top_col_mass, proportion);
            entropy -= proportion * std::log(proportion);
        }

        out.eff_column = std::exp(entropy);
        return out;
    }

private:
    static void validate(const population_state_type& population,
                         const reachable_states_type& states,
                         const payoff_landscape_type& payoff) {
        if (population.state_count() != states.size()) {
            throw std::invalid_argument("PopulationBookkeeping: population/state-space size mismatch");
        }
        if (population.island_count() != payoff.island_count()) {
            throw std::invalid_argument("PopulationBookkeeping: population/payoff island count mismatch");
        }
        if (states.lattice() != payoff.lattice()) {
            throw std::invalid_argument("PopulationBookkeeping: lattice mismatch");
        }
    }
};

} // namespace island_model
