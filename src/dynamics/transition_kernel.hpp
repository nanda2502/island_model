#pragma once

#include <cstddef>
#include <stdexcept>

#include "../dynamics/weights.hpp"
#include "../state/population_state.hpp"
#include "../state_space/reachable_states.hpp"

namespace island_model {

class TransitionKernel {
public:
    using reachable_states_type = ReachableStates;
    using population_state_type = PopulationState;
    using weight_field_type = WeightField;

    static population_state_type apply(const population_state_type& migrated_state,
                                       const reachable_states_type& states,
                                       const weight_field_type& weights,
                                       double rho,
                                       double mu,
                                       double eta) {
        if (rho < 0.0 || rho > 1.0) {
            throw std::invalid_argument("TransitionKernel::apply: rho must be in [0, 1]");
        }
        if (mu < 0.0 || mu > 1.0) {
            throw std::invalid_argument("TransitionKernel::apply: mu must be in [0, 1]");
        }
        if (eta < 0.0 || eta > 1.0) {
            throw std::invalid_argument("TransitionKernel::apply: eta must be in [0, 1]");
        }
        if (migrated_state.state_count() != states.size()) {
            throw std::invalid_argument("TransitionKernel::apply: population/state space size mismatch");
        }
        if (migrated_state.island_count() != weights.island_count()) {
            throw std::invalid_argument("TransitionKernel::apply: island count mismatch");
        }
        if (states.lattice() != weights.lattice()) {
            throw std::invalid_argument("TransitionKernel::apply: lattice mismatch");
        }

        const std::size_t island_count = migrated_state.island_count();
        const std::size_t state_count = migrated_state.state_count();
        const StateId empty = reachable_states_type::empty_state();

        population_state_type next(island_count, states);
        next.fill(0.0);

        #pragma omp parallel for schedule(static)
        for (std::size_t island_idx = 0; island_idx < island_count; ++island_idx) {
            const auto island = static_cast<IslandId>(island_idx);
            for (StateId state = 0; state < state_count; ++state) {
                const double mass = migrated_state(island, state);
                if (mass == 0.0) {
                    continue;
                }

                const auto& accessible = states.accessible_traits(state);
                const auto& successors = states.successors(state);
                const auto& present_traits = states.present_traits(state);

                if (accessible.size() != successors.size()) {
                    throw std::runtime_error("TransitionKernel::apply: accessible/successor size mismatch");
                }

                const bool is_empty_state = (state == empty);

                double self_prob = is_empty_state ? rho : 0.0;
                if (!is_empty_state) {
                    next(island, empty) += mass * rho;
                }

                const double no_reset = 1.0 - rho;

                if (!accessible.empty()) {
                    const double innovation_prob_per_trait =
                        no_reset * mu / static_cast<double>(accessible.size());

                    for (std::size_t k = 0; k < accessible.size(); ++k) {
                        next(island, successors[k]) += mass * innovation_prob_per_trait;
                    }
                } else {
                    self_prob += no_reset * mu;
                }

                double occupied_social_mass = 0.0;
                for (TraitId trait : present_traits) {
                    occupied_social_mass += weights(island, trait);
                }
                const double available_social_mass = 1.0 - occupied_social_mass;

                double accessible_social_mass = 0.0;
                const double social_scale = no_reset * (1.0 - mu);
                double inaccessible_social_mass = available_social_mass;
                if (available_social_mass > 0.0) {
                    for (TraitId trait : accessible) {
                        inaccessible_social_mass -= weights(island, trait);
                    }
                    if (inaccessible_social_mass < 0.0) {
                        inaccessible_social_mass = 0.0;
                    }

                    const double effective_available_social_mass =
                        available_social_mass - eta * inaccessible_social_mass;

                    if (effective_available_social_mass > 0.0) {
                        for (std::size_t k = 0; k < accessible.size(); ++k) {
                            const TraitId trait = accessible[k];
                            const StateId successor = successors[k];
                            const double renormalized_weight =
                                weights(island, trait) / effective_available_social_mass;
                            accessible_social_mass += renormalized_weight;
                            next(island, successor) += mass * social_scale * renormalized_weight;
                        }
                    }
                }

                if (accessible_social_mass > 1.0) {
                    accessible_social_mass = 1.0;
                }

                if (available_social_mass <= 0.0) {
                    accessible_social_mass = 0.0;
                }

                self_prob += social_scale * (1.0 - accessible_social_mass);

                next(island, state) += mass * self_prob;
            }
        }

        return next;
    }
};

} // namespace island_model
