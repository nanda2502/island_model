#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "../state/population_state.hpp"

namespace island_model {

class Migration {
public:
    using population_state_type = PopulationState;

    static population_state_type apply(const population_state_type& state, double m) {
        if (m < 0.0 || m > 1.0) {
            throw std::invalid_argument("Migration::apply: migration rate m must be in [0, 1]");
        }

        if (m == 0.0) {
            return state;
        }

        population_state_type out = state;

        const std::size_t island_count = state.island_count();
        const std::size_t state_count = state.state_count();

        std::vector<double> mean(state_count, 0.0);

        for (IslandId i = 0; i < island_count; ++i) {
            for (StateId s = 0; s < state_count; ++s) {
                mean[s] += state(i, s);
            }
        }

        const double inv_island_count = 1.0 / static_cast<double>(island_count);
        for (StateId s = 0; s < state_count; ++s) {
            mean[s] *= inv_island_count;
        }

        for (IslandId i = 0; i < island_count; ++i) {
            for (StateId s = 0; s < state_count; ++s) {
                out(i, s) = (1.0 - m) * state(i, s) + m * mean[s];
            }
        }

        return out;
    }
};

} // namespace island_model
