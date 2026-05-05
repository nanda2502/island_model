#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../state/population_state.hpp"

namespace island_model {

class Convergence {
public:
    using population_state_type = PopulationState;

    [[nodiscard]] static double max_island_l1(const population_state_type& a,
                                              const population_state_type& b) {
        validate_same_shape(a, b);

        double max_distance = 0.0;

        #pragma omp parallel for schedule(static) reduction(max:max_distance)
        for (std::size_t island_idx = 0; island_idx < a.island_count(); ++island_idx) {
            const auto island = static_cast<IslandId>(island_idx);
            double distance = 0.0;
            for (StateId state = 0; state < a.state_count(); ++state) {
                distance += std::abs(a(island, state) - b(island, state));
            }
            max_distance = std::max(max_distance, distance);
        }

        return max_distance;
    }

    [[nodiscard]] static bool is_converged(const population_state_type& a,
                                           const population_state_type& b,
                                           double tolerance) {
        if (tolerance < 0.0) {
            throw std::invalid_argument("Convergence::is_converged: tolerance must be >= 0");
        }

        return max_island_l1(a, b) < tolerance;
    }

private:
    static void validate_same_shape(const population_state_type& a,
                                    const population_state_type& b) {
        if (a.island_count() != b.island_count()) {
            throw std::invalid_argument("Convergence: island count mismatch");
        }
        if (a.state_count() != b.state_count()) {
            throw std::invalid_argument("Convergence: state count mismatch");
        }
    }
};

} // namespace island_model
