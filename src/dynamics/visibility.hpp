#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "../model/payoff_landscape.hpp"
#include "../state/population_state.hpp"
#include "../state_space/reachable_states.hpp"

namespace island_model {

class VisibilityField {
public:
    using lattice_type = Lattice;

    VisibilityField() = default;

    VisibilityField(lattice_type lattice, std::size_t island_count)
        : lattice_(lattice),
          island_count_(island_count),
          values_(island_count_ * lattice_.trait_count(), 0.0) {}

    [[nodiscard]] const lattice_type& lattice() const noexcept {
        return lattice_;
    }

    [[nodiscard]] std::size_t island_count() const noexcept {
        return island_count_;
    }

    [[nodiscard]] std::size_t trait_count() const noexcept {
        return lattice_.trait_count();
    }

    [[nodiscard]] double operator()(IslandId island, TraitId trait) const {
        check_island(island);
        check_trait(trait);
        return values_[index(island, trait)];
    }

    [[nodiscard]] double& operator()(IslandId island, TraitId trait) {
        check_island(island);
        check_trait(trait);
        return values_[index(island, trait)];
    }

    void fill(double value) {
        std::fill(values_.begin(), values_.end(), value);
    }

    [[nodiscard]] const std::vector<double>& data() const noexcept {
        return values_;
    }

private:
    [[nodiscard]] std::size_t index(IslandId island, TraitId trait) const noexcept {
        return static_cast<std::size_t>(island) * lattice_.trait_count()
             + static_cast<std::size_t>(trait);
    }

    void check_island(IslandId island) const {
        if (static_cast<std::size_t>(island) >= island_count_) {
            throw std::out_of_range("VisibilityField: invalid island id");
        }
    }

    void check_trait(TraitId trait) const {
        if (!lattice_.valid_trait(trait)) {
            throw std::out_of_range("VisibilityField: invalid trait id");
        }
    }

    lattice_type lattice_{};
    std::size_t island_count_{0};
    std::vector<double> values_{};
};

class Visibility {
public:
    using lattice_type = Lattice;
    using reachable_states_type = ReachableStates;
    using population_state_type = PopulationState;
    using payoff_landscape_type = PayoffLandscape;
    using visibility_field_type = VisibilityField;

    static visibility_field_type compute(const population_state_type& migrated_state,
                                         const reachable_states_type& states,
                                         const payoff_landscape_type& payoff,
                                         double beta) {
        if (beta < 0.0) {
            throw std::invalid_argument("Visibility::compute: beta must be >= 0");
        }

        if (migrated_state.state_count() != states.size()) {
            throw std::invalid_argument("Visibility::compute: population/state space size mismatch");
        }

        if (migrated_state.island_count() != payoff.island_count()) {
            throw std::invalid_argument("Visibility::compute: island count mismatch");
        }
        if (states.lattice() != payoff.lattice()) {
            throw std::invalid_argument("Visibility::compute: lattice mismatch");
        }

        const std::size_t island_count = migrated_state.island_count();
        const std::size_t state_count = states.size();
        visibility_field_type out(payoff.lattice(), island_count);
        out.fill(0.0);

        std::vector<double> denominators(island_count * state_count, 0.0);

        #pragma omp parallel for schedule(static)
        for (std::size_t island_idx = 0; island_idx < island_count; ++island_idx) {
            const auto island = static_cast<IslandId>(island_idx);
            for (StateId state = 0; state < state_count; ++state) {
                const auto& present_traits = states.present_traits(state);

                double denom = 0.0;
                for (TraitId trait : present_traits) {
                    denom += std::pow(payoff(island, trait), beta);
                }

                denominators[index(island, state, state_count)] = denom;
            }
        }

        #pragma omp parallel for schedule(static)
        for (std::size_t island_idx = 0; island_idx < island_count; ++island_idx) {
            const auto island = static_cast<IslandId>(island_idx);
            for (StateId state = 0; state < state_count; ++state) {
                const double mass = migrated_state(island, state);
                if (mass == 0.0) {
                    continue;
                }

                const double denom = denominators[index(island, state, state_count)];
                if (denom == 0.0) {
                    continue;
                }

                const auto& present_traits = states.present_traits(state);

                for (TraitId trait : present_traits) {
                    const double numer = std::pow(payoff(island, trait), beta);
                    out(island, trait) += mass * (numer / denom);
                }
            }
        }

        return out;
    }

private:
    [[nodiscard]] static std::size_t index(IslandId island,
                                           StateId state,
                                           std::size_t state_count) noexcept {
        return static_cast<std::size_t>(island) * state_count
             + static_cast<std::size_t>(state);
    }
};

} // namespace island_model
