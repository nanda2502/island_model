#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "payoff_landscape.hpp"
#include "../state_space/reachable_states.hpp"

namespace island_model {

class RepertoirePayoffCache {
public:
    using payoff_landscape_type = PayoffLandscape;
    using reachable_states_type = ReachableStates;

    RepertoirePayoffCache() = default;

    RepertoirePayoffCache(const reachable_states_type& states,
                          const payoff_landscape_type& payoff,
                          double beta)
        : island_count_(payoff.island_count()),
          state_count_(states.size()),
          payoff_sums_(island_count_ * state_count_, 0.0),
          expression_bias_denominators_(island_count_ * state_count_, 0.0) {
        if (beta < 0.0) {
            throw std::invalid_argument("RepertoirePayoffCache: beta must be >= 0");
        }
        if (states.lattice() != payoff.lattice()) {
            throw std::invalid_argument("RepertoirePayoffCache: lattice mismatch");
        }

        build(states, payoff, beta);
    }

    [[nodiscard]] std::size_t island_count() const noexcept {
        return island_count_;
    }

    [[nodiscard]] std::size_t state_count() const noexcept {
        return state_count_;
    }

    [[nodiscard]] double payoff_sum(IslandId island, StateId state) const {
        check_indices(island, state);
        return payoff_sums_[index(island, state)];
    }

    [[nodiscard]] double expression_bias_denominator(IslandId island, StateId state) const {
        check_indices(island, state);
        return expression_bias_denominators_[index(island, state)];
    }

private:
    void build(const reachable_states_type& states,
               const payoff_landscape_type& payoff,
               double beta) {
        for (IslandId island = 0; island < island_count_; ++island) {
            for (StateId state = 0; state < state_count_; ++state) {
                double payoff_sum = 0.0;
                double expression_bias_denominator = 0.0;

                for (TraitId trait : states.present_traits(state)) {
                    const double trait_payoff = payoff(island, trait);
                    payoff_sum += trait_payoff;
                    expression_bias_denominator += std::pow(trait_payoff, beta);
                }

                const std::size_t flat_index = index(island, state);
                payoff_sums_[flat_index] = payoff_sum;
                expression_bias_denominators_[flat_index] = expression_bias_denominator;
            }
        }
    }

    [[nodiscard]] std::size_t index(IslandId island, StateId state) const noexcept {
        return static_cast<std::size_t>(island) * state_count_
             + static_cast<std::size_t>(state);
    }

    void check_indices(IslandId island, StateId state) const {
        if (static_cast<std::size_t>(island) >= island_count_) {
            throw std::out_of_range("RepertoirePayoffCache: invalid island id");
        }
        if (static_cast<std::size_t>(state) >= state_count_) {
            throw std::out_of_range("RepertoirePayoffCache: invalid state id");
        }
    }

    std::size_t island_count_{0};
    std::size_t state_count_{0};
    std::vector<double> payoff_sums_{};
    std::vector<double> expression_bias_denominators_{};
};

} // namespace island_model
