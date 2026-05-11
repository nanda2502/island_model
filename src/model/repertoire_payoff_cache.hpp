#pragma once

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
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
                          double beta,
                          double lambda = 0.0)
        : island_count_(payoff.island_count()),
          state_count_(states.size()),
          states_(&states),
          payoff_(&payoff),
          beta_(beta),
          cache_payoff_sums_(lambda != 0.0),
          dense_(dense_cache_enabled()) {
        if (beta < 0.0) {
            throw std::invalid_argument("RepertoirePayoffCache: beta must be >= 0");
        }
        if (lambda < 0.0) {
            throw std::invalid_argument("RepertoirePayoffCache: lambda must be >= 0");
        }
        if (states.lattice() != payoff.lattice()) {
            throw std::invalid_argument("RepertoirePayoffCache: lattice mismatch");
        }

        if (dense_) {
            if (cache_payoff_sums_) {
                payoff_sums_.assign(island_count_ * state_count_, 0.0);
            }
            expression_bias_denominators_.assign(island_count_ * state_count_, 0.0);
            build(states, payoff, beta);
        }
    }

    [[nodiscard]] std::size_t island_count() const noexcept {
        return island_count_;
    }

    [[nodiscard]] std::size_t state_count() const noexcept {
        return state_count_;
    }

    [[nodiscard]] double payoff_sum(IslandId island, StateId state) const {
        check_indices(island, state);
        if (!cache_payoff_sums_) {
            return compute_payoff_sum(island, state);
        }
        return payoff_sums_[index(island, state)];
    }

    [[nodiscard]] double expression_bias_denominator(IslandId island, StateId state) const {
        check_indices(island, state);
        if (!dense_) {
            return compute_expression_bias_denominator(island, state);
        }
        return expression_bias_denominators_[index(island, state)];
    }

    [[nodiscard]] bool dense() const noexcept {
        return dense_;
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
                if (cache_payoff_sums_) {
                    payoff_sums_[flat_index] = payoff_sum;
                }
                expression_bias_denominators_[flat_index] = expression_bias_denominator;
            }
        }
    }

    [[nodiscard]] double compute_payoff_sum(IslandId island, StateId state) const {
        double out = 0.0;
        for (TraitId trait : states_->present_traits(state)) {
            out += (*payoff_)(island, trait);
        }
        return out;
    }

    [[nodiscard]] double compute_expression_bias_denominator(IslandId island, StateId state) const {
        double out = 0.0;
        for (TraitId trait : states_->present_traits(state)) {
            out += std::pow((*payoff_)(island, trait), beta_);
        }
        return out;
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

    [[nodiscard]] static bool dense_cache_enabled() {
        const char* value = std::getenv("ISLAND_MODEL_DENSE_PAYOFF_CACHE");
        if (value == nullptr) {
            return true;
        }

        const std::string flag(value);
        return flag != "0" && flag != "false" && flag != "FALSE"
            && flag != "off" && flag != "OFF";
    }

    std::size_t island_count_{0};
    std::size_t state_count_{0};
    const reachable_states_type* states_{nullptr};
    const payoff_landscape_type* payoff_{nullptr};
    double beta_{0.0};
    bool cache_payoff_sums_{false};
    bool dense_{true};
    std::vector<double> payoff_sums_{};
    std::vector<double> expression_bias_denominators_{};
};

} // namespace island_model
