#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "../state_space/reachable_states.hpp"

namespace island_model {

class PopulationState {
public:
    using reachable_states_type = ReachableStates;

    PopulationState() = default;

    explicit PopulationState(std::size_t island_count, const reachable_states_type& states)
        : island_count_(island_count),
          state_count_(states.size()),
          probabilities_(island_count_ * state_count_, 0.0) {}

    [[nodiscard]] std::size_t island_count() const noexcept {
        return island_count_;
    }

    [[nodiscard]] std::size_t state_count() const noexcept {
        return state_count_;
    }

    [[nodiscard]] double operator()(IslandId island, StateId state) const {
        check_island(island);
        check_state(state);
        return probabilities_[index(island, state)];
    }

    [[nodiscard]] double& operator()(IslandId island, StateId state) {
        check_island(island);
        check_state(state);
        return probabilities_[index(island, state)];
    }

    void fill(double value) {
        std::fill(probabilities_.begin(), probabilities_.end(), value);
    }

    void set_all_empty(const reachable_states_type& states) {
        if (state_count_ != states.size()) {
            throw std::invalid_argument("PopulationState::set_all_empty: state space size mismatch");
        }

        fill(0.0);

        const StateId empty = reachable_states_type::empty_state();
        for (IslandId i = 0; i < island_count_; ++i) {
            (*this)(i, empty) = 1.0;
        }
    }

    [[nodiscard]] std::vector<double> island_distribution(IslandId island) const {
        check_island(island);
        const auto begin = probabilities_.begin() + static_cast<std::ptrdiff_t>(island * state_count_);
        const auto end = begin + static_cast<std::ptrdiff_t>(state_count_);
        return std::vector<double>(begin, end);
    }

    [[nodiscard]] const std::vector<double>& data() const noexcept {
        return probabilities_;
    }

    [[nodiscard]] std::vector<double>& data() noexcept {
        return probabilities_;
    }

    void normalize_island(IslandId island) {
        check_island(island);

        double sum = 0.0;
        for (StateId s = 0; s < state_count_; ++s) {
            sum += (*this)(island, s);
        }

        if (sum <= 0.0) {
            throw std::runtime_error("PopulationState::normalize_island: non-positive mass");
        }

        for (StateId s = 0; s < state_count_; ++s) {
            (*this)(island, s) /= sum;
        }
    }

    void normalize_all() {
        for (IslandId i = 0; i < island_count_; ++i) {
            normalize_island(i);
        }
    }

private:
    [[nodiscard]] std::size_t index(IslandId island, StateId state) const noexcept {
        return static_cast<std::size_t>(island) * state_count_ + static_cast<std::size_t>(state);
    }

    void check_island(IslandId island) const {
        if (static_cast<std::size_t>(island) >= island_count_) {
            throw std::out_of_range("PopulationState: invalid island id");
        }
    }

    void check_state(StateId state) const {
        if (static_cast<std::size_t>(state) >= state_count_) {
            throw std::out_of_range("PopulationState: invalid state id");
        }
    }

    std::size_t island_count_{0};
    std::size_t state_count_{0};
    std::vector<double> probabilities_{};
};

} // namespace island_model
