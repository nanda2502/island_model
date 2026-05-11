#pragma once

#include <cstddef>
#include <functional>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../model/lattice.hpp"
#include "../model/prerequisites.hpp"
#include "repertoire.hpp"

namespace island_model {

class ReachableStates {
public:
    using lattice_type = Lattice;
    using prerequisites_type = Prerequisites;
    using repertoire_type = Repertoire;
    using repertoire_hash_type = RepertoireHash;
    using build_progress_callback_type =
        std::function<void(std::size_t processed_states,
                           std::size_t discovered_states,
                           std::size_t frontier_size)>;

    struct StateRecord {
        repertoire_type repertoire{};
        std::vector<TraitId> present_traits{};
        std::vector<TraitId> accessible_traits{};
        std::vector<StateId> successors{};
    };

    ReachableStates() = default;

    explicit ReachableStates(lattice_type lattice, double strictness = 1.0)
        : lattice_(lattice),
          prerequisites_(lattice_, strictness) {
        build();
    }

    ReachableStates(lattice_type lattice,
                    double strictness,
                    std::size_t log_interval,
                    const build_progress_callback_type& progress_callback)
        : lattice_(lattice),
          prerequisites_(lattice_, strictness) {
        build(log_interval, progress_callback);
    }

    [[nodiscard]] const lattice_type& lattice() const noexcept {
        return lattice_;
    }

    [[nodiscard]] double strictness() const noexcept {
        return prerequisites_.strictness();
    }

    [[nodiscard]] static StateId empty_state()  noexcept {
        return StateId{0};
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return states_.size();
    }

    [[nodiscard]] const repertoire_type& repertoire(StateId s) const {
        return states_.at(s).repertoire;
    }

    [[nodiscard]] const std::vector<TraitId>& accessible_traits(StateId s) const {
        return states_.at(s).accessible_traits;
    }

    [[nodiscard]] const std::vector<TraitId>& present_traits(StateId s) const {
        return states_.at(s).present_traits;
    }

    [[nodiscard]] const std::vector<StateId>& successors(StateId s) const {
        return states_.at(s).successors;
    }

    [[nodiscard]] StateId successor(StateId s, std::size_t k) const {
        return states_.at(s).successors.at(k);
    }

    [[nodiscard]] bool contains(const repertoire_type& r) const noexcept {
        return index_.contains(r);
    }

    [[nodiscard]] StateId state_id(const repertoire_type& r) const {
        const auto it = index_.find(r);
        if (it == index_.end()) {
            throw std::out_of_range("ReachableStates::state_id: repertoire not found");
        }
        return it->second;
    }

private:
    void build(std::size_t log_interval = 0,
               const build_progress_callback_type& progress_callback = {}) {
        if (lattice_.trait_count() == 0) {
            throw std::invalid_argument("ReachableStates: lattice must be initialized");
        }

        states_.clear();
        index_.clear();

        repertoire_type empty(lattice_.trait_count());
        const StateId root = insert_state(empty);
        if (root != empty_state()) {
            throw std::logic_error("ReachableStates: empty state must have id 0");
        }

        std::queue<StateId> frontier;
        frontier.push(root);
        std::size_t processed_states = 0;

        while (!frontier.empty()) {
            const StateId current = frontier.front();
            frontier.pop();

            const repertoire_type current_repertoire = states_[current].repertoire;
            auto accessible_traits = prerequisites_.accessible_traits(current_repertoire);
            std::vector<StateId> successors;
            successors.reserve(accessible_traits.size());

            for (TraitId trait : accessible_traits) {
                const repertoire_type next_rep = current_repertoire.with_added(trait);
                const auto [next_state, inserted] = find_or_insert_state(next_rep);
                successors.push_back(next_state);

                if (inserted) {
                    frontier.push(next_state);
                }
            }

            states_[current].accessible_traits = std::move(accessible_traits);
            states_[current].successors = std::move(successors);
            ++processed_states;

            if (progress_callback && log_interval != 0 && processed_states % log_interval == 0) {
                progress_callback(processed_states, states_.size(), frontier.size());
            }
        }

        if (progress_callback) {
            progress_callback(processed_states, states_.size(), frontier.size());
        }
    }

    [[nodiscard]] StateId insert_state(const repertoire_type& r) {
        const auto id = static_cast<StateId>(states_.size());
        states_.push_back(StateRecord{
            .repertoire = r,
            .present_traits = enumerate_present_traits(r)
        });
        index_.emplace(r, id);
        return id;
    }

    [[nodiscard]] std::vector<TraitId> enumerate_present_traits(const repertoire_type& r) const {
        std::vector<TraitId> present;
        present.reserve(r.size());

        for (TraitId trait = 0; trait < lattice_.trait_count(); ++trait) {
            if (r.contains(trait)) {
                present.push_back(trait);
            }
        }

        return present;
    }

    [[nodiscard]] std::pair<StateId, bool> find_or_insert_state(const repertoire_type& r) {
        if (const auto it = index_.find(r); it != index_.end()) {
            return {it->second, false};
        }

        const StateId id = insert_state(r);
        return {id, true};
    }

    std::vector<StateRecord> states_{};
    std::unordered_map<repertoire_type, StateId, repertoire_hash_type> index_{};
    lattice_type lattice_{};
    prerequisites_type prerequisites_{};
};

} // namespace island_model
