#pragma once

#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
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
        std::vector<TraitId> present_traits;
        std::vector<TraitId> accessible_traits;
        std::vector<StateId> successors;
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
        return uses_encoded_states_ ? state_count_limit_ : states_.size();
    }

    [[nodiscard]] repertoire_type repertoire(StateId s) const {
        check_state(s);

        if (encoding_ == Encoding::height) {
            return repertoire_from_height_state(s);
        }
        if (encoding_ == Encoding::loose_bitmask) {
            return repertoire_from_bitmask_state(s);
        }

        return generic_repertoires_.at(s);
    }

    [[nodiscard]] const std::vector<TraitId>& accessible_traits(StateId s) const {
        check_state(s);
        if (encoding_ == Encoding::height) {
            thread_local std::vector<TraitId> accessible;
            accessible = enumerate_accessible_traits_from_height_state(s);
            return accessible;
        }
        if (encoding_ == Encoding::loose_bitmask) {
            thread_local std::vector<TraitId> accessible;
            accessible = enumerate_accessible_traits_from_bitmask_state(s);
            return accessible;
        }

        return states_.at(s).accessible_traits;
    }

    [[nodiscard]] const std::vector<TraitId>& present_traits(StateId s) const {
        check_state(s);
        if (encoding_ == Encoding::height) {
            thread_local std::vector<TraitId> present;
            present = enumerate_present_traits_from_height_state(s);
            return present;
        }
        if (encoding_ == Encoding::loose_bitmask) {
            thread_local std::vector<TraitId> present;
            present = enumerate_present_traits_from_bitmask_state(s);
            return present;
        }

        return states_.at(s).present_traits;
    }

    [[nodiscard]] const std::vector<StateId>& successors(StateId s) const {
        check_state(s);
        if (encoding_ == Encoding::height) {
            thread_local std::vector<StateId> successors;
            successors = enumerate_successors_from_height_state(s);
            return successors;
        }
        if (encoding_ == Encoding::loose_bitmask) {
            thread_local std::vector<StateId> successors;
            successors = enumerate_successors_from_bitmask_state(s);
            return successors;
        }

        return states_.at(s).successors;
    }

    [[nodiscard]] StateId successor(StateId s, std::size_t k) const {
        return successors(s).at(k);
    }

    [[nodiscard]] bool contains(const repertoire_type& r) const noexcept {
        if (encoding_ == Encoding::height) {
            return state_id_for_height_repertoire(r).has_value();
        }
        if (encoding_ == Encoding::loose_bitmask) {
            return state_id_for_bitmask_repertoire(r).has_value();
        }

        return index_.contains(r);
    }

    [[nodiscard]] StateId state_id(const repertoire_type& r) const {
        if (encoding_ == Encoding::height) {
            if (const auto id = state_id_for_height_repertoire(r)) {
                return *id;
            }
            throw std::out_of_range("ReachableStates::state_id: repertoire not found");
        }
        if (encoding_ == Encoding::loose_bitmask) {
            if (const auto id = state_id_for_bitmask_repertoire(r)) {
                return *id;
            }
            throw std::out_of_range("ReachableStates::state_id: repertoire not found");
        }

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
        generic_repertoires_.clear();
        index_.clear();
        height_base_ = lattice_.layers() + 1U;
        state_count_limit_ = 0;
        uses_encoded_states_ = false;
        encoding_ = Encoding::generic;

        if (lattice_.cross_column_depth() == 1 && prerequisites_.strictness() == 1.0) {
            encoding_ = Encoding::height;
            uses_encoded_states_ = true;
            build_height_encoded(log_interval, progress_callback);
            return;
        }
        if (lattice_.cross_column_depth() == 1 && prerequisites_.strictness() == 0.0) {
            encoding_ = Encoding::loose_bitmask;
            uses_encoded_states_ = true;
            build_loose_bitmask_encoded(log_interval, progress_callback);
            return;
        }

        build_generic(log_interval, progress_callback);
    }

    void build_height_encoded(std::size_t log_interval,
                              const build_progress_callback_type& progress_callback) {
        static_cast<void>(log_interval);
        state_count_limit_ = checked_height_state_count();

        if (progress_callback) {
            progress_callback(state_count_limit_, state_count_limit_, 0);
        }
    }

    void build_loose_bitmask_encoded(std::size_t log_interval,
                                     const build_progress_callback_type& progress_callback) {
        static_cast<void>(log_interval);
        state_count_limit_ = checked_bitmask_state_count();

        if (progress_callback) {
            progress_callback(state_count_limit_, state_count_limit_, 0);
        }
    }

    void build_generic(std::size_t log_interval,
                       const build_progress_callback_type& progress_callback) {

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

            const repertoire_type current_repertoire = generic_repertoires_[current];
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
            .present_traits = enumerate_present_traits(r),
            .accessible_traits = {},
            .successors = {}
        });
        generic_repertoires_.push_back(r);
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

    [[nodiscard]] std::size_t checked_height_state_count() const {
        std::size_t count = 1;
        for (std::size_t column = 0; column < lattice_.columns(); ++column) {
            if (count > std::numeric_limits<StateId>::max() / height_base_) {
                throw std::overflow_error("ReachableStates: depth-1 state count exceeds StateId range");
            }
            count *= height_base_;
        }
        return count;
    }

    [[nodiscard]] std::size_t checked_bitmask_state_count() const {
        const auto trait_count = lattice_.trait_count();
        if (trait_count >= std::numeric_limits<StateId>::digits) {
            throw std::overflow_error("ReachableStates: loose state count exceeds StateId range");
        }
        return std::size_t{1} << trait_count;
    }

    [[nodiscard]] std::size_t height_multiplier(Column column) const {
        std::size_t multiplier = 1;
        for (Column c = 0; c < column; ++c) {
            multiplier *= height_base_;
        }
        return multiplier;
    }

    [[nodiscard]] std::size_t column_height(StateId state, Column column) const {
        auto value = static_cast<std::size_t>(state);
        for (Column c = 0; c < column; ++c) {
            value /= height_base_;
        }
        return value % height_base_;
    }

    [[nodiscard]] std::vector<TraitId> enumerate_present_traits_from_height_state(StateId state) const {
        std::vector<TraitId> present;
        std::size_t repertoire_size = 0;
        for (Column column = 0; column < lattice_.columns(); ++column) {
            repertoire_size += column_height(state, column);
        }
        present.reserve(repertoire_size);

        for (Column column = 0; column < lattice_.columns(); ++column) {
            const auto height = column_height(state, column);
            for (Layer layer = 0; layer < height; ++layer) {
                present.push_back(lattice_.id(column, layer));
            }
        }

        return present;
    }

    [[nodiscard]] std::vector<TraitId> enumerate_accessible_traits_from_height_state(StateId state) const {
        std::vector<TraitId> accessible;
        accessible.reserve(lattice_.columns());

        for (Column column = 0; column < lattice_.columns(); ++column) {
            const auto height = column_height(state, column);
            if (height < lattice_.layers()) {
                accessible.push_back(lattice_.id(column, static_cast<Layer>(height)));
            }
        }

        return accessible;
    }

    [[nodiscard]] std::vector<StateId> enumerate_successors_from_height_state(StateId state) const {
        std::vector<StateId> successors;
        successors.reserve(lattice_.columns());
        const auto state_index = static_cast<std::size_t>(state);

        for (Column column = 0; column < lattice_.columns(); ++column) {
            const auto height = column_height(state, column);
            if (height < lattice_.layers()) {
                successors.push_back(
                    static_cast<StateId>(state_index + height_multiplier(column)));
            }
        }

        return successors;
    }

    [[nodiscard]] std::vector<TraitId> enumerate_present_traits_from_bitmask_state(StateId state) const {
        std::vector<TraitId> present;
        present.reserve(lattice_.trait_count());

        for (TraitId trait = 0; trait < lattice_.trait_count(); ++trait) {
            if ((state & trait_mask(trait)) != 0U) {
                present.push_back(trait);
            }
        }

        return present;
    }

    [[nodiscard]] std::vector<TraitId> enumerate_accessible_traits_from_bitmask_state(StateId state) const {
        std::vector<TraitId> accessible;
        accessible.reserve(lattice_.trait_count());

        for (TraitId trait = 0; trait < lattice_.trait_count(); ++trait) {
            if ((state & trait_mask(trait)) == 0U) {
                accessible.push_back(trait);
            }
        }

        return accessible;
    }

    [[nodiscard]] std::vector<StateId> enumerate_successors_from_bitmask_state(StateId state) const {
        std::vector<StateId> successors;
        successors.reserve(lattice_.trait_count());

        for (TraitId trait = 0; trait < lattice_.trait_count(); ++trait) {
            const StateId mask = trait_mask(trait);
            if ((state & mask) == 0U) {
                successors.push_back(state | mask);
            }
        }

        return successors;
    }

    [[nodiscard]] repertoire_type repertoire_from_height_state(StateId state) const {
        repertoire_type out(lattice_.trait_count());
        for (const TraitId trait : enumerate_present_traits_from_height_state(state)) {
            out.add(trait);
        }
        return out;
    }

    [[nodiscard]] repertoire_type repertoire_from_bitmask_state(StateId state) const {
        repertoire_type out(lattice_.trait_count());
        for (TraitId trait = 0; trait < lattice_.trait_count(); ++trait) {
            if ((state & trait_mask(trait)) != 0U) {
                out.add(trait);
            }
        }
        return out;
    }

    [[nodiscard]] std::optional<StateId> state_id_for_height_repertoire(const repertoire_type& r) const {
        if (r.trait_count() != lattice_.trait_count()) {
            return {};
        }

        std::size_t state = 0;
        for (Column column = 0; column < lattice_.columns(); ++column) {
            std::size_t height = 0;
            bool found_gap = false;

            for (Layer layer = 0; layer < lattice_.layers(); ++layer) {
                const bool present = r.contains(lattice_.id(column, layer));
                if (present && found_gap) {
                    return {};
                }
                if (present) {
                    ++height;
                } else {
                    found_gap = true;
                }
            }

            state += height * height_multiplier(column);
        }

        if (state >= state_count_limit_) {
            return {};
        }
        return static_cast<StateId>(state);
    }

    [[nodiscard]] std::optional<StateId> state_id_for_bitmask_repertoire(const repertoire_type& r) const {
        if (r.trait_count() != lattice_.trait_count()) {
            return {};
        }

        StateId state = 0;
        for (TraitId trait = 0; trait < lattice_.trait_count(); ++trait) {
            if (r.contains(trait)) {
                state |= trait_mask(trait);
            }
        }

        if (static_cast<std::size_t>(state) >= state_count_limit_) {
            return {};
        }
        return state;
    }

    [[nodiscard]] static StateId trait_mask(TraitId trait) noexcept {
        return StateId{1} << trait;
    }

    [[nodiscard]] std::pair<StateId, bool> find_or_insert_state(const repertoire_type& r) {
        if (const auto it = index_.find(r); it != index_.end()) {
            return {it->second, false};
        }

        const StateId id = insert_state(r);
        return {id, true};
    }

    void check_state(StateId state) const {
        if (static_cast<std::size_t>(state) >= size()) {
            throw std::out_of_range("ReachableStates: invalid state id");
        }
    }

    std::vector<StateRecord> states_;
    std::vector<repertoire_type> generic_repertoires_;
    std::unordered_map<repertoire_type, StateId, repertoire_hash_type> index_;
    lattice_type lattice_;
    prerequisites_type prerequisites_;
    enum class Encoding : std::uint8_t {
        generic,
        height,
        loose_bitmask
    };
    Encoding encoding_{Encoding::generic};
    bool uses_encoded_states_{false};
    std::size_t height_base_{0};
    std::size_t state_count_limit_{0};
};

} // namespace island_model
