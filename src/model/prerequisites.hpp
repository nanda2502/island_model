#pragma once

#include <vector>

#include "../state_space/repertoire.hpp"
#include "lattice.hpp"
#include "../core/ids.hpp"

namespace island_model {

class Prerequisites {
public: 
    using lattice_type = Lattice;
    using repertoire_type = Repertoire;

    Prerequisites() = default;

    explicit Prerequisites(lattice_type lattice)
        : lattice_(lattice) {}

    [[nodiscard]] const lattice_type& lattice() const noexcept {
        return lattice_;
    }

    [[nodiscard]] bool has_any_parent(TraitId trait, const repertoire_type& r) const {
        const auto parents = lattice_.parent_neighborhood(trait);
        for (const TraitId parent : parents) {
            if (r.contains(parent)) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] bool is_accessible(TraitId trait, const repertoire_type& r) const {
        if (r.contains(trait)) {
            return false;
        }

        if (lattice_.is_base_layer(trait)) {
            return true;
        }

        return has_any_parent(trait, r);
    }

    [[nodiscard]] std::vector<TraitId> accessible_traits(const repertoire_type& r) const {
        std::vector<TraitId> traits;
        traits.reserve(lattice_.trait_count());

        for (TraitId t = 0; t < lattice_.trait_count(); ++t) {
            if (is_accessible(t, r)) {
                traits.push_back(t);
            }
        }

        return traits;
    }

private:
    lattice_type lattice_{};
};

} // namespace island_model
