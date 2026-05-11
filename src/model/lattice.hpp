#pragma once

#include "../core/ids.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace island_model {

struct TraitPos {
    Column column{};
    Layer layer{};

    auto operator <=> (const TraitPos&) const = default;
};

class Lattice {
public:
    Lattice() = default;

    Lattice(std::size_t columns,
            std::size_t layers,
            std::size_t cross_column_depth = 1)
        : columns_(columns),
          layers_(layers),
          cross_column_depth_(checked_cross_column_depth(cross_column_depth, layers)),
          trait_count_(checked_trait_count(columns, layers)) {}

    [[nodiscard]] std::size_t columns() const noexcept { return columns_; }
    [[nodiscard]] std::size_t layers() const noexcept { return layers_; }
    [[nodiscard]] std::size_t cross_column_depth() const noexcept { return cross_column_depth_; }
    [[nodiscard]] std::size_t trait_count() const noexcept { return trait_count_; }
    [[nodiscard]] bool operator==(const Lattice& other) const noexcept {
        return columns_ == other.columns_
            && layers_ == other.layers_
            && cross_column_depth_ == other.cross_column_depth_
            && trait_count_ == other.trait_count_;
    }

    [[nodiscard]] bool operator!=(const Lattice& other) const noexcept {
        return !(*this == other);
    }

    [[nodiscard]] bool valid_column(Column c) const noexcept {
        return static_cast<std::size_t>(c) < columns_;
    }

    [[nodiscard]] bool valid_layer(Layer l) const noexcept {
        return static_cast<std::size_t>(l) < layers_;
    }

    [[nodiscard]] bool valid_trait(TraitId t) const noexcept {
        return static_cast<std::size_t>(t) < trait_count_;
    }

    [[nodiscard]] TraitId id(Column c, Layer l) const {
        if (!valid_column(c)) {
            throw std::out_of_range("Lattice::id: invalid column");
        }
        if (!valid_layer(l)) {
            throw std::out_of_range("Lattice::id: invalid layer");
        }

        return static_cast<TraitId>(static_cast<std::size_t>(l) * columns_ + static_cast<std::size_t>(c));
    }

    [[nodiscard]] TraitPos pos(TraitId t) const {
        if (!valid_trait(t)) {
            throw std::out_of_range("Lattice::pos: invalid trait id");
        }

        const auto idx = static_cast<std::size_t>(t);
        return TraitPos{
            .column = static_cast<Column>(idx % columns_),
            .layer = static_cast<Layer>(idx / columns_)
        };
    }

    [[nodiscard]] bool is_base_layer(TraitId t) const {
        return pos(t).layer == 0;
    }

    [[nodiscard]] Column wrap_left(Column c) const {
        if (!valid_column(c)) {
            throw std::out_of_range("Lattice::wrap_left: invalid column");
        }
        return (c == 0) ? static_cast<Column>(columns_ - 1) : static_cast<Column>(c - 1);
    }

    [[nodiscard]] Column wrap_right(Column c) const {
        if (!valid_column(c)) {
            throw std::out_of_range("Lattice::wrap_right: invalid column");
        }
        return static_cast<Column>((static_cast<std::size_t>(c) + 1) % columns_);
    }

    [[nodiscard]] std::vector<TraitId> parent_neighborhood(TraitId t) const {
        const auto p = pos(t);

        if (p.layer == 0) {
            throw std::logic_error("Lattice::parent_neighborhood: base-layer traits have no parents");
        }

        const auto parent_layer = static_cast<Layer>(p.layer - 1);

        std::vector<TraitId> parents;
        parents.reserve(has_cross_column_parents(p) ? 3 : 1);

        if (has_cross_column_parents(p)) {
            parents.push_back(id(wrap_left(p.column), parent_layer));
        }
        parents.push_back(id(p.column, parent_layer));
        if (has_cross_column_parents(p)) {
            parents.push_back(id(wrap_right(p.column), parent_layer));
        }

        return parents;
    }

    [[nodiscard]] std::size_t column_distance(Column a, Column b) const {
        if (!valid_column(a) || !valid_column(b)) {
            throw std::out_of_range("Lattice::column_distance: invalid column");
        }

        const auto da = static_cast<std::size_t>(a);
        const auto db = static_cast<std::size_t>(b);
        const auto direct = (da > db) ? (da - db) : (db - da);
        const auto wrap = columns_ - direct;
        return (direct < wrap) ? direct : wrap;
    }

private:
    static std::size_t checked_trait_count(std::size_t columns, std::size_t layers) {
        if (columns == 0) {
            throw std::invalid_argument("Lattice: columns must be > 0");
        }
        if (layers == 0) {
            throw std::invalid_argument("Lattice: layers must be > 0");
        }

        const auto max_traits = static_cast<std::size_t>(std::numeric_limits<TraitId>::max()) + 1ULL;
        if (columns > max_traits / layers) {
            throw std::invalid_argument("Lattice: TraitId is too small for this lattice size");
        }

        return columns * layers;
    }

    static std::size_t checked_cross_column_depth(std::size_t cross_column_depth,
                                                  std::size_t layers) {
        if (cross_column_depth == std::numeric_limits<std::size_t>::max()) {
            return 1;
        }
        if (cross_column_depth == 0) {
            throw std::invalid_argument("Lattice: cross_column_depth must be > 0");
        }
        if (cross_column_depth != 1) {
            throw std::invalid_argument("Lattice: cross_column_depth is fixed at 1");
        }
        if (cross_column_depth > layers) {
            throw std::invalid_argument("Lattice: cross_column_depth must be <= layers");
        }
        return cross_column_depth;
    }

    [[nodiscard]] bool has_cross_column_parents(TraitPos p) const noexcept {
        return static_cast<std::size_t>(p.layer) + 1U <= cross_column_depth_;
    }

    std::size_t columns_{0};
    std::size_t layers_{0};
    std::size_t cross_column_depth_{0};
    std::size_t trait_count_{0};
};

} // namespace island_model
