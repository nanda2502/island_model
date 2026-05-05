#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "../dynamics/visibility.hpp"
#include "../model/payoff_landscape.hpp"

namespace island_model {

class WeightField {
public:
    using lattice_type = Lattice;

    WeightField() = default;

    WeightField(lattice_type lattice, std::size_t island_count)
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
            throw std::out_of_range("WeightField: invalid island id");
        }
    }

    void check_trait(TraitId trait) const {
        if (!lattice_.valid_trait(trait)) {
            throw std::out_of_range("WeightField: invalid trait id");
        }
    }

    lattice_type lattice_{};
    std::size_t island_count_{0};
    std::vector<double> values_{};
};

class Weights {
public:
    using lattice_type = Lattice;
    using visibility_field_type = VisibilityField;
    using payoff_landscape_type = PayoffLandscape;
    using weight_field_type = WeightField;

    static weight_field_type compute(const visibility_field_type& visibility,
                                     const payoff_landscape_type& payoff,
                                     double alpha,
                                     double gamma) {
        if (alpha < 0.0) {
            throw std::invalid_argument("Weights::compute: alpha must be >= 0");
        }
        if (gamma < 0.0) {
            throw std::invalid_argument("Weights::compute: gamma must be >= 0");
        }
        if (visibility.island_count() != payoff.island_count()) {
            throw std::invalid_argument("Weights::compute: island count mismatch");
        }
        if (visibility.lattice() != payoff.lattice()) {
            throw std::invalid_argument("Weights::compute: lattice mismatch");
        }

        const std::size_t island_count = visibility.island_count();
        const std::size_t trait_count = payoff.trait_count();

        weight_field_type out(payoff.lattice(), island_count);
        out.fill(0.0);

        #pragma omp parallel for schedule(static)
        for (std::size_t island_idx = 0; island_idx < island_count; ++island_idx) {
            const auto island = static_cast<IslandId>(island_idx);
            double payoff_sum = 0.0;
            for (TraitId trait = 0; trait < trait_count; ++trait) {
                payoff_sum += payoff(island, trait);
            }
            if (payoff_sum <= 0.0) {
                throw std::runtime_error("Weights::compute: non-positive island payoff sum");
            }

            double normalizer = 0.0;

            for (TraitId trait = 0; trait < trait_count; ++trait) {
                const double normalized_payoff = payoff(island, trait) / payoff_sum;
                const double value =
                    std::pow(visibility(island, trait), alpha) *
                    std::pow(normalized_payoff, gamma);

                out(island, trait) = value;
                normalizer += value;
            }

            if (normalizer > 0.0) {
                for (TraitId trait = 0; trait < trait_count; ++trait) {
                    out(island, trait) /= normalizer;
                }
            }
        }

        return out;
    }
};

} // namespace island_model
