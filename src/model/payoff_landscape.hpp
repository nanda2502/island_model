#pragma once

#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

#include "lattice.hpp"

namespace island_model {

struct PayoffParams {
    double delta{0.0};
    double sigma_b{0.0};
    double sigma_nu{0.0};
    double k{0.0};
};

class PayoffLandscape {
public:
    using lattice_type = Lattice;

    PayoffLandscape() = default;

    template <class Rng>
    PayoffLandscape(lattice_type lattice,
                    std::size_t island_count,
                    PayoffParams params,
                    Rng& rng)
        : lattice_(lattice),
          island_count_(island_count),
          payoffs_(island_count_ * lattice_.trait_count(), 0.0) {
        validate(params);
        if (lattice_.trait_count() == 0) {
            throw std::invalid_argument("PayoffLandscape: lattice must be initialized");
        }
        build(params, rng);
        validate_generated_payoffs();
        
    }

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
        return payoffs_[index(island, trait)];
    }

    [[nodiscard]] const std::vector<double>& data() const noexcept {
        return payoffs_;
    }

private:
    static constexpr double payoff_buffer_epsilon_{1e-6};

    template <class Rng>
    void build(const PayoffParams& params, Rng& rng) {
        std::normal_distribution<double> standard_normal(0.0, 1.0);

        // Shared column effect across islands.
        std::vector<double> shared(lattice_.columns(), 0.0);
        std::vector<double> column_effect(island_count_ * lattice_.columns(), 0.0);

        const double shared_scale = params.sigma_b * std::sqrt(params.k);
        const double residual_scale = params.sigma_b * std::sqrt(1.0 - params.k);

        for (double& value : shared) {
            value = standard_normal(rng);
        }

        for (IslandId island = 0; island < island_count_; ++island) {
            for (Column column = 0; column < lattice_.columns(); ++column) {
                column_effect[column_index(island, column)] =
                    shared_scale * shared[column] +
                    residual_scale * standard_normal(rng);
            }
        }

        for (Layer layer = 0; layer < lattice_.layers(); ++layer) {
            for (IslandId island = 0; island < island_count_; ++island) {
                for (Column column = 0; column < lattice_.columns(); ++column) {
                    const TraitId trait = lattice_.id(column, layer);
                    const double mean = params.delta * static_cast<double>(layer + 1);
                    const double nu = params.sigma_nu * standard_normal(rng);
                    payoffs_[index(island, trait)] =
                        mean + column_effect[column_index(island, column)] + nu;
                }
            }
        }

        affine_normalize_payoffs();
    }

    void affine_normalize_payoffs() {
        if (payoffs_.empty()) {
            return;
        }

        double min_value = payoffs_.front();
        for (const double value : payoffs_) {
            if (value < min_value) {
                min_value = value;
            }
        }

        const double shift = payoff_buffer_epsilon_ - min_value;
        double sum = 0.0;
        for (double& value : payoffs_) {
            value += shift;
            sum += value;
        }

        const double mean =
            sum / static_cast<double>(payoffs_.size());
        if (mean <= 0.0 || std::isnan(mean)) {
            throw std::runtime_error("PayoffLandscape: invalid affine normalization mean");
        }

        for (double& value : payoffs_) {
            value /= mean;
        }
    }

    [[nodiscard]] std::size_t index(IslandId island, TraitId trait) const noexcept {
        return static_cast<std::size_t>(island) * lattice_.trait_count()
             + static_cast<std::size_t>(trait);
    }

    [[nodiscard]] std::size_t column_index(IslandId island, Column column) const noexcept {
        return static_cast<std::size_t>(island) * lattice_.columns()
             + static_cast<std::size_t>(column);
    }

    static void validate(const PayoffParams& params) {
        if (params.sigma_b < 0.0) {
            throw std::invalid_argument("PayoffLandscape: sigma_b must be >= 0");
        }
        if (params.sigma_nu < 0.0) {
            throw std::invalid_argument("PayoffLandscape: sigma_nu must be >= 0");
        }
        if (params.k < 0.0 || params.k > 1.0) {
            throw std::invalid_argument("PayoffLandscape: k must be in [0, 1]");
        }
    }

    void validate_generated_payoffs() const {
        for (IslandId island = 0; island < island_count_; ++island) {
            for (TraitId trait = 0; trait < lattice_.trait_count(); ++trait) {
                const double value = (*this)(island, trait);
                if (value <= 0.0 || std::isnan(value)) {
                    throw std::runtime_error("PayoffLandscape: non-positive payoff generated");
                }
            }
        }
    }

    void check_island(IslandId island) const {
        if (static_cast<std::size_t>(island) >= island_count_) {
            throw std::out_of_range("PayoffLandscape: invalid island id");
        }
    }

    void check_trait(TraitId trait) const {
        if (!lattice_.valid_trait(trait)) {
            throw std::out_of_range("PayoffLandscape: invalid trait id");
        }
    }

    lattice_type lattice_{};
    std::size_t island_count_{0};
    std::vector<double> payoffs_{};
};

} // namespace island_model
