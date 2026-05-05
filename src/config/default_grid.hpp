#pragma once

#include "../config/parameter_grid.hpp"

namespace island_model {

inline ParameterGrid make_default_grid() {
    ParameterGrid grid;

    // Lattice structure
    grid.columns = {8};
    grid.layers = {6};
    grid.cross_column_depths = {6};
    grid.island_counts = {1};
    grid.seeds = {12345, 67890, 11111, 22222, 33333};

    // Single island: migration and cross-island correlation are fixed and inactive
    grid.ms = {0.0};
    grid.ks = {0.0};

    // Learning parameters
    grid.rhos = {0.005, 0.02, 0.1};
    grid.mus = {0.01, 0.05};
    grid.alphas = {1.0, 2.0, 4.0};
    grid.betas = {0.0, 1.0, 2.0};
    grid.gammas = {0.0, 0.5};
    grid.etas = {0.0};

    // Payoff structure
    grid.deltas = {0.2, 0.4};
    grid.sigma_bs = {0.5, 1.0};
    grid.sigma_nus = {0.25};

    return grid;
}


inline ParameterGrid make_test_grid() {
    ParameterGrid grid;

    // Lattice structure
    grid.columns = {4};
    grid.layers = {6};
    grid.cross_column_depths = {6};
    grid.island_counts = {10};
    grid.seeds = {12345};

    // Migration 
    grid.ms = {0.0};

    // Learning parameters
    grid.rhos = {0.01, 0.02};
    grid.mus = {0.05, 0.1};
    grid.alphas = {1};
    grid.betas = {1.0}; 
    grid.gammas = {0.0};
    grid.etas = {0.0};

    // Payoff structure
    grid.deltas = {0.3};
    grid.sigma_bs = {0.5};
    grid.sigma_nus = {0.1};
    grid.ks = {0.2};

    return grid;
}

} // namespace island_model
