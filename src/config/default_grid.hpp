#pragma once

#include "../config/parameter_grid.hpp"

namespace island_model {

inline ParameterGrid make_default_grid() {
    ParameterGrid grid;

    grid.seeds = {12345, 67890, 11111, 22222, 33333};
    grid.island_counts = {10,20};
    
    // Lattice structure
    grid.columns = {4};
    grid.layers = {6};
    grid.cross_column_depths = {1};
    grid.strictnesses = {1.0};
    
    
    // Migration
    grid.ms = {0.0, 0.01, 0.1, 0.3};

    // Reset and innovation rate
    grid.rhos = {0.02, 0.05, 0.1, 0.2};
    grid.mus = {0.01};
    
    // Conformity
    grid.alphas = {1.0, 3.0};
    // Payoff expression bias
    grid.betas = {1.0};
    // Prestige
    grid.lambdas = {0.0};
    // Payoff
    grid.gammas = {0.0};
    // Transparency
    grid.etas = {1.0};

    // Payoff structure
    // Increase in payoff for deeper traits
    grid.deltas = {1.0};
    // Ruggedness of payoffs across columns
    grid.sigma_bs = {0.0, 0.25, 1.0};
    // Trait-level variation
    grid.sigma_nus = {0.0};
    // Payoff correlation between islands
    grid.ks = {0.0};

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
    grid.strictnesses = {1.0};

    // Migration 
    grid.ms = {0.0};

    // Learning parameters
    grid.rhos = {0.01, 0.02};
    grid.mus = {0.05, 0.1};
    grid.alphas = {1};
    grid.betas = {1.0}; 
    grid.lambdas = {0.0};
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
