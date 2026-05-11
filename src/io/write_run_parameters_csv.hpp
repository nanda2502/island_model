#pragma once

#include <ostream>

#include "write_equilibrium_csv.hpp"

namespace island_model {

class RunParametersCsvWriter {
public:
    static void write_header(std::ostream& out) {
        out
            << "run_index,"
            << "run_id,"
            << "seed,"
            << "columns,"
            << "layers,"
            << "cross_column_depth,"
            << "island_count,"
            << "strictness,"
            << "m,"
            << "rho,"
            << "mu,"
            << "alpha,"
            << "beta,"
            << "lambda,"
            << "gamma,"
            << "eta,"
            << "delta,"
            << "sigma_b,"
            << "sigma_nu,"
            << "k,"
            << "converged,"
            << "steps_to_equilibrium,"
            << "final_distance\n";
    }

    static void write_row(std::ostream& out, const EquilibriumRunMetadata& meta) {
        out << meta.run_index << ','
            << meta.run_id << ','
            << meta.seed << ','
            << meta.columns << ','
            << meta.layers << ','
            << meta.cross_column_depth << ','
            << meta.island_count << ','
            << meta.strictness << ','
            << meta.m << ','
            << meta.rho << ','
            << meta.mu << ','
            << meta.alpha << ','
            << meta.beta << ','
            << meta.lambda << ','
            << meta.gamma << ','
            << meta.eta << ','
            << meta.delta << ','
            << meta.sigma_b << ','
            << meta.sigma_nu << ','
            << meta.k << ','
            << meta.converged << ','
            << meta.steps_to_equilibrium << ','
            << meta.final_distance
            << '\n';
    }
};

} // namespace island_model
