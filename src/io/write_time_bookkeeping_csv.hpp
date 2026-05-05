#pragma once

#include <ostream>

#include "../analysis/time_bookkeeping.hpp"
#include "write_equilibrium_csv.hpp"

namespace island_model {

class TimeBookkeepingCsvWriter {
public:
    static void write_header(std::ostream& out) {
        out
            << "run_id,"
            << "seed,"
            << "columns,"
            << "layers,"
            << "cross_column_depth,"
            << "island_count,"
            << "m,"
            << "rho,"
            << "mu,"
            << "alpha,"
            << "beta,"
            << "gamma,"
            << "eta,"
            << "delta,"
            << "sigma_b,"
            << "sigma_nu,"
            << "k,"
            << "converged,"
            << "steps_to_equilibrium,"
            << "final_distance,"
            << "step,"
            << "f_rep,"
            << "within_distance,"
            << "total_distance,"
            << "mean_payoff,"
            << "adj_payoff,"
            << "mean_max_depth,"
            << "mean_depth,"
            << "eff_column,"
            << "top_col_mass,"
            << "mean_rep_size,"
            << "empty_rep_size\n";
    }

    static void write_rows(std::ostream& out,
                           const EquilibriumRunMetadata& meta,
                           const std::vector<PopulationBookkeepingSnapshot>& snapshots) {
        for (const auto& snapshot : snapshots) {
            out << meta.run_id << ','
                << meta.seed << ','
                << meta.columns << ','
                << meta.layers << ','
                << meta.cross_column_depth << ','
                << meta.island_count << ','
                << meta.m << ','
                << meta.rho << ','
                << meta.mu << ','
                << meta.alpha << ','
                << meta.beta << ','
                << meta.gamma << ','
                << meta.eta << ','
                << meta.delta << ','
                << meta.sigma_b << ','
                << meta.sigma_nu << ','
                << meta.k << ','
                << meta.converged << ','
                << meta.steps_to_equilibrium << ','
                << meta.final_distance << ','
                << snapshot.step << ','
                << snapshot.f_rep << ','
                << snapshot.within_distance << ','
                << snapshot.total_distance << ','
                << snapshot.mean_payoff << ','
                << snapshot.adj_payoff << ','
                << snapshot.mean_max_depth << ','
                << snapshot.mean_depth << ','
                << snapshot.eff_column << ','
                << snapshot.top_col_mass << ','
                << snapshot.mean_rep_size << ','
                << snapshot.empty_rep_size
                << '\n';
        }
    }
};

} // namespace island_model
