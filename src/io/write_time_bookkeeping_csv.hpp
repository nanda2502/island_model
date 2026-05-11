#pragma once

#include <ostream>

#include "../analysis/time_bookkeeping.hpp"
#include "write_equilibrium_csv.hpp"

namespace island_model {

class TimeBookkeepingCsvWriter {
public:
    static void write_header(std::ostream& out) {
        out
            << "run_index,"
            << "run_id,"
            << "step,"
            << "adaptive_divergence,"
            << "cultural_divergence,"
            << "divergence_pair_count,"
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
            out << meta.run_index << ','
                << meta.run_id << ','
                << snapshot.step << ','
                << snapshot.adaptive_divergence << ','
                << snapshot.cultural_divergence << ','
                << snapshot.divergence_pair_count << ','
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
