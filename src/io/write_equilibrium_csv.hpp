#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <stdexcept>

#include "../analysis/metrics.hpp"
#include "../analysis/time_bookkeeping.hpp"
#include "../model/payoff_landscape.hpp"
#include "../state/population_state.hpp"
#include "../state_space/reachable_states.hpp"

namespace island_model {

struct EquilibriumRunMetadata {
    std::size_t run_index{0};
    std::uint64_t run_id{0};
    std::uint64_t seed{0};

    std::size_t columns{0};
    std::size_t layers{0};
    std::size_t cross_column_depth{0};
    std::size_t island_count{0};
    double strictness{1.0};

    double m{0.0};
    double rho{0.0};
    double mu{0.0};
    double alpha{0.0};
    double beta{0.0};
    double lambda{0.0};
    double gamma{0.0};
    double eta{0.0};

    double delta{0.0};
    double sigma_b{0.0};
    double sigma_nu{0.0};
    double k{0.0};

    std::uint32_t converged{0};
    std::size_t steps_to_equilibrium{0};
    double final_distance{0.0};
};

class EquilibriumCsvWriter {
public:
    using reachable_states_type = ReachableStates;
    using population_state_type = PopulationState;
    using payoff_landscape_type = PayoffLandscape;

    static void write_header(std::ostream& out) {
        out
            << "run_index,"
            << "run_id,"
            << "island,"
            << "repertoire_id,"
            << "frequency,"
            << "repertoire_size,"
            << "max_layer,"
            << "repertoire_payoff_sum,"
            << "repertoire_payoff_mean,"
            << "repertoire_payoff_max,"
            << "adaptive_divergence_run,"
            << "cultural_divergence_run,"
            << "divergence_pair_count_run\n";
    }

    static void write_rows(std::ostream& out,
                           const EquilibriumRunMetadata& meta,
                           const population_state_type& equilibrium_state,
                           const reachable_states_type& states,
                           const payoff_landscape_type& payoff,
                           double frequency_threshold = 0.0) {
        validate(meta, equilibrium_state, states, payoff, frequency_threshold);
        const auto adaptive_divergence =
            DifferentiationMetrics::adaptive_divergence(
                equilibrium_state,
                states,
                payoff);
        const auto cultural_divergence =
            DifferentiationMetrics::cultural_divergence(
                equilibrium_state,
                states);

        for (IslandId island = 0; island < equilibrium_state.island_count(); ++island) {
            for (StateId repertoire_id = 0; repertoire_id < equilibrium_state.state_count(); ++repertoire_id) {
                const double frequency = equilibrium_state(island, repertoire_id);
                if (frequency <= frequency_threshold) {
                    continue;
                }

                const auto summary =
                    RepertoireSummaries::summarize(
                        repertoire_id,
                        island,
                        states,
                        payoff);

                write_row(
                    out,
                    meta,
                    island,
                    repertoire_id,
                    frequency,
                    summary,
                    adaptive_divergence.mean_distance,
                    cultural_divergence.mean_distance,
                    adaptive_divergence.pair_count);
            }
        }
    }

private:
    static void validate(const EquilibriumRunMetadata& meta,
                         const population_state_type& equilibrium_state,
                         const reachable_states_type& states,
                         const payoff_landscape_type& payoff,
                         double frequency_threshold) {
        if (meta.columns != states.lattice().columns()) {
            throw std::invalid_argument("EquilibriumCsvWriter: metadata columns mismatch");
        }
        if (meta.layers != states.lattice().layers()) {
            throw std::invalid_argument("EquilibriumCsvWriter: metadata layers mismatch");
        }
        if (meta.cross_column_depth != states.lattice().cross_column_depth()) {
            throw std::invalid_argument("EquilibriumCsvWriter: metadata cross column depth mismatch");
        }
        if (meta.island_count != equilibrium_state.island_count()) {
            throw std::invalid_argument("EquilibriumCsvWriter: metadata island count mismatch");
        }
        if (equilibrium_state.state_count() != states.size()) {
            throw std::invalid_argument("EquilibriumCsvWriter: equilibrium/state space size mismatch");
        }
        if (equilibrium_state.island_count() != payoff.island_count()) {
            throw std::invalid_argument("EquilibriumCsvWriter: equilibrium/payoff island count mismatch");
        }
        if (states.lattice() != payoff.lattice()) {
            throw std::invalid_argument("EquilibriumCsvWriter: lattice mismatch");
        }
        if (frequency_threshold < 0.0) {
            throw std::invalid_argument("EquilibriumCsvWriter: frequency threshold must be >= 0");
        }
    }

    static void write_row(std::ostream& out,
                          const EquilibriumRunMetadata& meta,
                          IslandId island,
                          StateId repertoire_id,
                          double frequency,
                          const RepertoireSummary& summary,
                          double adaptive_divergence_run,
                          double cultural_divergence_run,
                          std::size_t divergence_pair_count_run) {
        out << meta.run_index << ','
            << meta.run_id << ','
            << island << ','
            << repertoire_id << ','
            << frequency << ','
            << summary.repertoire_size << ','
            << summary.max_layer << ','
            << summary.repertoire_payoff_sum << ','
            << summary.repertoire_payoff_mean << ','
            << summary.repertoire_payoff_max << ','
            << adaptive_divergence_run << ','
            << cultural_divergence_run << ','
            << divergence_pair_count_run
            << '\n';
    }
};

class EquilibriumSummaryCsvWriter {
public:
    static void write_header(std::ostream& out) {
        out
            << "run_index,"
            << "run_id,"
            << "converged,"
            << "steps_to_equilibrium,"
            << "final_distance,"
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

    static void write_row(std::ostream& out,
                          const EquilibriumRunMetadata& meta,
                          const PopulationBookkeepingSnapshot& summary) {
        out << meta.run_index << ','
            << meta.run_id << ','
            << meta.converged << ','
            << meta.steps_to_equilibrium << ','
            << meta.final_distance << ','
            << summary.adaptive_divergence << ','
            << summary.cultural_divergence << ','
            << summary.divergence_pair_count << ','
            << summary.mean_payoff << ','
            << summary.adj_payoff << ','
            << summary.mean_max_depth << ','
            << summary.mean_depth << ','
            << summary.eff_column << ','
            << summary.top_col_mass << ','
            << summary.mean_rep_size << ','
            << summary.empty_rep_size
            << '\n';
    }
};

class TraitEquilibriumCsvWriter {
public:
    using reachable_states_type = ReachableStates;
    using population_state_type = PopulationState;

    static void write_header(std::ostream& out) {
        out
            << "run_index,"
            << "run_id,"
            << "island,"
            << "trait_id,"
            << "trait_column,"
            << "trait_layer,"
            << "is_base_layer,"
            << "frequency\n";
    }

    static void write_rows(std::ostream& out,
                           const EquilibriumRunMetadata& meta,
                           const population_state_type& equilibrium_state,
                           const reachable_states_type& states) {
        validate(meta, equilibrium_state, states);
        const auto frequencies =
            DifferentiationMetrics::trait_frequencies(equilibrium_state, states);
        const auto& lattice = states.lattice();

        for (IslandId island = 0; island < frequencies.island_count; ++island) {
            for (TraitId trait = 0; trait < frequencies.trait_count; ++trait) {
                const auto pos = lattice.pos(trait);
                out << meta.run_index << ','
                    << meta.run_id << ','
                    << island << ','
                    << trait << ','
                    << pos.column << ','
                    << pos.layer << ','
                    << static_cast<std::uint32_t>(pos.layer == 0 ? 1 : 0) << ','
                    << frequencies(island, trait)
                    << '\n';
            }
        }
    }

private:
    static void validate(const EquilibriumRunMetadata& meta,
                         const population_state_type& equilibrium_state,
                         const reachable_states_type& states) {
        if (meta.columns != states.lattice().columns()) {
            throw std::invalid_argument("TraitEquilibriumCsvWriter: metadata columns mismatch");
        }
        if (meta.layers != states.lattice().layers()) {
            throw std::invalid_argument("TraitEquilibriumCsvWriter: metadata layers mismatch");
        }
        if (meta.cross_column_depth != states.lattice().cross_column_depth()) {
            throw std::invalid_argument("TraitEquilibriumCsvWriter: metadata cross column depth mismatch");
        }
        if (meta.island_count != equilibrium_state.island_count()) {
            throw std::invalid_argument("TraitEquilibriumCsvWriter: metadata island count mismatch");
        }
        if (equilibrium_state.state_count() != states.size()) {
            throw std::invalid_argument("TraitEquilibriumCsvWriter: equilibrium/state space size mismatch");
        }
    }
};

} // namespace island_model
