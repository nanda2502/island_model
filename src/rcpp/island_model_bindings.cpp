// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../analysis/metrics.hpp"
#include "../analysis/time_bookkeeping.hpp"
#include "../config/parameter_grid.hpp"
#include "../dynamics/simulator.hpp"
#include "../io/write_equilibrium_csv.hpp"
#include "../model/lattice.hpp"
#include "../model/payoff_landscape.hpp"
#include "../state_space/reachable_states.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace im = island_model;

namespace {

Rcpp::DataFrame lattice_vertices_df(const im::Lattice& lattice) {
    const auto trait_count = lattice.trait_count();

    Rcpp::IntegerVector trait_id(trait_count);
    Rcpp::IntegerVector column(trait_count);
    Rcpp::IntegerVector layer(trait_count);
    Rcpp::LogicalVector is_base_layer(trait_count);

    for (std::size_t trait = 0; trait < trait_count; ++trait) {
        const auto pos = lattice.pos(static_cast<TraitId>(trait));
        trait_id[trait] = static_cast<int>(trait);
        column[trait] = static_cast<int>(pos.column);
        layer[trait] = static_cast<int>(pos.layer);
        is_base_layer[trait] = (pos.layer == 0);
    }

    return Rcpp::DataFrame::create(
        Rcpp::Named("trait_id") = trait_id,
        Rcpp::Named("column") = column,
        Rcpp::Named("layer") = layer,
        Rcpp::Named("is_base_layer") = is_base_layer
    );
}

Rcpp::DataFrame lattice_edges_df(const im::Lattice& lattice) {
    std::vector<int> from;
    std::vector<int> to;
    from.reserve((lattice.layers() > 1) ? (lattice.layers() - 1) * lattice.columns() * 3 : 0);
    to.reserve(from.capacity());

    for (TraitId trait = 0; trait < lattice.trait_count(); ++trait) {
        if (lattice.is_base_layer(trait)) {
            continue;
        }

        const auto parents = lattice.parent_neighborhood(trait);
        for (const TraitId parent : parents) {
            from.push_back(static_cast<int>(parent));
            to.push_back(static_cast<int>(trait));
        }
    }

    return Rcpp::DataFrame::create(
        Rcpp::Named("from") = from,
        Rcpp::Named("to") = to
    );
}

void write_u64_le(std::uint64_t value, Rcpp::RawVector& out, std::size_t offset) {
    for (std::size_t byte = 0; byte < 8; ++byte) {
        out[offset + byte] = static_cast<Rbyte>((value >> (8 * byte)) & 0xFFU);
    }
}

Rcpp::List reachable_states_payload(const im::ReachableStates& states) {
    const std::size_t state_count = states.size();
    const std::size_t word_count =
        states.size() == 0 ? 0 : states.repertoire(im::ReachableStates::empty_state()).words().size();

    Rcpp::RawVector repertoire_raw(state_count * word_count * 8);

    std::size_t raw_offset = 0;

    for (StateId state = 0; state < state_count; ++state) {
        const auto& repertoire_words = states.repertoire(state).words();
        for (const std::uint64_t word : repertoire_words) {
            write_u64_le(word, repertoire_raw, raw_offset);
            raw_offset += 8;
        }
    }

    repertoire_raw.attr("dim") = Rcpp::IntegerVector::create(
        static_cast<int>(state_count),
        static_cast<int>(word_count * 8)
    );

    return Rcpp::List::create(
        Rcpp::Named("empty_state") = 0,
        Rcpp::Named("word_bits") = static_cast<int>(im::Repertoire::word_bits_v),
        Rcpp::Named("word_count") = static_cast<int>(word_count),
        Rcpp::Named("repertoire_raw") = repertoire_raw
    );
}

struct LatticeConfigKey {
    std::size_t columns{0};
    std::size_t layers{0};
    std::size_t cross_column_depth{0};

    [[nodiscard]] bool operator==(const LatticeConfigKey& other) const noexcept {
        return columns == other.columns
            && layers == other.layers
            && cross_column_depth == other.cross_column_depth;
    }
};

struct LatticeConfigKeyHash {
    [[nodiscard]] std::size_t operator()(const LatticeConfigKey& key) const noexcept {
        return std::hash<std::size_t>{}(key.columns)
            ^ (std::hash<std::size_t>{}(key.layers) << 1U)
            ^ (std::hash<std::size_t>{}(key.cross_column_depth) << 2U);
    }
};

using reachable_states_ptr = std::shared_ptr<const im::ReachableStates>;

struct EquilibriumRow {
    std::uint64_t run_id{0};
    std::uint64_t seed{0};
    std::size_t columns{0};
    std::size_t layers{0};
    std::size_t cross_column_depth{0};
    std::size_t island_count{0};
    double m{0.0};
    double rho{0.0};
    double mu{0.0};
    double alpha{0.0};
    double beta{0.0};
    double gamma{0.0};
    double eta{0.0};
    double delta{0.0};
    double sigma_b{0.0};
    double sigma_nu{0.0};
    double k{0.0};
    std::uint32_t converged{0};
    std::size_t steps_to_equilibrium{0};
    double final_distance{0.0};
    std::size_t island{0};
    std::size_t repertoire_id{0};
    double frequency{0.0};
    std::size_t repertoire_size{0};
    std::size_t max_layer{0};
    double repertoire_payoff_sum{0.0};
    double repertoire_payoff_mean{0.0};
    double repertoire_payoff_max{0.0};
    double f_rep_run{0.0};
    double within_distance_run{0.0};
    double total_distance_run{0.0};
};

struct TimeRow {
    std::uint64_t run_id{0};
    std::uint64_t seed{0};
    std::size_t columns{0};
    std::size_t layers{0};
    std::size_t cross_column_depth{0};
    std::size_t island_count{0};
    double m{0.0};
    double rho{0.0};
    double mu{0.0};
    double alpha{0.0};
    double beta{0.0};
    double gamma{0.0};
    double eta{0.0};
    double delta{0.0};
    double sigma_b{0.0};
    double sigma_nu{0.0};
    double k{0.0};
    std::uint32_t converged{0};
    std::size_t steps_to_equilibrium{0};
    double final_distance{0.0};
    std::size_t step{0};
    double f_rep{0.0};
    double within_distance{0.0};
    double total_distance{0.0};
    double mean_payoff{0.0};
    double adj_payoff{0.0};
    double mean_max_depth{0.0};
    double mean_depth{0.0};
    double eff_column{0.0};
    double top_col_mass{0.0};
    double mean_rep_size{0.0};
    double empty_rep_size{0.0};
};

struct TraitRow {
    std::uint64_t run_id{0};
    std::uint64_t seed{0};
    std::size_t columns{0};
    std::size_t layers{0};
    std::size_t cross_column_depth{0};
    std::size_t island_count{0};
    double m{0.0};
    double rho{0.0};
    double mu{0.0};
    double alpha{0.0};
    double beta{0.0};
    double gamma{0.0};
    double eta{0.0};
    double delta{0.0};
    double sigma_b{0.0};
    double sigma_nu{0.0};
    double k{0.0};
    std::uint32_t converged{0};
    std::size_t steps_to_equilibrium{0};
    double final_distance{0.0};
    std::size_t island{0};
    std::size_t trait_id{0};
    std::size_t trait_column{0};
    std::size_t trait_layer{0};
    std::uint32_t is_base_layer{0};
    double frequency{0.0};
};

struct ErrorRow {
    std::uint64_t run_id{0};
    std::string message{};
};

struct RunOutcome {
    std::vector<EquilibriumRow> equilibrium_rows;
    std::vector<TimeRow> time_rows;
    std::vector<TraitRow> trait_rows;
    std::string error{};
};

std::size_t get_required_count(const Rcpp::DataFrame& runs) {
    return static_cast<std::size_t>(runs.nrows());
}

std::size_t resolve_cross_column_depth(int cross_column_depth, int layers) {
    return cross_column_depth < 0
        ? static_cast<std::size_t>(layers)
        : static_cast<std::size_t>(cross_column_depth);
}

template <typename VectorType, typename ValueType>
std::vector<ValueType> as_std_vector(const Rcpp::DataFrame& runs, const char* name) {
    if (!runs.containsElementNamed(name)) {
        throw std::invalid_argument(std::string("Missing required column: ") + name);
    }

    const VectorType values = runs[name];
    return Rcpp::as<std::vector<ValueType>>(values);
}

std::vector<im::RunConfig> run_configs_from_df(const Rcpp::DataFrame& runs) {
    const auto row_count = get_required_count(runs);

    const auto seeds = as_std_vector<Rcpp::NumericVector, std::uint64_t>(runs, "seed");
    const auto columns = as_std_vector<Rcpp::IntegerVector, std::size_t>(runs, "columns");
    const auto layers = as_std_vector<Rcpp::IntegerVector, std::size_t>(runs, "layers");
    const auto cross_column_depths = as_std_vector<Rcpp::IntegerVector, std::size_t>(runs, "cross_column_depth");
    const auto island_counts = as_std_vector<Rcpp::IntegerVector, std::size_t>(runs, "island_count");
    const auto ms = as_std_vector<Rcpp::NumericVector, double>(runs, "m");
    const auto rhos = as_std_vector<Rcpp::NumericVector, double>(runs, "rho");
    const auto mus = as_std_vector<Rcpp::NumericVector, double>(runs, "mu");
    const auto alphas = as_std_vector<Rcpp::NumericVector, double>(runs, "alpha");
    const auto betas = as_std_vector<Rcpp::NumericVector, double>(runs, "beta");
    const auto gammas = as_std_vector<Rcpp::NumericVector, double>(runs, "gamma");
    const auto etas = as_std_vector<Rcpp::NumericVector, double>(runs, "eta");
    const auto deltas = as_std_vector<Rcpp::NumericVector, double>(runs, "delta");
    const auto sigma_bs = as_std_vector<Rcpp::NumericVector, double>(runs, "sigma_b");
    const auto sigma_nus = as_std_vector<Rcpp::NumericVector, double>(runs, "sigma_nu");
    const auto ks = as_std_vector<Rcpp::NumericVector, double>(runs, "k");

    std::vector<im::RunConfig> configs;
    configs.reserve(row_count);

    for (std::size_t i = 0; i < row_count; ++i) {
        im::RunConfig cfg{
            .run_id = static_cast<std::uint64_t>(i),
            .seed = seeds[i],
            .columns = columns[i],
            .layers = layers[i],
            .cross_column_depth = cross_column_depths[i],
            .island_count = island_counts[i],
            .m = ms[i],
            .rho = rhos[i],
            .mu = mus[i],
            .alpha = alphas[i],
            .beta = betas[i],
            .gamma = gammas[i],
            .eta = etas[i],
            .delta = deltas[i],
            .sigma_b = sigma_bs[i],
            .sigma_nu = sigma_nus[i],
            .k = ks[i]
        };

        if (!im::is_valid_run_config(cfg)) {
            std::ostringstream msg;
            msg << "Invalid parameter values in row " << (i + 1);
            throw std::invalid_argument(msg.str());
        }

        configs.push_back(cfg);
    }

    return configs;
}

im::EquilibriumRunMetadata metadata_from_config(const im::RunConfig& cfg,
                                                const im::Simulator::EquilibriumResult& eq) {
    return im::EquilibriumRunMetadata{
        .run_id = cfg.run_id,
        .seed = cfg.seed,
        .columns = cfg.columns,
        .layers = cfg.layers,
        .cross_column_depth = cfg.cross_column_depth,
        .island_count = cfg.island_count,
        .m = cfg.m,
        .rho = cfg.rho,
        .mu = cfg.mu,
        .alpha = cfg.alpha,
        .beta = cfg.beta,
        .gamma = cfg.gamma,
        .eta = cfg.eta,
        .delta = cfg.delta,
        .sigma_b = cfg.sigma_b,
        .sigma_nu = cfg.sigma_nu,
        .k = cfg.k,
        .converged = static_cast<std::uint32_t>(eq.converged ? 1 : 0),
        .steps_to_equilibrium = eq.steps,
        .final_distance = eq.final_distance
    };
}

std::vector<EquilibriumRow> collect_equilibrium_rows(const im::EquilibriumRunMetadata& meta,
                                                     const im::PopulationState& equilibrium_state,
                                                     const im::ReachableStates& states,
                                                     const im::PayoffLandscape& payoff,
                                                     double frequency_threshold) {
    const auto differentiation_summary =
        im::DifferentiationMetrics::repertoire_differentiation(equilibrium_state, states);
    std::vector<EquilibriumRow> rows;

    for (IslandId island = 0; island < equilibrium_state.island_count(); ++island) {
        for (StateId repertoire_id = 0; repertoire_id < equilibrium_state.state_count(); ++repertoire_id) {
            const double frequency = equilibrium_state(island, repertoire_id);
            if (frequency <= frequency_threshold) {
                continue;
            }

            const auto summary =
                im::RepertoireSummaries::summarize(repertoire_id, island, states, payoff);

            rows.push_back(EquilibriumRow{
                .run_id = meta.run_id,
                .seed = meta.seed,
                .columns = meta.columns,
                .layers = meta.layers,
                .cross_column_depth = meta.cross_column_depth,
                .island_count = meta.island_count,
                .m = meta.m,
                .rho = meta.rho,
                .mu = meta.mu,
                .alpha = meta.alpha,
                .beta = meta.beta,
                .gamma = meta.gamma,
                .eta = meta.eta,
                .delta = meta.delta,
                .sigma_b = meta.sigma_b,
                .sigma_nu = meta.sigma_nu,
                .k = meta.k,
                .converged = meta.converged,
                .steps_to_equilibrium = meta.steps_to_equilibrium,
                .final_distance = meta.final_distance,
                .island = static_cast<std::size_t>(island),
                .repertoire_id = static_cast<std::size_t>(repertoire_id),
                .frequency = frequency,
                .repertoire_size = summary.repertoire_size,
                .max_layer = summary.max_layer,
                .repertoire_payoff_sum = summary.repertoire_payoff_sum,
                .repertoire_payoff_mean = summary.repertoire_payoff_mean,
                .repertoire_payoff_max = summary.repertoire_payoff_max,
                .f_rep_run = differentiation_summary.f_rep,
                .within_distance_run = differentiation_summary.within_distance,
                .total_distance_run = differentiation_summary.total_distance
            });
        }
    }

    return rows;
}

std::vector<TimeRow> collect_time_rows(const im::EquilibriumRunMetadata& meta,
                                       const std::vector<im::PopulationBookkeepingSnapshot>& snapshots) {
    std::vector<TimeRow> rows;
    rows.reserve(snapshots.size());

    for (const auto& snapshot : snapshots) {
        rows.push_back(TimeRow{
            .run_id = meta.run_id,
            .seed = meta.seed,
            .columns = meta.columns,
            .layers = meta.layers,
            .cross_column_depth = meta.cross_column_depth,
            .island_count = meta.island_count,
            .m = meta.m,
            .rho = meta.rho,
            .mu = meta.mu,
            .alpha = meta.alpha,
            .beta = meta.beta,
            .gamma = meta.gamma,
            .eta = meta.eta,
            .delta = meta.delta,
            .sigma_b = meta.sigma_b,
            .sigma_nu = meta.sigma_nu,
            .k = meta.k,
            .converged = meta.converged,
            .steps_to_equilibrium = meta.steps_to_equilibrium,
            .final_distance = meta.final_distance,
            .step = snapshot.step,
            .f_rep = snapshot.f_rep,
            .within_distance = snapshot.within_distance,
            .total_distance = snapshot.total_distance,
            .mean_payoff = snapshot.mean_payoff,
            .adj_payoff = snapshot.adj_payoff,
            .mean_max_depth = snapshot.mean_max_depth,
            .mean_depth = snapshot.mean_depth,
            .eff_column = snapshot.eff_column,
            .top_col_mass = snapshot.top_col_mass,
            .mean_rep_size = snapshot.mean_rep_size,
            .empty_rep_size = snapshot.empty_rep_size
        });
    }

    return rows;
}

std::vector<TraitRow> collect_trait_rows(const im::EquilibriumRunMetadata& meta,
                                         const im::PopulationState& equilibrium_state,
                                         const im::ReachableStates& states) {
    const auto frequencies =
        im::DifferentiationMetrics::trait_frequencies(equilibrium_state, states);
    const auto& lattice = states.lattice();
    std::vector<TraitRow> rows;
    rows.reserve(frequencies.island_count * frequencies.trait_count);

    for (IslandId island = 0; island < frequencies.island_count; ++island) {
        for (TraitId trait = 0; trait < frequencies.trait_count; ++trait) {
            const auto pos = lattice.pos(trait);
            rows.push_back(TraitRow{
                .run_id = meta.run_id,
                .seed = meta.seed,
                .columns = meta.columns,
                .layers = meta.layers,
                .cross_column_depth = meta.cross_column_depth,
                .island_count = meta.island_count,
                .m = meta.m,
                .rho = meta.rho,
                .mu = meta.mu,
                .alpha = meta.alpha,
                .beta = meta.beta,
                .gamma = meta.gamma,
                .eta = meta.eta,
                .delta = meta.delta,
                .sigma_b = meta.sigma_b,
                .sigma_nu = meta.sigma_nu,
                .k = meta.k,
                .converged = meta.converged,
                .steps_to_equilibrium = meta.steps_to_equilibrium,
                .final_distance = meta.final_distance,
                .island = static_cast<std::size_t>(island),
                .trait_id = static_cast<std::size_t>(trait),
                .trait_column = static_cast<std::size_t>(pos.column),
                .trait_layer = static_cast<std::size_t>(pos.layer),
                .is_base_layer = static_cast<std::uint32_t>(pos.layer == 0 ? 1 : 0),
                .frequency = frequencies(island, trait)
            });
        }
    }

    return rows;
}

Rcpp::DataFrame equilibrium_df_from_rows(const std::vector<EquilibriumRow>& rows) {
    const auto n = static_cast<R_xlen_t>(rows.size());

    Rcpp::NumericVector run_id(n), seed(n);
    Rcpp::IntegerVector columns(n), layers(n), cross_column_depth(n), island_count(n), converged(n), island(n),
        repertoire_id(n), repertoire_size(n), max_layer(n), steps_to_equilibrium(n);
    Rcpp::NumericVector m(n), rho(n), mu(n), alpha(n), beta(n), gamma(n), eta(n), delta(n), sigma_b(n), sigma_nu(n), k(n),
        final_distance(n), frequency(n), repertoire_payoff_sum(n), repertoire_payoff_mean(n),
        repertoire_payoff_max(n), f_rep_run(n), within_distance_run(n), total_distance_run(n);

    for (R_xlen_t i = 0; i < n; ++i) {
        const auto& row = rows[static_cast<std::size_t>(i)];
        run_id[i] = static_cast<double>(row.run_id);
        seed[i] = static_cast<double>(row.seed);
        columns[i] = static_cast<int>(row.columns);
        layers[i] = static_cast<int>(row.layers);
        cross_column_depth[i] = static_cast<int>(row.cross_column_depth);
        island_count[i] = static_cast<int>(row.island_count);
        m[i] = row.m;
        rho[i] = row.rho;
        mu[i] = row.mu;
        alpha[i] = row.alpha;
        beta[i] = row.beta;
        gamma[i] = row.gamma;
        eta[i] = row.eta;
        delta[i] = row.delta;
        sigma_b[i] = row.sigma_b;
        sigma_nu[i] = row.sigma_nu;
        k[i] = row.k;
        converged[i] = static_cast<int>(row.converged);
        steps_to_equilibrium[i] = static_cast<int>(row.steps_to_equilibrium);
        final_distance[i] = row.final_distance;
        island[i] = static_cast<int>(row.island);
        repertoire_id[i] = static_cast<int>(row.repertoire_id);
        frequency[i] = row.frequency;
        repertoire_size[i] = static_cast<int>(row.repertoire_size);
        max_layer[i] = static_cast<int>(row.max_layer);
        repertoire_payoff_sum[i] = row.repertoire_payoff_sum;
        repertoire_payoff_mean[i] = row.repertoire_payoff_mean;
        repertoire_payoff_max[i] = row.repertoire_payoff_max;
        f_rep_run[i] = row.f_rep_run;
        within_distance_run[i] = row.within_distance_run;
        total_distance_run[i] = row.total_distance_run;
    }

    Rcpp::List out(31);
    out[0] = run_id;
    out[1] = seed;
    out[2] = columns;
    out[3] = layers;
    out[4] = cross_column_depth;
    out[5] = island_count;
    out[6] = m;
    out[7] = rho;
    out[8] = mu;
    out[9] = alpha;
    out[10] = beta;
    out[11] = gamma;
    out[12] = eta;
    out[13] = delta;
    out[14] = sigma_b;
    out[15] = sigma_nu;
    out[16] = k;
    out[17] = converged;
    out[18] = steps_to_equilibrium;
    out[19] = final_distance;
    out[20] = island;
    out[21] = repertoire_id;
    out[22] = frequency;
    out[23] = repertoire_size;
    out[24] = max_layer;
    out[25] = repertoire_payoff_sum;
    out[26] = repertoire_payoff_mean;
    out[27] = repertoire_payoff_max;
    out[28] = f_rep_run;
    out[29] = within_distance_run;
    out[30] = total_distance_run;
    Rcpp::CharacterVector names(31);
    names[0] = "run_id";
    names[1] = "seed";
    names[2] = "columns";
    names[3] = "layers";
    names[4] = "cross_column_depth";
    names[5] = "island_count";
    names[6] = "m";
    names[7] = "rho";
    names[8] = "mu";
    names[9] = "alpha";
    names[10] = "beta";
    names[11] = "gamma";
    names[12] = "eta";
    names[13] = "delta";
    names[14] = "sigma_b";
    names[15] = "sigma_nu";
    names[16] = "k";
    names[17] = "converged";
    names[18] = "steps_to_equilibrium";
    names[19] = "final_distance";
    names[20] = "island";
    names[21] = "repertoire_id";
    names[22] = "frequency";
    names[23] = "repertoire_size";
    names[24] = "max_layer";
    names[25] = "repertoire_payoff_sum";
    names[26] = "repertoire_payoff_mean";
    names[27] = "repertoire_payoff_max";
    names[28] = "f_rep_run";
    names[29] = "within_distance_run";
    names[30] = "total_distance_run";
    out.attr("names") = names;
    out.attr("class") = "data.frame";
    out.attr("row.names") = Rcpp::IntegerVector::create(NA_INTEGER, -n);
    return Rcpp::DataFrame(out);
}

Rcpp::DataFrame traits_df_from_rows(const std::vector<TraitRow>& rows) {
    const auto n = static_cast<R_xlen_t>(rows.size());

    Rcpp::NumericVector run_id(n), seed(n);
    Rcpp::IntegerVector columns(n), layers(n), cross_column_depth(n), island_count(n), converged(n),
        steps_to_equilibrium(n), island(n), trait_id(n), trait_column(n), trait_layer(n), is_base_layer(n);
    Rcpp::NumericVector m(n), rho(n), mu(n), alpha(n), beta(n), gamma(n), eta(n), delta(n), sigma_b(n), sigma_nu(n), k(n),
        final_distance(n), frequency(n);

    for (R_xlen_t i = 0; i < n; ++i) {
        const auto& row = rows[static_cast<std::size_t>(i)];
        run_id[i] = static_cast<double>(row.run_id);
        seed[i] = static_cast<double>(row.seed);
        columns[i] = static_cast<int>(row.columns);
        layers[i] = static_cast<int>(row.layers);
        cross_column_depth[i] = static_cast<int>(row.cross_column_depth);
        island_count[i] = static_cast<int>(row.island_count);
        m[i] = row.m;
        rho[i] = row.rho;
        mu[i] = row.mu;
        alpha[i] = row.alpha;
        beta[i] = row.beta;
        gamma[i] = row.gamma;
        eta[i] = row.eta;
        delta[i] = row.delta;
        sigma_b[i] = row.sigma_b;
        sigma_nu[i] = row.sigma_nu;
        k[i] = row.k;
        converged[i] = static_cast<int>(row.converged);
        steps_to_equilibrium[i] = static_cast<int>(row.steps_to_equilibrium);
        final_distance[i] = row.final_distance;
        island[i] = static_cast<int>(row.island);
        trait_id[i] = static_cast<int>(row.trait_id);
        trait_column[i] = static_cast<int>(row.trait_column);
        trait_layer[i] = static_cast<int>(row.trait_layer);
        is_base_layer[i] = static_cast<int>(row.is_base_layer);
        frequency[i] = row.frequency;
    }

    Rcpp::List out(26);
    out[0] = run_id;
    out[1] = seed;
    out[2] = columns;
    out[3] = layers;
    out[4] = cross_column_depth;
    out[5] = island_count;
    out[6] = m;
    out[7] = rho;
    out[8] = mu;
    out[9] = alpha;
    out[10] = beta;
    out[11] = gamma;
    out[12] = eta;
    out[13] = delta;
    out[14] = sigma_b;
    out[15] = sigma_nu;
    out[16] = k;
    out[17] = converged;
    out[18] = steps_to_equilibrium;
    out[19] = final_distance;
    out[20] = island;
    out[21] = trait_id;
    out[22] = trait_column;
    out[23] = trait_layer;
    out[24] = is_base_layer;
    out[25] = frequency;
    Rcpp::CharacterVector names(26);
    names[0] = "run_id";
    names[1] = "seed";
    names[2] = "columns";
    names[3] = "layers";
    names[4] = "cross_column_depth";
    names[5] = "island_count";
    names[6] = "m";
    names[7] = "rho";
    names[8] = "mu";
    names[9] = "alpha";
    names[10] = "beta";
    names[11] = "gamma";
    names[12] = "eta";
    names[13] = "delta";
    names[14] = "sigma_b";
    names[15] = "sigma_nu";
    names[16] = "k";
    names[17] = "converged";
    names[18] = "steps_to_equilibrium";
    names[19] = "final_distance";
    names[20] = "island";
    names[21] = "trait_id";
    names[22] = "trait_column";
    names[23] = "trait_layer";
    names[24] = "is_base_layer";
    names[25] = "frequency";
    out.attr("names") = names;
    out.attr("class") = "data.frame";
    out.attr("row.names") = Rcpp::IntegerVector::create(NA_INTEGER, -n);
    return Rcpp::DataFrame(out);
}

Rcpp::DataFrame time_df_from_rows(const std::vector<TimeRow>& rows) {
    const auto n = static_cast<R_xlen_t>(rows.size());

    Rcpp::NumericVector run_id(n), seed(n);
    Rcpp::IntegerVector columns(n), layers(n), cross_column_depth(n), island_count(n), converged(n),
        steps_to_equilibrium(n), step(n);
    Rcpp::NumericVector m(n), rho(n), mu(n), alpha(n), beta(n), gamma(n), eta(n), delta(n), sigma_b(n), sigma_nu(n), k(n),
        final_distance(n), f_rep(n), within_distance(n), total_distance(n), mean_payoff(n), adj_payoff(n),
        mean_max_depth(n), mean_depth(n), eff_column(n),
        top_col_mass(n), mean_rep_size(n), empty_rep_size(n);

    for (R_xlen_t i = 0; i < n; ++i) {
        const auto& row = rows[static_cast<std::size_t>(i)];
        run_id[i] = static_cast<double>(row.run_id);
        seed[i] = static_cast<double>(row.seed);
        columns[i] = static_cast<int>(row.columns);
        layers[i] = static_cast<int>(row.layers);
        cross_column_depth[i] = static_cast<int>(row.cross_column_depth);
        island_count[i] = static_cast<int>(row.island_count);
        m[i] = row.m;
        rho[i] = row.rho;
        mu[i] = row.mu;
        alpha[i] = row.alpha;
        beta[i] = row.beta;
        gamma[i] = row.gamma;
        eta[i] = row.eta;
        delta[i] = row.delta;
        sigma_b[i] = row.sigma_b;
        sigma_nu[i] = row.sigma_nu;
        k[i] = row.k;
        converged[i] = static_cast<int>(row.converged);
        steps_to_equilibrium[i] = static_cast<int>(row.steps_to_equilibrium);
        final_distance[i] = row.final_distance;
        step[i] = static_cast<int>(row.step);
        f_rep[i] = row.f_rep;
        within_distance[i] = row.within_distance;
        total_distance[i] = row.total_distance;
        mean_payoff[i] = row.mean_payoff;
        adj_payoff[i] = row.adj_payoff;
        mean_max_depth[i] = row.mean_max_depth;
        mean_depth[i] = row.mean_depth;
        eff_column[i] = row.eff_column;
        top_col_mass[i] = row.top_col_mass;
        mean_rep_size[i] = row.mean_rep_size;
        empty_rep_size[i] = row.empty_rep_size;
    }

    Rcpp::List out(32);
    out[0] = run_id;
    out[1] = seed;
    out[2] = columns;
    out[3] = layers;
    out[4] = cross_column_depth;
    out[5] = island_count;
    out[6] = m;
    out[7] = rho;
    out[8] = mu;
    out[9] = alpha;
    out[10] = beta;
    out[11] = gamma;
    out[12] = eta;
    out[13] = delta;
    out[14] = sigma_b;
    out[15] = sigma_nu;
    out[16] = k;
    out[17] = converged;
    out[18] = steps_to_equilibrium;
    out[19] = final_distance;
    out[20] = step;
    out[21] = f_rep;
    out[22] = within_distance;
    out[23] = total_distance;
    out[24] = mean_payoff;
    out[25] = adj_payoff;
    out[26] = mean_max_depth;
    out[27] = mean_depth;
    out[28] = eff_column;
    out[29] = top_col_mass;
    out[30] = mean_rep_size;
    out[31] = empty_rep_size;
    Rcpp::CharacterVector names(32);
    names[0] = "run_id";
    names[1] = "seed";
    names[2] = "columns";
    names[3] = "layers";
    names[4] = "cross_column_depth";
    names[5] = "island_count";
    names[6] = "m";
    names[7] = "rho";
    names[8] = "mu";
    names[9] = "alpha";
    names[10] = "beta";
    names[11] = "gamma";
    names[12] = "eta";
    names[13] = "delta";
    names[14] = "sigma_b";
    names[15] = "sigma_nu";
    names[16] = "k";
    names[17] = "converged";
    names[18] = "steps_to_equilibrium";
    names[19] = "final_distance";
    names[20] = "step";
    names[21] = "f_rep";
    names[22] = "within_distance";
    names[23] = "total_distance";
    names[24] = "mean_payoff";
    names[25] = "adj_payoff";
    names[26] = "mean_max_depth";
    names[27] = "mean_depth";
    names[28] = "eff_column";
    names[29] = "top_col_mass";
    names[30] = "mean_rep_size";
    names[31] = "empty_rep_size";
    out.attr("names") = names;
    out.attr("class") = "data.frame";
    out.attr("row.names") = Rcpp::IntegerVector::create(NA_INTEGER, -n);
    return Rcpp::DataFrame(out);
}

Rcpp::DataFrame error_df_from_rows(const std::vector<ErrorRow>& rows) {
    const auto n = static_cast<R_xlen_t>(rows.size());
    Rcpp::NumericVector run_id(n);
    Rcpp::CharacterVector message(n);

    for (R_xlen_t i = 0; i < n; ++i) {
        const auto& row = rows[static_cast<std::size_t>(i)];
        run_id[i] = static_cast<double>(row.run_id);
        message[i] = row.message;
    }

    Rcpp::List out(2);
    out[0] = run_id;
    out[1] = message;
    Rcpp::CharacterVector names(2);
    names[0] = "run_id";
    names[1] = "message";
    out.attr("names") = names;
    out.attr("class") = "data.frame";
    out.attr("row.names") = Rcpp::IntegerVector::create(NA_INTEGER, -n);
    return Rcpp::DataFrame(out);
}

} // namespace

// [[Rcpp::export]]
Rcpp::List im_lattice_structure_cpp(int columns, int layers, int cross_column_depth = -1) {
    const im::Lattice lattice(
        static_cast<std::size_t>(columns),
        static_cast<std::size_t>(layers),
        resolve_cross_column_depth(cross_column_depth, layers)
    );

    return Rcpp::List::create(
        Rcpp::Named("columns") = columns,
        Rcpp::Named("layers") = layers,
        Rcpp::Named("cross_column_depth") = static_cast<int>(lattice.cross_column_depth()),
        Rcpp::Named("trait_count") = static_cast<int>(lattice.trait_count()),
        Rcpp::Named("vertices") = lattice_vertices_df(lattice),
        Rcpp::Named("edges") = lattice_edges_df(lattice)
    );
}

// [[Rcpp::export]]
Rcpp::List im_payoff_landscape_cpp(int columns,
                                   int layers,
                                   int island_count,
                                   double delta,
                                   double sigma_b,
                                   double sigma_nu,
                                   double k,
                                   double seed,
                                   int cross_column_depth = -1) {
    const im::Lattice lattice(
        static_cast<std::size_t>(columns),
        static_cast<std::size_t>(layers),
        resolve_cross_column_depth(cross_column_depth, layers)
    );

    const im::PayoffParams params{
        .delta = delta,
        .sigma_b = sigma_b,
        .sigma_nu = sigma_nu,
        .k = k
    };

    std::mt19937_64 rng(static_cast<std::uint64_t>(seed));
    const im::PayoffLandscape payoff(
        lattice,
        static_cast<std::size_t>(island_count),
        params,
        rng
    );

    Rcpp::NumericMatrix payoff_matrix(island_count, static_cast<int>(lattice.trait_count()));
    for (int island = 0; island < island_count; ++island) {
        for (TraitId trait = 0; trait < lattice.trait_count(); ++trait) {
            payoff_matrix(island, static_cast<int>(trait)) = payoff(
                static_cast<IslandId>(island),
                trait
            );
        }
    }

    Rcpp::CharacterVector row_names(island_count);
    for (int island = 0; island < island_count; ++island) {
        row_names[island] = "island_" + std::to_string(island + 1);
    }
    payoff_matrix.attr("dimnames") = Rcpp::List::create(
        row_names,
        R_NilValue
    );

    return Rcpp::List::create(
        Rcpp::Named("columns") = columns,
        Rcpp::Named("layers") = layers,
        Rcpp::Named("cross_column_depth") = static_cast<int>(lattice.cross_column_depth()),
        Rcpp::Named("island_count") = island_count,
        Rcpp::Named("vertices") = lattice_vertices_df(lattice),
        Rcpp::Named("edges") = lattice_edges_df(lattice),
        Rcpp::Named("payoffs") = payoff_matrix
    );
}

// [[Rcpp::export]]
Rcpp::List im_reachable_states_cpp(int columns, int layers, int cross_column_depth = -1) {
    const im::Lattice lattice(
        static_cast<std::size_t>(columns),
        static_cast<std::size_t>(layers),
        resolve_cross_column_depth(cross_column_depth, layers)
    );
    const im::ReachableStates states(lattice);

    return Rcpp::List::create(
        Rcpp::Named("columns") = columns,
        Rcpp::Named("layers") = layers,
        Rcpp::Named("cross_column_depth") = static_cast<int>(lattice.cross_column_depth()),
        Rcpp::Named("trait_count") = static_cast<int>(lattice.trait_count()),
        Rcpp::Named("state_count") = static_cast<int>(states.size()),
        Rcpp::Named("state_payload") = reachable_states_payload(states)
    );
}

// [[Rcpp::export]]
Rcpp::List im_run_model_cpp(Rcpp::DataFrame runs,
                            double tolerance = 1e-10,
                            int max_steps = 50000,
                            int bookkeeping_interval = 50,
                            double frequency_threshold = 1e-6,
                            Rcpp::Nullable<int> threads = R_NilValue) {
    if (max_steps <= 0) {
        throw std::invalid_argument("max_steps must be > 0");
    }
    if (bookkeeping_interval < 0) {
        throw std::invalid_argument("bookkeeping_interval must be >= 0");
    }
    if (tolerance < 0.0) {
        throw std::invalid_argument("tolerance must be >= 0");
    }
    if (frequency_threshold < 0.0) {
        throw std::invalid_argument("frequency_threshold must be >= 0");
    }

    const auto configs = run_configs_from_df(runs);
    std::unordered_map<LatticeConfigKey, reachable_states_ptr, LatticeConfigKeyHash> reachable_states_cache;
    reachable_states_cache.reserve(configs.size());

    for (const auto& cfg : configs) {
        const LatticeConfigKey key{
            .columns = cfg.columns,
            .layers = cfg.layers,
            .cross_column_depth = cfg.cross_column_depth
        };
        if (reachable_states_cache.contains(key)) {
            continue;
        }

        reachable_states_cache.emplace(
            key,
            std::make_shared<im::ReachableStates>(
                im::Lattice(cfg.columns, cfg.layers, cfg.cross_column_depth)
            )
        );
    }

    std::vector<RunOutcome> outcomes(configs.size());
    const bool parallelize_single_run_by_island =
        configs.size() == 1 && configs.front().m == 0.0;

    #ifdef _OPENMP
    const int old_threads = omp_get_max_threads();
    if (threads.isNotNull()) {
        const int thread_count = Rcpp::as<int>(threads);
        if (thread_count <= 0) {
            throw std::invalid_argument("threads must be > 0");
        }
        omp_set_num_threads(thread_count);
    }
    #endif

    #pragma omp parallel for schedule(dynamic) if(!parallelize_single_run_by_island)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(configs.size()); ++i) {
        const auto& cfg = configs[static_cast<std::size_t>(i)];
        auto& outcome = outcomes[static_cast<std::size_t>(i)];

        try {
            const LatticeConfigKey key{
                .columns = cfg.columns,
                .layers = cfg.layers,
                .cross_column_depth = cfg.cross_column_depth
            };
            const auto states_it = reachable_states_cache.find(key);
            if (states_it == reachable_states_cache.end()) {
                throw std::logic_error("Missing reachable states cache entry for lattice");
            }

            const auto& states = states_it->second;
            const im::Lattice lattice(cfg.columns, cfg.layers, cfg.cross_column_depth);
            const im::DynamicsParams dynamics{
                .m = cfg.m,
                .rho = cfg.rho,
                .mu = cfg.mu,
                .alpha = cfg.alpha,
                .beta = cfg.beta,
                .gamma = cfg.gamma
                ,
                .eta = cfg.eta
            };
            const im::PayoffParams payoff_params{
                .delta = cfg.delta,
                .sigma_b = cfg.sigma_b,
                .sigma_nu = cfg.sigma_nu,
                .k = cfg.k
            };

            std::mt19937_64 rng(cfg.seed);
            im::PayoffLandscape payoff(lattice, cfg.island_count, payoff_params, rng);
            im::Simulator simulator(states, payoff, dynamics);

            const auto eq = simulator.run_to_equilibrium(
                simulator.initial_state_all_empty(),
                static_cast<std::size_t>(max_steps),
                tolerance,
                0,
                {},
                static_cast<std::size_t>(bookkeeping_interval)
            );

            const auto meta = metadata_from_config(cfg, eq);
            outcome.equilibrium_rows =
                collect_equilibrium_rows(meta, eq.state, *states, payoff, frequency_threshold);
            outcome.time_rows = collect_time_rows(meta, eq.bookkeeping);
            outcome.trait_rows = collect_trait_rows(meta, eq.state, *states);
        } catch (const std::exception& e) {
            outcome.error = e.what();
        }
    }

    #ifdef _OPENMP
    if (threads.isNotNull()) {
        omp_set_num_threads(old_threads);
    }
    #endif

    std::vector<EquilibriumRow> equilibrium_rows;
    std::vector<TimeRow> time_rows;
    std::vector<TraitRow> trait_rows;
    std::vector<ErrorRow> error_rows;

    for (std::size_t i = 0; i < outcomes.size(); ++i) {
        auto& outcome = outcomes[i];
        if (!outcome.error.empty()) {
            error_rows.push_back(ErrorRow{
                .run_id = configs[i].run_id,
                .message = outcome.error
            });
            continue;
        }

        equilibrium_rows.insert(
            equilibrium_rows.end(),
            outcome.equilibrium_rows.begin(),
            outcome.equilibrium_rows.end()
        );
        time_rows.insert(
            time_rows.end(),
            outcome.time_rows.begin(),
            outcome.time_rows.end()
        );
        trait_rows.insert(
            trait_rows.end(),
            outcome.trait_rows.begin(),
            outcome.trait_rows.end()
        );
    }

    return Rcpp::List::create(
        Rcpp::Named("equilibrium") = equilibrium_df_from_rows(equilibrium_rows),
        Rcpp::Named("traits") = traits_df_from_rows(trait_rows),
        Rcpp::Named("time") = time_df_from_rows(time_rows),
        Rcpp::Named("errors") = error_df_from_rows(error_rows)
    );
}
