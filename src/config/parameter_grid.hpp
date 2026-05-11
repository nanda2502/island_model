#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace island_model {

struct RunConfig {
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
};

struct ParameterGrid {
  std::vector<std::size_t> columns;
  std::vector<std::size_t> layers;
  std::vector<std::size_t> cross_column_depths;
  std::vector<std::size_t> island_counts;
  std::vector<std::uint64_t> seeds;
  std::vector<double> strictnesses;

  std::vector<double> ms;
  std::vector<double> rhos;
  std::vector<double> mus;
  std::vector<double> alphas;
  std::vector<double> betas;
  std::vector<double> lambdas;
  std::vector<double> gammas;
  std::vector<double> etas;

  std::vector<double> deltas;
  std::vector<double> sigma_bs;
  std::vector<double> sigma_nus;
  std::vector<double> ks;
};

inline void validate_grid(const ParameterGrid &grid) {
  if (grid.columns.empty())
    throw std::invalid_argument("ParameterGrid: columns is empty");
  if (grid.layers.empty())
    throw std::invalid_argument("ParameterGrid: layers is empty");
  if (grid.cross_column_depths.empty())
    throw std::invalid_argument("ParameterGrid: cross_column_depths is empty");
  if (grid.island_counts.empty())
    throw std::invalid_argument("ParameterGrid: island_counts is empty");
  if (grid.seeds.empty())
    throw std::invalid_argument("ParameterGrid: seeds is empty");
  if (grid.strictnesses.empty())
    throw std::invalid_argument("ParameterGrid: strictnesses is empty");

  if (grid.ms.empty())
    throw std::invalid_argument("ParameterGrid: ms is empty");
  if (grid.rhos.empty())
    throw std::invalid_argument("ParameterGrid: rhos is empty");
  if (grid.mus.empty())
    throw std::invalid_argument("ParameterGrid: mus is empty");
  if (grid.alphas.empty())
    throw std::invalid_argument("ParameterGrid: alphas is empty");
  if (grid.betas.empty())
    throw std::invalid_argument("ParameterGrid: betas is empty");
  if (grid.lambdas.empty())
    throw std::invalid_argument("ParameterGrid: lambdas is empty");
  if (grid.gammas.empty())
    throw std::invalid_argument("ParameterGrid: gammas is empty");
  if (grid.etas.empty())
    throw std::invalid_argument("ParameterGrid: etas is empty");

  if (grid.deltas.empty())
    throw std::invalid_argument("ParameterGrid: deltas is empty");
  if (grid.sigma_bs.empty())
    throw std::invalid_argument("ParameterGrid: sigma_bs is empty");
  if (grid.sigma_nus.empty())
    throw std::invalid_argument("ParameterGrid: sigma_nus is empty");
  if (grid.ks.empty())
    throw std::invalid_argument("ParameterGrid: ks is empty");
}

[[nodiscard]] inline bool is_valid_run_config(const RunConfig &cfg) {
  if (cfg.columns == 0 || cfg.layers == 0 || cfg.cross_column_depth == 0 ||
      cfg.island_count == 0) {
    return false;
  }
  if (cfg.cross_column_depth != 1)
    return false;
  if (cfg.strictness != 0.0 && cfg.strictness != 1.0)
    return false;
  if (cfg.m < 0.0 || cfg.m > 1.0)
    return false;
  if (cfg.rho < 0.0 || cfg.rho > 1.0)
    return false;
  if (cfg.mu < 0.0 || cfg.mu > 1.0)
    return false;
  if (cfg.alpha < 0.0)
    return false;
  if (cfg.beta < 0.0)
    return false;
  if (cfg.lambda < 0.0)
    return false;
  if (cfg.gamma < 0.0)
    return false;
  if (cfg.eta < 0.0 || cfg.eta > 1.0)
    return false;
  if (cfg.sigma_b < 0.0)
    return false;
  if (cfg.sigma_nu < 0.0)
    return false;
  if (cfg.k < 0.0 || cfg.k > 1.0)
    return false;

  return true;
}

[[nodiscard]] inline std::vector<RunConfig>
make_parameter_combinations(const ParameterGrid &grid) {
  validate_grid(grid);
  std::vector<RunConfig> combinations;
  std::uint64_t run_id = 0;
  for (const auto columns : grid.columns) {
    for (const auto layers : grid.layers) {
      for (const auto cross_column_depth : grid.cross_column_depths) {
        for (const auto island_count : grid.island_counts) {
          for (const auto seed : grid.seeds) {
            for (const auto strictness : grid.strictnesses) {
              for (const auto m : grid.ms) {
                for (const auto rho : grid.rhos) {
                  for (const auto mu : grid.mus) {
                    for (const auto alpha : grid.alphas) {
                      for (const auto beta : grid.betas) {
                        for (const auto lambda : grid.lambdas) {
                          for (const auto gamma : grid.gammas) {
                            for (const auto eta : grid.etas) {
                              for (const auto delta : grid.deltas) {
                                for (const auto sigma_b : grid.sigma_bs) {
                                  for (const auto sigma_nu : grid.sigma_nus) {
                                    for (const auto k : grid.ks) {
                                      RunConfig cfg{.run_id = run_id,
                                                    .seed = seed,
                                                    .columns = columns,
                                                    .layers = layers,
                                                    .cross_column_depth =
                                                        cross_column_depth,
                                                    .island_count = island_count,
                                                    .strictness = strictness,
                                                    .m = m,
                                                    .rho = rho,
                                                    .mu = mu,
                                                    .alpha = alpha,
                                                    .beta = beta,
                                                    .lambda = lambda,
                                                    .gamma = gamma,
                                                    .eta = eta,
                                                    .delta = delta,
                                                    .sigma_b = sigma_b,
                                                    .sigma_nu = sigma_nu,
                                                    .k = k};
                                      if (!is_valid_run_config(cfg)) {
                                        continue;
                                      }
                                      combinations.push_back(cfg);
                                      ++run_id;
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return combinations;
}

} // namespace island_model
