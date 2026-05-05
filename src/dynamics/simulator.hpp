#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>

#include "../analysis/convergence.hpp"
#include "../analysis/time_bookkeeping.hpp"
#include "../dynamics/migration.hpp"
#include "../dynamics/transition_kernel.hpp"
#include "../dynamics/visibility.hpp"
#include "../dynamics/weights.hpp"
#include "../model/payoff_landscape.hpp"
#include "../state/population_state.hpp"
#include "../state_space/reachable_states.hpp"

namespace island_model {

struct DynamicsParams {
    double m{0.0};
    double rho{0.0};
    double mu{0.0};
    double alpha{0.0};
    double beta{0.0};
    double gamma{0.0};
    double eta{0.0};
};

class Simulator {
public:
    using reachable_states_type = ReachableStates;
    using population_state_type = PopulationState;
    using payoff_landscape_type = PayoffLandscape;
    using visibility_field_type = VisibilityField;
    using weight_field_type = WeightField;

    struct StepResult {
        population_state_type migrated_state;
        visibility_field_type visibility;
        weight_field_type weights;
        population_state_type next_state;
    };

    struct EquilibriumResult {
        population_state_type state;
        std::size_t steps;
        double final_distance;
        bool converged;
        std::vector<PopulationBookkeepingSnapshot> bookkeeping;
    };

    Simulator(std::shared_ptr<const reachable_states_type> states,
              payoff_landscape_type payoff,
              DynamicsParams params)
        : states_(std::move(states)),
          payoff_(std::move(payoff)),
          params_(params) {
        if (!states_) {
            throw std::invalid_argument("Simulator: states pointer must not be null");
        }
        validate_params(params_);
        if (payoff_.island_count() == 0) {
            throw std::invalid_argument("Simulator: island count must be > 0");
        }
        if (states_->lattice() != payoff_.lattice()) {
            throw std::invalid_argument("Simulator: state space/payoff lattice mismatch");
        }
    }

    Simulator(reachable_states_type states,
              payoff_landscape_type payoff,
              DynamicsParams params)
        : Simulator(std::make_shared<reachable_states_type>(std::move(states)),
                    std::move(payoff),
                    params) {}

    [[nodiscard]] const reachable_states_type& states() const noexcept {
        return *states_;
    }

    [[nodiscard]] const payoff_landscape_type& payoff() const noexcept {
        return payoff_;
    }

    [[nodiscard]] const DynamicsParams& params() const noexcept {
        return params_;
    }

    [[nodiscard]] population_state_type initial_state_all_empty() const {
        population_state_type state(payoff_.island_count(), states());
        state.set_all_empty(states());
        return state;
    }

    [[nodiscard]] StepResult step(const population_state_type& current_state) const {
        validate_state(current_state);

        auto migrated_state =
            Migration::apply(current_state, params_.m);

        auto visibility =
            Visibility::compute(
                migrated_state,
                states(),
                payoff_,
                params_.beta);

        auto weights =
            Weights::compute(
                visibility,
                payoff_,
                params_.alpha,
                params_.gamma);

        auto next_state =
            TransitionKernel::apply(
                migrated_state,
                states(),
                weights,
                params_.rho,
                params_.mu,
                params_.eta);

        return StepResult{
            .migrated_state = migrated_state,
            .visibility = visibility,
            .weights = weights,
            .next_state = next_state
        };
    }

    [[nodiscard]] population_state_type step_state_only(const population_state_type& current_state) const {
        return step(current_state).next_state;
    }

    [[nodiscard]] EquilibriumResult run_to_equilibrium(population_state_type initial_state,
                                                       std::size_t max_steps,
                                                       double tolerance,
                                                       std::size_t log_interval = 0,
                                                       const std::function<void(std::size_t)>& progress_callback = {},
                                                       std::size_t bookkeeping_interval = 0) const {
        validate_state(initial_state);

        population_state_type current = std::move(initial_state);
        std::vector<PopulationBookkeepingSnapshot> bookkeeping;

        for (std::size_t step = 1; step <= max_steps; ++step) {
            population_state_type next = step_state_only(current);
            const double distance =
                Convergence::max_island_l1(current, next);

            maybe_record_bookkeeping(bookkeeping, bookkeeping_interval, step, next);

            if (progress_callback && log_interval != 0 && step % log_interval == 0) {
                progress_callback(step);
            }

            if (distance < tolerance) {
                ensure_final_bookkeeping(bookkeeping, step, next);
                return EquilibriumResult{
                    .state = next,
                    .steps = step,
                    .final_distance = distance,
                    .converged = true,
                    .bookkeeping = std::move(bookkeeping)
                };
            }

            current = std::move(next);
        }

        const population_state_type next = step_state_only(current);
        const double distance =
            Convergence::max_island_l1(current, next);

        ensure_final_bookkeeping(bookkeeping, max_steps, current);

        return EquilibriumResult{
            .state = current,
            .steps = max_steps,
            .final_distance = distance,
            .converged = false,
            .bookkeeping = std::move(bookkeeping)
        };
    }

private:
    void maybe_record_bookkeeping(std::vector<PopulationBookkeepingSnapshot>& bookkeeping,
                                  std::size_t bookkeeping_interval,
                                  std::size_t step,
                                  const population_state_type& state) const {
        if (bookkeeping_interval == 0 || step % bookkeeping_interval != 0) {
            return;
        }

        bookkeeping.push_back(
            PopulationBookkeeping::summarize(step, state, states(), payoff_));
    }

    void ensure_final_bookkeeping(std::vector<PopulationBookkeepingSnapshot>& bookkeeping,
                                  std::size_t step,
                                  const population_state_type& state) const {
        if (!bookkeeping.empty() && bookkeeping.back().step == step) {
            return;
        }

        bookkeeping.push_back(
            PopulationBookkeeping::summarize(step, state, states(), payoff_));
    }

    static void validate_params(const DynamicsParams& params) {
        if (params.m < 0.0 || params.m > 1.0) {
            throw std::invalid_argument("Simulator: m must be in [0, 1]");
        }
        if (params.rho < 0.0 || params.rho > 1.0) {
            throw std::invalid_argument("Simulator: rho must be in [0, 1]");
        }
        if (params.mu < 0.0 || params.mu > 1.0) {
            throw std::invalid_argument("Simulator: mu must be in [0, 1]");
        }
        if (params.alpha < 0.0) {
            throw std::invalid_argument("Simulator: alpha must be >= 0");
        }
        if (params.beta < 0.0) {
            throw std::invalid_argument("Simulator: beta must be >= 0");
        }
        if (params.gamma < 0.0) {
            throw std::invalid_argument("Simulator: gamma must be >= 0");
        }
        if (params.eta < 0.0 || params.eta > 1.0) {
            throw std::invalid_argument("Simulator: eta must be in [0, 1]");
        }
    }

    void validate_state(const population_state_type& state) const {
        if (state.island_count() != payoff_.island_count()) {
            throw std::invalid_argument("Simulator: state/payoff island count mismatch");
        }
        if (state.state_count() != states_->size()) {
            throw std::invalid_argument("Simulator: state/state-space size mismatch");
        }
    }

    std::shared_ptr<const reachable_states_type> states_;
    payoff_landscape_type payoff_;
    DynamicsParams params_;
};

} // namespace island_model
