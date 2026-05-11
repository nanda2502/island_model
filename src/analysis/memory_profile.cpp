#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <fstream>
#include <unistd.h>
#endif

#include "../dynamics/simulator.hpp"
#include "../model/lattice.hpp"
#include "../model/payoff_landscape.hpp"
#include "../state_space/reachable_states.hpp"

namespace im = island_model;

namespace {

struct CaseConfig {
    std::size_t columns{0};
    std::size_t layers{0};
    std::size_t cross_column_depth{0};
    std::size_t island_count{20};
    double strictness{1.0};
    std::size_t steps{1};
};

struct MemorySnapshot {
    std::size_t working_set_bytes{0};
    std::size_t peak_working_set_bytes{0};
    std::size_t private_bytes{0};
};

[[nodiscard]] MemorySnapshot memory_snapshot() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (!GetProcessMemoryInfo(GetCurrentProcess(),
                              reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                              sizeof(counters))) {
        throw std::runtime_error("GetProcessMemoryInfo failed");
    }

    return MemorySnapshot{
        .working_set_bytes = static_cast<std::size_t>(counters.WorkingSetSize),
        .peak_working_set_bytes = static_cast<std::size_t>(counters.PeakWorkingSetSize),
        .private_bytes = static_cast<std::size_t>(counters.PrivateUsage)
    };
#else
    std::ifstream statm("/proc/self/statm");
    std::size_t size_pages = 0;
    std::size_t resident_pages = 0;
    statm >> size_pages >> resident_pages;
    const auto page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    return MemorySnapshot{
        .working_set_bytes = resident_pages * page_size,
        .peak_working_set_bytes = resident_pages * page_size,
        .private_bytes = size_pages * page_size
    };
#endif
}

[[nodiscard]] double mib(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

void print_snapshot(const std::string& label) {
    const auto mem = memory_snapshot();
    std::cout << std::left << std::setw(28) << label
              << " rss=" << std::right << std::setw(9) << std::fixed << std::setprecision(2)
              << mib(mem.working_set_bytes) << " MiB"
              << " peak_rss=" << std::setw(9) << mib(mem.peak_working_set_bytes) << " MiB"
              << " private=" << std::setw(9) << mib(mem.private_bytes) << " MiB"
              << '\n';
}

[[nodiscard]] std::string case_name(const CaseConfig& cfg) {
    std::ostringstream out;
    out << cfg.columns << "x" << cfg.layers
        << " depth=" << cfg.cross_column_depth
        << " islands=" << cfg.island_count;
    return out.str();
}

[[nodiscard]] std::vector<CaseConfig> default_cases() {
    return {
        {.columns = 4, .layers = 6, .cross_column_depth = 1, .island_count = 20, .steps = 2},
        {.columns = 5, .layers = 6, .cross_column_depth = 1, .island_count = 20, .steps = 2},
        {.columns = 6, .layers = 6, .cross_column_depth = 1, .island_count = 20, .steps = 2},
        {.columns = 4, .layers = 6, .cross_column_depth = 6, .island_count = 20, .steps = 2},
    };
}

[[nodiscard]] std::size_t parse_size(const char* value, const char* name) {
    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0) {
        throw std::invalid_argument(std::string("Invalid ") + name + ": " + value);
    }
    return static_cast<std::size_t>(parsed);
}

[[nodiscard]] std::vector<CaseConfig> parse_cases(int argc, char* argv[]) {
    if (argc == 1) {
        return default_cases();
    }
    if (argc != 5 && argc != 6 && argc != 7) {
        throw std::invalid_argument(
            "Usage: memory_profile [columns layers cross_column_depth island_count [steps [strictness]]]");
    }

    CaseConfig cfg{
        .columns = parse_size(argv[1], "columns"),
        .layers = parse_size(argv[2], "layers"),
        .cross_column_depth = parse_size(argv[3], "cross_column_depth"),
        .island_count = parse_size(argv[4], "island_count"),
        .steps = argc >= 6 ? parse_size(argv[5], "steps") : 2
    };
    if (argc == 7) {
        cfg.strictness = std::stod(argv[6]);
    }
    return {cfg};
}

void run_case(const CaseConfig& cfg) {
    std::cout << "\n== " << case_name(cfg) << " ==\n";
    const auto start = std::chrono::steady_clock::now();
    print_snapshot("start");

    const im::Lattice lattice(cfg.columns, cfg.layers, cfg.cross_column_depth);
    print_snapshot("after lattice");

    auto states = std::make_shared<im::ReachableStates>(lattice, cfg.strictness);
    std::cout << "reachable_states=" << states->size()
              << " traits=" << lattice.trait_count() << '\n';
    print_snapshot("after reachable states");

    im::PayoffParams payoff_params{
        .delta = 1.0,
        .sigma_b = 0.5,
        .sigma_nu = 0.1,
        .k = 0.25
    };
    std::mt19937_64 rng(12345);
    im::PayoffLandscape payoff(lattice, cfg.island_count, payoff_params, rng);
    print_snapshot("after payoff landscape");

    im::DynamicsParams dynamics{
        .m = 0.01,
        .rho = 0.05,
        .mu = 0.03,
        .alpha = 1.5,
        .beta = 1.0,
        .lambda = 0.0,
        .gamma = 0.0,
        .eta = 0.0
    };
    im::Simulator simulator(states, std::move(payoff), dynamics);
    print_snapshot("after simulator cache");

    auto state = simulator.initial_state_all_empty();
    print_snapshot("after initial state");

    for (std::size_t step = 1; step <= cfg.steps; ++step) {
        auto result = simulator.step(state);
        std::cout << "step=" << step << '\n';
        print_snapshot("after full step result");
        state = std::move(result.next_state);
        print_snapshot("after keeping next only");
    }

    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
    std::cout << "elapsed_seconds=" << std::fixed << std::setprecision(3) << elapsed.count() << '\n';
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        for (const auto& cfg : parse_cases(argc, argv)) {
            run_case(cfg);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "memory_profile: " << e.what() << '\n';
        return 1;
    }
}
