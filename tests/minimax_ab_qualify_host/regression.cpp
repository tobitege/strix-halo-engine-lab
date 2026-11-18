#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/prefill_headless.hpp"

struct Engine {
    struct Impl { std::uint64_t pf_dense_mlp_calls = 0; } storage;
    Impl* impl_ = &storage;
    std::vector<std::uint64_t> deltas;
    std::size_t next = 0;
    unsigned rank_ = 0;
    bool enabled = false;
    std::uint64_t prefill_dense_mlp_count() const;
    // These dense-MLP scenarios never execute headless prefill.
    std::uint64_t prefill_headless_chunks() const { return 0; }
    std::size_t prefill_headless_expected(std::size_t) const { return 0; }
    unsigned rank() const { return rank_; }
    std::uint32_t prefill(int, const std::vector<std::uint32_t>&, int) {
        if (enabled != (next % 2 == 1)) throw std::runtime_error("wrong arm order");
        storage.pf_dense_mlp_calls += deltas.at(next++);
        return 42;
    }
    std::uint64_t state_fnv(int, std::size_t) const { return 17; }
    std::uint64_t prefill_out_fnv() const { return 19; }
};
#include "accessor.inc"

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

bool run(const std::string& scenario, unsigned rank) {
    Engine eng;
    eng.rank_ = rank;
    eng.storage.pf_dense_mlp_calls = scenario == "stale" ? 1000 : 0;
    eng.deltas = {0, 3, 0, 3, 0, 3};
    if (scenario == "fallback" || scenario == "stale" || scenario == "ordinary") eng.deltas = {0, 0, 0, 0, 0, 0};
    if (scenario == "contaminated") eng.deltas[2] = 1;
    if (scenario == "partial") eng.deltas[3] = 0;
    const std::string flag = scenario == "ordinary" ? "pfmsa" : "pfdensemlp";
    const int prefills = 6;
    const bool hash_state = true;
    std::vector<std::uint32_t> prompt(9, 1), first;
    std::vector<std::uint64_t> states, outs;
    std::vector<double> ttft[2];
    const auto set = [&](bool on) { eng.enabled = on; };
    bool g_exact_fail = false;
    // These scenarios never select --ab pfflashkeys, so production's FlashKeysAB is inactive (no-op).
    struct {
        void begin() {}
        void finish(int, std::uint64_t, std::uint64_t) const {}
    } flash_keys;
#include "prefill.inc"
#include "receipt.inc"
    if (eng.next != 6 || states.size() != 6 || outs.size() != 6 || first.size() != 6)
        throw std::runtime_error("prefill observations missing");
    return g_exact_fail;
}

struct DenseSite { std::string name; };
struct Gate {
    std::vector<DenseSite> dense{{"q"}, {"k"}, {"v"}, {"o"}, {"dense_gate"}, {"dense_up"}, {"dense_down"}};
#include "groups.inc"
};

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("case required");
        const std::string scenario = argv[1];
        if (scenario == "groups") {
            Gate gate;
            const auto groups = gate.dense_v4_groups();
            for (const std::vector<std::string>& names : std::vector<std::vector<std::string>>{
                     {"q"}, {"k", "v"}, {"o"}, {"q", "k", "v"},
                     {"dense_gate"}, {"dense_up"}, {"dense_down"}, {"dense_gate", "dense_up"}}) {
                const bool present = std::any_of(groups.begin(), groups.end(), [&](const auto& group) {
                    return group.size() == names.size() && std::equal(group.begin(), group.end(), names.begin(),
                        [](const DenseSite* site, const std::string& name) { return site->name == name; });
                });
                if (!present) throw std::runtime_error("missing dense-v4 operator group");
            }
        } else {
            const bool expected_failure = scenario != "covered" && scenario != "ordinary";
            for (unsigned rank : {0U, 1U})
                if (run(scenario, rank) != expected_failure) throw std::runtime_error("incorrect qualification outcome");
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
