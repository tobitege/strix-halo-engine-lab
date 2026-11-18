#pragma once

// DS41 prefill adapter of the resident runner (strix/serve/resident_session.hpp): `strix-deepseek41-prefill ...
// --ttft-timing --plan FILE --plan-out DIR` loads the weights once and runs every arm of the plan (arm = a set of
// environment variables) in this process, with the agent's own binary. Each slot: the runner sets the arm's
// environment, bumps the lever generation (every strix::lever_cache resolves again), calls apply_levers() (the
// per-object switches of DevicePrefill), meets the peer at the barrier and times one DevicePrefill::run exactly as
// --ttft-timing does (barrier, then the wall of run() to the result on the host).
//
// Lever classes come from strix/deepseek41_prefill_levers.hpp; a name missing there is a reload lever, so an arm
// that changes it is refused ("needs reload") instead of being measured with the old value.
//
// Host-only template code (the engine is a template parameter), so the host tests drive it with a fake prefill.

#include "resident_tp2.hpp"
#include "strix/deepseek41_prefill_levers.hpp"
#include "strix/serve/resident_session.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::deepseek41::resident {
namespace rp = strix::serve::resident;

inline constexpr const char* kEngineName = "deepseek41-prefill";

[[nodiscard]] inline rp::LeverClass to_runner_class(PrefillLeverClass c) noexcept {
    switch (c) {
    case PrefillLeverClass::runtime: return rp::LeverClass::runtime;
    case PrefillLeverClass::construct: return rp::LeverClass::construct;
    case PrefillLeverClass::cached: return rp::LeverClass::cached;
    case PrefillLeverClass::reload: return rp::LeverClass::reload;
    }
    return rp::LeverClass::reload;
}

[[nodiscard]] inline rp::LeverTable lever_table() {
    rp::LeverTable t;
    for (const PrefillLever& l : kPrefillLevers) t.push_back(rp::LeverEntry{l.name, to_runner_class(l.cls)});
    return t;
}

// The prompt format of --prompt: comma-separated decimal ids (the app's read_prompt).
[[nodiscard]] inline std::vector<std::uint32_t> read_csv_prompt(const std::string& path, std::uint64_t rows = 0) {
    const std::string text = rp::read_text_file(path, rp::kMaxPromptBytes);
    std::vector<std::uint32_t> ids;
    std::stringstream ss(text);
    for (std::string t; std::getline(ss, t, ',');)
        if (!t.empty()) ids.push_back(static_cast<std::uint32_t>(std::stoul(t)));
    if (rows != 0) {
        if (rows > ids.size()) throw std::invalid_argument("prompt shorter than --rows");
        ids.resize(static_cast<std::size_t>(rows));
    }
    return ids;
}

// Node state beside every run (validity rule: clock, busy, MemAvailable). -1 = not readable here.
struct Telemetry {
    double mem_available_gib = -1.0;
    int sclk_mhz = -1;
    int busy_pct = -1;
};

[[nodiscard]] inline long long meminfo_field_kb(const char* key) {
    std::ifstream in("/proc/meminfo");
    for (std::string line; std::getline(in, line);)
        if (line.rfind(key, 0) == 0) return std::atoll(line.c_str() + std::char_traits<char>::length(key));
    return -1;
}

[[nodiscard]] inline Telemetry read_telemetry() {
    Telemetry t;
    if (const long long kb = meminfo_field_kb("MemAvailable:"); kb >= 0) t.mem_available_gib = static_cast<double>(kb) / 1048576.0;
    for (int card = 0; card < 4 && t.sclk_mhz < 0; ++card) {
        const std::string dev = "/sys/class/drm/card" + std::to_string(card) + "/device/";
        std::ifstream s(dev + "pp_dpm_sclk");
        for (std::string line; s && std::getline(s, line);)
            if (line.find('*') != std::string::npos) {
                const auto colon = line.find(':');
                if (colon != std::string::npos) t.sclk_mhz = std::atoi(line.c_str() + colon + 1);
            }
        if (t.sclk_mhz >= 0) {
            std::ifstream b(dev + "gpu_busy_percent");
            if (b) b >> t.busy_pct;
        }
    }
    return t;
}

// FNV-1a 64 over the raw bytes of the logits (the app's logits_fnv).
[[nodiscard]] inline std::uint64_t logits_fnv(const std::vector<float>& v) noexcept {
    return rp::fnv1a64_bytes(v.data(), v.size() * sizeof(float));
}

// What the adapter needs from the app (world 2: the peer exchange and the device barrier of --ttft-timing).
struct Wiring {
    unsigned rank = 0, world = 1;
    std::uint64_t prompt_max = 0;  // --rows: plan prompts are cut like --prompt
    std::uint64_t capacity = 0;    // rows the prefill was sized for (the --prompt rows)
    // Byte exchange with the peer (both ranks, same size); empty for world 1.
    std::function<void(const void* out, void* in, std::size_t n)> exchange;
    // The app's timed-run barrier (same one --ttft-timing uses); empty for world 1.
    std::function<void(std::uint32_t round)> barrier;
    bool print = true;
};

// Prefill: run(std::span<const std::uint32_t>) -> result with logits, next_token, seconds, engram_seconds,
// head_seconds; apply_levers().
template<class Prefill>
[[nodiscard]] rp::Hooks make_hooks(Prefill& pre, const Wiring& w) {
    rp::Hooks h;
    h.engine_names = {kEngineName};
    h.levers = lever_table();
    h.rank = w.rank;
    h.world = w.world;
    h.capacity = w.capacity;
    h.prompt_max = w.prompt_max;
    h.read_prompt = [rows = w.prompt_max](const std::string& path) { return read_csv_prompt(path, rows); };
    h.validate_arm = [](const rp::Arm& arm) {
        if (arm.steps != 1) throw rp::PlanError("deepseek41-prefill requires steps:1");
        // Check the throwing construct resolvers before any environment/object mutation.
        for (const auto& [key, value] : arm.env) {
            if (value.empty()) continue;
            if (key == "STRIX_DS41_PREFILL_FP16_PROJ") {
                std::stringstream ss(value);
                for (std::string t; std::getline(ss, t, ',');)
                    if (t != "q_a" && t != "kv" && t != "q_b" && t != "o_a" && t != "o_b")
                        throw rp::PlanError(key + ": " + t);
            } else if (key == "STRIX_DS41_PREFILL_APPROX_LAYERS") {
                char* end = nullptr;
                (void)std::strtoul(value.c_str(), &end, 10);
                if (end == nullptr || *end != '-') throw rp::PlanError(key + ": first-end");
            } else if (key == "STRIX_DS41_PREFILL_DENSE_MIN_ROWS") {
                (void)std::stoul(value);
            }
        }
    };
    h.validate_batch = [](const rp::BatchSpec& spec) {
        for (const auto& request : spec.requests)
            if (request.max_tokens != 1) throw rp::PlanError("deepseek41-prefill requires batch max_tokens:1");
    };
    if (w.world > 1) {
        if (!w.exchange || !w.barrier) throw std::invalid_argument("resident: world 2 needs the peer exchange and barrier");
        auto ex = w.exchange;
        h.agree = [ex](const std::array<std::uint32_t, 8>& f) { return rp::tp2::agree_bytes(f, ex); };
        auto bar = w.barrier;
        // the runner's slot check, then the same device barrier the fresh --ttft-timing run uses
        h.barrier = [ex, bar](std::uint64_t seq) {
            rp::tp2::barrier_bytes(seq, ex);
            bar(static_cast<std::uint32_t>(seq & 0x7fffffffU));
        };
    }
    h.apply_levers = [&pre] { pre.apply_levers(); };
    h.run = [&pre](const std::vector<std::uint32_t>& prompt, unsigned, const rp::LogitsSink* sink, bool) {
        // nothing between the barrier and the clock start (as --ttft-timing); node state is read after the run, and the
        // wall-clock window lets a 40 ms clock/busy sampler of the ticket give the window means
        const auto t0 = std::chrono::steady_clock::now();
        const auto e0 = std::chrono::system_clock::now();
        auto r = pre.run(std::span<const std::uint32_t>(prompt));
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double epoch0 = std::chrono::duration<double>(e0.time_since_epoch()).count();
        const Telemetry after = read_telemetry();
        if (sink != nullptr && *sink) (*sink)(r.logits.data(), r.logits.size());
        rp::RunIo io;
        io.tokens = {r.next_token};
        io.prefill_s = wall;
        char note[320];
        std::snprintf(note, sizeof(note),
                      "run %.3f ms, engram read %.3f ms, head %.3f ms, next token %u, logits_fnv %016llx, window %.3f+%.3f s, "
                      "after: sclk %d MHz, busy %d %%, MemAvailable %.2f GiB",
                      r.seconds * 1e3, r.engram_seconds * 1e3, r.head_seconds * 1e3, r.next_token,
                      static_cast<unsigned long long>(logits_fnv(r.logits)), epoch0, wall, after.sclk_mhz, after.busy_pct,
                      after.mem_available_gib);
        io.note = note;
        return io;
    };
    return h;
}

// One plan on the loaded prefill: rank 0 writes summary.json, every rank its receipts. Returns the outcome.
template<class Prefill>
rp::PlanOutcome run_plan_file(Prefill& pre, const Wiring& w, const std::string& plan_path, const std::string& out_dir,
                              double budget_s) {
    const rp::Plan plan = rp::parse_plan(rp::read_text_file(plan_path));
    rp::Hooks h = make_hooks(pre, w);
    rp::PlanOutcome oc = rp::run_plan(plan, out_dir, budget_s, h);
    rp::write_text_atomic(std::filesystem::path(out_dir) / ("status" + rp::rank_suffix(w.rank) + ".json"),
                          rp::outcome_json(oc, "plan", w.rank).dump());
    if (w.print)
        std::printf("rank %u RESIDENT done: plan %s, %zu runs, %zu errors, state %s\n", w.rank, oc.plan_hash.c_str(),
                    oc.runs, oc.errors, oc.state.c_str());
    std::fflush(stdout);
    return oc;
}

}  // namespace strix::deepseek41::resident
