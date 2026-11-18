#pragma once

#include <exception>

// Resident test server: the engine-neutral runner on top of resident_plan.hpp. An engine app loads its model
// once, fills a Hooks struct (lever table, rank agreement, arm switch, reset, one timed run) and then either
//   run_plan()    runs one plan (--resident-plan FILE), or
//   run_session() serves a stream of jobs a driver writes into a session directory (--resident-session DIR),
// so the load is paid once for every plan of the session. Nothing here touches a GPU; the engine adapters
// (apps/*_resident.hpp) supply the device work through the hooks.
//
// Session directory protocol (one directory per rank, same path on both nodes; the driver writes, the engine reads):
//   ready.r<R>         engine -> driver: the model is loaded ({"rank","pid","load_s"})
//   job-<N>.json       driver -> engine: {"id","plan":{...},"out":"/abs/dir","budget_s":S}  (N = 1, 2, ...; atomic rename)
//   stop-<N>           driver -> engine: no job N; the session ends
//   done-<N>.r<R>      engine -> driver: status of job N on rank R (the same JSON as <out>/status[.r<R>].json)
// Both ranks agree on every job (kind, size, fnv64 of the bytes) before touching it; a mismatch throws on both.
// A job that fails deterministically (bad JSON, wrong engine, reload-only arms, a missing prompt file) gets an
// error status and the session goes on; an engine exception during a run ends the process (the driver then
// marks the job crashed and serves the rest in the next session).

#include "strix/lever_cache.hpp"
#include "strix/serve/resident_plan.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace strix::serve::resident {

// ---------------------------------------------------------------------------------------------------
// SHA-256 (logits anchors of some engines are sha256 prefixes of the raw f32 row)

namespace detail {
struct Sha256 {
    std::uint32_t h[8] = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                          0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    std::uint8_t buf[64] = {};
    std::size_t used = 0;
    std::uint64_t bits = 0;
    static std::uint32_t rotr(std::uint32_t x, unsigned n) noexcept { return (x >> n) | (x << (32U - n)); }
    void block(const std::uint8_t* p) noexcept {
        static constexpr std::uint32_t k[64] = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
        std::uint32_t w[64];
        for (unsigned i = 0; i < 16; ++i)
            w[i] = (static_cast<std::uint32_t>(p[4 * i]) << 24U) | (static_cast<std::uint32_t>(p[4 * i + 1]) << 16U) |
                   (static_cast<std::uint32_t>(p[4 * i + 2]) << 8U) | static_cast<std::uint32_t>(p[4 * i + 3]);
        for (unsigned i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3U);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10U);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (unsigned i = 0; i < 64; ++i) {
            const std::uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const std::uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }
    void update(const std::uint8_t* p, std::size_t n) noexcept {
        bits += static_cast<std::uint64_t>(n) * 8U;
        while (n > 0) {
            const std::size_t take = std::min<std::size_t>(64U - used, n);
            std::memcpy(buf + used, p, take);
            used += take;
            p += take;
            n -= take;
            if (used == 64U) {
                block(buf);
                used = 0;
            }
        }
    }
    std::string hex() noexcept {
        const std::uint64_t total = bits;
        const std::uint8_t one = 0x80U;
        update(&one, 1);
        const std::uint8_t zero = 0;
        while (used != 56U) update(&zero, 1);
        std::uint8_t len[8];
        for (unsigned i = 0; i < 8; ++i) len[i] = static_cast<std::uint8_t>(total >> (56U - 8U * i));
        update(len, 8);
        std::string out;
        for (const std::uint32_t v : h) out += hex32(v);
        return out;
    }
};
}  // namespace detail

[[nodiscard]] inline std::string sha256_hex(const void* data, std::size_t n) {
    detail::Sha256 s;
    s.update(static_cast<const std::uint8_t*>(data), n);
    return s.hex();
}

[[nodiscard]] inline std::uint64_t fnv1a64_bytes(const void* data, std::size_t n, std::uint64_t h = kFnv64Offset) noexcept {
    const auto* p = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= kFnv64Prime;
    }
    return h;
}

// ---------------------------------------------------------------------------------------------------
// Arm switching (env levers)

// Process environment observed before the first arm: every key any arm overrides, present or absent. Keys that
// the session was launched with (reload levers included) are part of it, so an arm that repeats the launch value
// of a reload lever stays resident.
struct Baseline {
    std::vector<std::pair<std::string, std::optional<std::string>>> env;  // sorted by name
};

template<class GetEnv>
[[nodiscard]] Baseline capture_baseline(const Plan& plan, GetEnv&& getenv_fn) {
    Baseline b;
    for (const Arm& arm : plan.arms)
        for (const auto& kv : arm.env) {
            const bool seen =
                std::any_of(b.env.begin(), b.env.end(), [&](const auto& e) { return e.first == kv.first; });
            if (seen) continue;
            const char* v = getenv_fn(kv.first.c_str());
            b.env.emplace_back(kv.first, v ? std::optional<std::string>(v) : std::nullopt);
        }
    std::sort(b.env.begin(), b.env.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
    return b;
}

[[nodiscard]] inline const std::optional<std::string>* baseline_value(const Baseline& base, std::string_view key) {
    for (const auto& e : base.env)
        if (e.first == key) return &e.second;
    return nullptr;
}

// Reload-class overrides of an arm that differ from the launch environment, as "K=V" text ("" = resident).
[[nodiscard]] inline std::string reload_mismatch(const Arm& arm, const LeverTable& table, const Baseline& base) {
    std::string out;
    for (const auto& kv : arm.env) {
        if (lever_class(table, kv.first) != LeverClass::reload) continue;
        const auto* bv = baseline_value(base, kv.first);
        if (bv != nullptr && bv->has_value() && **bv == kv.second) continue;
        if (!out.empty()) out += ' ';
        out += kv.first + "=" + kv.second;
    }
    return out;
}

// Switching to one arm: every baseline key is set to the arm's value or restored to its baseline (unset when it
// was absent), so no override of an earlier arm leaks into the next one.
struct ArmApplication {
    std::vector<std::pair<std::string, std::string>> set;
    std::vector<std::string> unset;
};

[[nodiscard]] inline ArmApplication application(const Arm& arm, const LeverTable& table, const Baseline& base) {
    const std::string mismatch = reload_mismatch(arm, table, base);
    if (!mismatch.empty()) throw PlanError("resident plan: arm \"" + arm.name + "\" needs a reload for " + mismatch);
    ArmApplication app;
    for (const auto& b : base.env) {
        const auto it =
            std::find_if(arm.env.begin(), arm.env.end(), [&](const auto& kv) { return kv.first == b.first; });
        if (it != arm.env.end()) app.set.emplace_back(it->first, it->second);
        else if (b.second) app.set.emplace_back(b.first, *b.second);
        else app.unset.push_back(b.first);
    }
    return app;
}

// ---------------------------------------------------------------------------------------------------
// Hooks an engine supplies

struct RunIo {
    std::vector<std::uint32_t> tokens;  // generated ids (prefill token first)
    double prefill_s = 0.0;             // prefill wall (TTFT)
    double decode_s = 0.0;              // steps-1 decode steps
    std::string note;                   // free text for the RESIDENT_RUN log line
};

// Called with each host logits row the run keeps: row 0 is the prefill (first-token) distribution of this rank's
// vocab slice, rows 1.. the decode steps (only when the arm captures "logits").
using LogitsSink = std::function<void(const float* row, std::size_t n)>;

struct BatchRequest {
    std::size_t prompt;
    std::uint32_t max_tokens;
    std::uint64_t seed;
};
enum class BatchMode { Lockstep, Continuous };
struct MtpPolicy {
    std::uint32_t depth = 0, rows = 0;
};
struct BatchSpec {
    BatchMode mode;
    std::vector<BatchRequest> requests;
    MtpPolicy mtp;
};
struct BatchCycle {
    std::uint32_t rows = 0, useful_rows = 0, lanes = 0;
    double ms = 0;
    std::int64_t expert_union = -1;
};
struct BatchIo {
    std::vector<std::vector<std::uint32_t>> tokens;
    std::vector<double> ttft_s;
    std::vector<BatchCycle> cycles;
    double prefill_s = 0, decode_s = 0;
    std::string note;
    std::string unsupported;
};

struct Hooks {
    std::vector<std::string> engine_names;  // accepted plan "engine" values
    LeverTable levers;
    unsigned rank = 0;
    unsigned world = 1;
    std::uint64_t capacity = 0;  // max prompt + steps positions (0 = unchecked)
    std::uint64_t prompt_max = 0;  // > 0: prompts are cut to this many tokens
    // Optional prompt reader for the plan's prompt files (default: little-endian u32 ids, read_u32_file), so an
    // engine whose CLI reads another format runs the same files as its fresh processes.
    std::function<std::vector<std::uint32_t>(const std::string& path)> read_prompt;
    // TP2 agreement on 8 fields: field 0 = this rank's eligibility, 1..7 must match exactly (mismatch throws on both
    // ranks). Returns true when every rank is eligible. Single node: leave empty.
    std::function<bool(const std::array<std::uint32_t, 8>&)> agree;
    // Optional check of every arm before the plan starts (throw PlanError for a value the engine cannot apply).
    std::function<void(const Arm&)> validate_arm;
    std::function<void(const BatchSpec&)> validate_batch;
    // After the env of an arm (or the baseline) is set: re-resolve construct/cached levers from the environment.
    std::function<void()> apply_levers;
    // Reset per-request state (KV, positions, draft state, RNG).
    std::function<void()> reset;
    // Optional peer barrier just before the timed run (TP2 ranks start the clock together).
    std::function<void(std::uint64_t seq)> barrier;
    // One timed request: prefill `prompt`, decode to `steps` tokens in total. When `sink` is set, call it with the
    // prefill logits row (outside the timed regions) and, when `all_rows`, with every decode step's row too.
    std::function<RunIo(const std::vector<std::uint32_t>& prompt, unsigned steps, const LogitsSink* sink, bool all_rows)>
        run;
    // Optional digest of the engine state after a run of n prompt tokens (leak check), hex.
    std::function<std::string(std::size_t n)> state_digest;
    // One engine-owned batched job; an empty hook or unsupported result permits solo controls.
    std::function<BatchIo(const BatchSpec&, const std::vector<std::vector<std::uint32_t>>& prompts)> run_batch;
};

inline constexpr std::uint32_t kTagPlan = 0x52504c4eU;  // "RPLN"
inline constexpr std::uint32_t kTagCtl = 0x5243544cU;   // "RCTL": rank 0 continue/stop vote per slot
inline constexpr std::uint32_t kTagSlot = 0x52534c54U;  // "RSLT": arm switch + prompt per slot
inline constexpr std::uint32_t kTagJob = 0x524a4f42U;   // "RJOB": session job
inline constexpr std::uint32_t kTagBatch = 0x52424154U; // "RBAT": N, mode, MTP and total token agreement
inline constexpr std::uint32_t kTagInput = 0x52494e50U; // "RINP": effective input digest, count and request index

[[nodiscard]] inline std::array<std::uint32_t, 8> agreement_fields(std::uint32_t tag, std::uint64_t hash,
                                                                   std::uint32_t a, std::uint32_t b, std::uint32_t c,
                                                                   std::uint32_t d) {
    return {1U, tag, static_cast<std::uint32_t>(hash & 0xffffffffULL), static_cast<std::uint32_t>(hash >> 32U), a, b, c, d};
}

[[nodiscard]] inline bool agree_all(const Hooks& h, const std::array<std::uint32_t, 8>& f) {
    if (h.world <= 1 || !h.agree) return f[0] != 0;
    return h.agree(f);
}

// ---------------------------------------------------------------------------------------------------
// Files

[[nodiscard]] inline std::vector<std::uint32_t> read_u32_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open prompt " + path);
    const std::streamoff n = in.tellg();
    if (n <= 0 || n % 4 != 0 || static_cast<std::uint64_t>(n) > kMaxPromptBytes)
        throw std::runtime_error("prompt " + path + " is empty, oversized or not u32 LE");
    std::vector<std::uint32_t> out(static_cast<std::size_t>(n / 4));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(out.data()), n);
    if (!in) throw std::runtime_error("cannot read prompt " + path);
    return out;
}

[[nodiscard]] inline std::string read_text_file(const std::filesystem::path& path, std::size_t limit = kMaxPlanBytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path.string());
    std::string text;
    char buf[4096];
    while (in) {
        in.read(buf, sizeof(buf));
        const auto n = static_cast<std::size_t>(in.gcount());
        if (n > limit - text.size()) throw PlanError("resident input exceeds byte limit: " + path.string());
        text.append(buf, n);
    }
    if (!in.eof()) throw std::runtime_error("cannot read " + path.string());
    return text;
}

// Writes via a temporary name and a rename, so a reader never sees a partial file.
inline void write_text_atomic(const std::filesystem::path& path, const std::string& text) {
    const std::filesystem::path tmp = path.string() + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f << text << '\n';
        if (!f) throw std::runtime_error("write " + tmp.string());
    }
    std::filesystem::rename(tmp, path);
}

[[nodiscard]] inline std::string rank_suffix(unsigned rank) { return rank == 0 ? "" : ".r" + std::to_string(rank); }

[[nodiscard]] inline std::string logits_name(const Plan& plan, const RunSlot& slot, unsigned rank) {
    char seq[24];
    std::snprintf(seq, sizeof seq, "%05llu", static_cast<unsigned long long>(slot.seq));
    return "logits-" + std::string(seq) + "-" + plan.arms[slot.arm].name + "-" + plan.prompts[slot.prompt].name +
           "-r" + std::to_string(rank) + ".f32";
}

// ---------------------------------------------------------------------------------------------------
// One plan

struct PlanOutcome {
    std::string state = "ok";  // ok | stopped-early | over-budget | stopped-by-rank0 | error
    std::string error;
    std::string plan_hash;
    std::size_t slots = 0;
    std::size_t runs = 0;     // receipts written
    std::size_t errors = 0;   // receipts with an error (needs reload, missing prompt, capacity)
    std::size_t skipped = 0;  // slots not run (early stop or plan budget)
    double wall_s = 0.0;
    double budget_s = 0.0;
    std::vector<std::string> reload_groups;  // reload overrides of arms this session could not run
    std::vector<std::pair<std::string, double>> arm_s;  // GPU wall per arm (warmups included), plan order
};

[[nodiscard]] inline Json outcome_json(const PlanOutcome& o, std::string_view id, unsigned rank) {
    Json j = Json::object();
    j.set("id", Json(std::string(id)));
    j.set("rank", Json(static_cast<std::uint64_t>(rank)));
    j.set("state", Json(o.state));
    if (!o.error.empty()) j.set("error", Json(o.error));
    j.set("plan_hash", Json(o.plan_hash));
    j.set("slots", Json(static_cast<std::uint64_t>(o.slots)));
    j.set("runs", Json(static_cast<std::uint64_t>(o.runs)));
    j.set("run_errors", Json(static_cast<std::uint64_t>(o.errors)));
    j.set("skipped", Json(static_cast<std::uint64_t>(o.skipped)));
    j.set("wall_s", Json(o.wall_s));
    j.set("budget_s", Json(o.budget_s));
    Json arms = Json::object();
    for (const auto& a : o.arm_s) arms.set(a.first, Json(a.second));
    j.set("arm_s", std::move(arms));
    Json rg = Json::array();
    for (const std::string& g : o.reload_groups) rg.push(Json(g));
    j.set("needs_reload", std::move(rg));
    return j;
}

namespace detail {
inline int set_env(const char* k, const char* v) {
#if defined(_WIN32)
    return _putenv_s(k, v);
#else
    return ::setenv(k, v, 1);
#endif
}
inline void unset_env(const char* k) {
#if defined(_WIN32)
    (void)_putenv_s(k, "");
#else
    (void)::unsetenv(k);
#endif
}
inline long process_id() {
#if defined(_WIN32)
    return static_cast<long>(_getpid());
#else
    return static_cast<long>(::getpid());
#endif
}
using Clock = std::chrono::steady_clock;
inline double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }
}  // namespace detail

// Per-plan GPU time budget of a session job: the default when a job names none, and the hard cap (tests stay short).
inline constexpr double kDefaultBudgetS = 120.0;
inline constexpr double kMaxBudgetS = 300.0;

[[nodiscard]] inline BatchSpec batch_spec(const Plan& plan) {
    BatchSpec spec{plan.batch_mode.value_or("lockstep") == "continuous" ? BatchMode::Continuous : BatchMode::Lockstep,
                   {}, {}};
    if (plan.mtp) spec.mtp = {plan.mtp->depth, plan.mtp->rows};
    for (std::size_t i = 0; i < plan.concurrency.value_or(0); ++i) {
        if (plan.requests) {
            const auto& r = plan.requests->at(i);
            spec.requests.push_back({r.prompt, r.max_tokens, r.seed});
        } else {
            spec.requests.push_back({i % plan.prompts.size(), plan.steps, 0});
        }
    }
    return spec;
}

[[nodiscard]] inline const char* batch_mode_name(BatchMode mode) {
    return mode == BatchMode::Continuous ? "continuous" : "lockstep";
}

[[nodiscard]] inline std::string batch_tokens_fnv(const std::vector<std::uint32_t>& tokens) {
    std::vector<std::int32_t> ids;
    ids.reserve(tokens.size());
    for (const auto t : tokens) ids.push_back(static_cast<std::int32_t>(t));
    return hex64(token_digest(ids));
}

// A missing solo result means it was disabled or the job budget stopped before that request.
// Use null for unmeasured comparisons rather than asserting equality without a control.
[[nodiscard]] inline Json batch_receipt(const BatchSpec& spec, const BatchIo& io,
                                         const std::vector<std::optional<RunIo>>& solos) {
    Json out = Json::object();
    if (!io.unsupported.empty()) {
        out.set("status", Json("unsupported"));
        out.set("reason", Json(io.unsupported));
        return out;
    }
    if (io.tokens.size() != spec.requests.size() ||
        (!io.ttft_s.empty() && io.ttft_s.size() != spec.requests.size()))
        throw std::runtime_error("resident: batch result request count mismatch");
    out.set("status", Json("ok"));
    out.set("mode", Json(batch_mode_name(spec.mode)));
    out.set("n", Json(static_cast<std::uint64_t>(spec.requests.size())));
    Json mtp = Json::object();
    mtp.set("depth", Json(static_cast<std::uint64_t>(spec.mtp.depth)));
    mtp.set("rows", Json(static_cast<std::uint64_t>(spec.mtp.rows)));
    out.set("mtp", std::move(mtp));
    Json requests = Json::array();
    std::uint64_t total_tokens = 0, decode_tokens = 0, solo_tokens = 0;
    double solo_s = 0;
    bool all_match = true;
    for (std::size_t i = 0; i < spec.requests.size(); ++i) {
        const auto& r = spec.requests[i];
        const auto& tokens = io.tokens[i];
        Json item = Json::object();
        item.set("i", Json(static_cast<std::uint64_t>(i)));
        item.set("prompt", Json(static_cast<std::uint64_t>(r.prompt)));
        item.set("max_tokens", Json(static_cast<std::uint64_t>(r.max_tokens)));
        item.set("seed", Json(r.seed));
        item.set("tokens_fnv", Json(batch_tokens_fnv(tokens)));
        item.set("n_tokens", Json(static_cast<std::uint64_t>(tokens.size())));
        item.set("ttft_s", io.ttft_s.empty() ? Json() : Json(io.ttft_s[i]));
        const bool match = solos[i] && tokens == solos[i]->tokens;
        item.set("solo_tokens_fnv", solos[i] ? Json(batch_tokens_fnv(solos[i]->tokens)) : Json());
        item.set("match_solo", solos[i] ? Json(match) : Json());
        all_match = all_match && match;
        if (solos[i]) {
            solo_s += solos[i]->prefill_s + solos[i]->decode_s;
            solo_tokens += static_cast<std::uint64_t>(solos[i]->tokens.size());
        }
        total_tokens += static_cast<std::uint64_t>(tokens.size());
        if (!tokens.empty()) decode_tokens += static_cast<std::uint64_t>(tokens.size() - 1);
        requests.push(std::move(item));
    }
    out.set("requests", std::move(requests));
    Json cycles = Json::object(), detail = Json::array();
    std::uint64_t rows = 0, useful = 0;
    std::uint32_t lanes = 0;
    double ms_sum = 0, expert_sum = 0;
    std::size_t experts = 0;
    std::vector<double> timings;
    for (const auto& c : io.cycles) {
        rows += c.rows;
        useful += c.useful_rows;
        lanes = std::max(lanes, c.lanes);
        ms_sum += c.ms;
        timings.push_back(c.ms);
        if (c.expert_union >= 0) {
            expert_sum += static_cast<double>(c.expert_union);
            ++experts;
        }
        if (detail.size() < 64) {
            Json item = Json::object();
            item.set("rows", Json(static_cast<std::uint64_t>(c.rows)));
            item.set("useful_rows", Json(static_cast<std::uint64_t>(c.useful_rows)));
            item.set("lanes", Json(static_cast<std::uint64_t>(c.lanes)));
            item.set("ms", Json(c.ms));
            item.set("expert_union", Json(c.expert_union));
            detail.push(std::move(item));
        }
    }
    std::sort(timings.begin(), timings.end());
    double p50 = 0;
    if (!timings.empty()) {
        const std::size_t mid = timings.size() / 2;
        p50 = timings.size() % 2 != 0 ? timings[mid] : (timings[mid - 1] + timings[mid]) / 2;
    }
    cycles.set("n", Json(static_cast<std::uint64_t>(io.cycles.size())));
    cycles.set("rows_sum", Json(rows));
    cycles.set("useful_rows_sum", Json(useful));
    cycles.set("lanes_max", Json(static_cast<std::uint64_t>(lanes)));
    cycles.set("ms_mean", Json(timings.empty() ? 0.0 : ms_sum / static_cast<double>(timings.size())));
    cycles.set("ms_p50", Json(p50));
    cycles.set("expert_union_mean", experts == 0 ? Json() : Json(expert_sum / static_cast<double>(experts)));
    cycles.set("detail", std::move(detail));
    out.set("cycles", std::move(cycles));
    out.set("prefill_s", Json(io.prefill_s));
    out.set("decode_s", Json(io.decode_s));
    const double total_s = io.prefill_s + io.decode_s;
    out.set("agg_tok_s", Json(total_s > 0 ? static_cast<double>(total_tokens) / total_s : 0.0));
    out.set("decode_tok_s", Json(io.decode_s > 0 ? static_cast<double>(decode_tokens) / io.decode_s : 0.0));
    Json solo = Json::object();
    solo.set("total_s", Json(solo_s));
    solo.set("agg_tok_s", Json(solo_s > 0 ? static_cast<double>(solo_tokens) / solo_s : 0.0));
    out.set("solo", std::move(solo));
    out.set("all_match_solo", Json(all_match));
    out.set("note", Json(io.note));
    return out;
}

// Runs every slot of `plan` on the loaded engine and writes receipts, logits files and (rank 0) summary.json into
// `out`. budget_s > 0: rank 0 stops the plan before a slot that would end past the budget (estimate: the longest
// slot so far). PlanError means pre-execution validation failed; execution failures are fatal runtime errors.
// Reload groups only diagnose/refuse incompatible arms: no reload or fresh cold control is performed.
// An "ok" outcome may contain refused/missing-input slots. Automation must inspect errors (summary: errors;
// session status: run_errors), not just state or the CLI exit code.
[[nodiscard]] inline PlanOutcome run_plan(const Plan& plan, const std::filesystem::path& out, double budget_s,
                                          Hooks& h) {
    using detail::Clock;
    const auto t_plan = Clock::now();
    PlanOutcome oc;
    oc.plan_hash = hex64(plan.hash);
    oc.budget_s = budget_s;
    for (const Arm& a : plan.arms) oc.arm_s.emplace_back(a.name, 0.0);
    if (std::find(h.engine_names.begin(), h.engine_names.end(), plan.engine) == h.engine_names.end())
        throw PlanError("resident plan: engine \"" + plan.engine + "\" is not served by this session");
    if (!h.run) throw std::invalid_argument("resident: engine adapter has no run hook");
    const BatchSpec spec = batch_spec(plan);
    try {
        if (h.validate_arm)
            for (const Arm& arm : plan.arms) h.validate_arm(arm);
        if (plan.concurrency && h.validate_batch) h.validate_batch(spec);
    } catch (const std::invalid_argument& e) {
        throw PlanError(e.what());
    } catch (const std::out_of_range& e) {
        throw PlanError(e.what());
    }
    const auto groups = group_arms(plan, h.levers);
    // Each batch is one occurrence, regardless of the arm's serial prompt list.
    const auto slots = [&] {
        if (!plan.concurrency) return schedule(plan, groups);
        Plan scheduled_plan = plan;
        for (auto& arm : scheduled_plan.arms) arm.prompts.resize(1);
        return schedule(scheduled_plan, groups);
    }();
    oc.slots = slots.size();
    const Baseline base = capture_baseline(plan, [](const char* k) { return std::getenv(k); });
    std::vector<std::string> mismatch(plan.arms.size());
    for (std::size_t a = 0; a < plan.arms.size(); ++a) {
        mismatch[a] = reload_mismatch(plan.arms[a], h.levers, base);
        if (!mismatch[a].empty() &&
            std::find(oc.reload_groups.begin(), oc.reload_groups.end(), mismatch[a]) == oc.reload_groups.end())
            oc.reload_groups.push_back(mismatch[a]);
    }
    if (!agree_all(h, agreement_fields(kTagPlan, plan.hash, static_cast<std::uint32_t>(plan.arms.size()),
                                       static_cast<std::uint32_t>(slots.size()), 0U, 0U)))
        throw PlanError("resident plan: plan agreement refused");
    std::filesystem::create_directories(out);
    const std::string suffix = rank_suffix(h.rank);
    std::printf("rank %u RESIDENT plan hash %s arms %zu groups %zu slots %zu\n", h.rank, oc.plan_hash.c_str(),
                plan.arms.size(), groups.size(), slots.size());
    std::fflush(stdout);
    auto apply_env = [&](const ArmApplication& app) {
        for (const auto& kv : app.set)
            if (detail::set_env(kv.first.c_str(), kv.second.c_str()) != 0) throw std::runtime_error("setenv " + kv.first);
        for (const auto& k : app.unset) detail::unset_env(k.c_str());
        strix::lever_environment_changed();  // every strix::lever_cache resolves again from the arm's environment
        if (h.apply_levers) h.apply_levers();
    };
    bool env_dirty = false;
    const auto restore = [&] {
        if (!env_dirty) return;
        apply_env(application(Arm{}, h.levers, base));
        env_dirty = false;
    };
    try {
    std::vector<RunRecord> records;
    StopDecision stop;
    double longest = 0.0;
    bool budget_hit = false;
    std::size_t applied_arm = plan.arms.size();
    // The first run after the load or an arm switch is the cold run (validity rule: cold and warm are reported apart).
    bool after_switch = true;
    const auto load_prompt = [&](const std::string& path) { return h.read_prompt ? h.read_prompt(path) : read_u32_file(path); };
    for (std::size_t si = 0; si < slots.size(); ++si) {
        const RunSlot& slot = slots[si];
        // Rank 0 owns the continue/stop vote (early stop or plan budget); every rank follows it.
        bool go = !stop.stop;
        if (go && budget_s > 0.0 && detail::secs(t_plan, Clock::now()) + longest > budget_s) {
            go = false;
            budget_hit = true;
        }
        auto ctl = agreement_fields(kTagCtl, plan.hash, static_cast<std::uint32_t>(slot.seq), 0U, 0U, 0U);
        ctl[0] = (h.rank != 0 || go) ? 1U : 0U;
        if (!agree_all(h, ctl)) {
            oc.skipped = slots.size() - si;
            if (h.rank != 0) oc.state = "stopped-by-rank0";  // the other ranks learn only that rank 0 stopped
            break;
        }
        const auto t_slot = Clock::now();
        const Arm& arm = plan.arms[slot.arm];
        const std::string& ppath = plan.prompts[slot.prompt].path;
        RunRecord rec{slot, {}};
        Json batch;
        std::vector<std::vector<std::uint32_t>> batch_prompts;
        std::vector<std::uint32_t> toks;
        std::string local_error;
        if (slot.cold) {
            local_error = "needs reload (cold control)";
        } else if (!mismatch[slot.arm].empty()) {
            local_error = "needs reload: " + mismatch[slot.arm];
        } else {
            try {
                if (plan.concurrency) {
                    batch_prompts.resize(plan.prompts.size());
                    for (const auto& request : spec.requests) {
                        auto& prompt = batch_prompts[request.prompt];
                        if (prompt.empty()) prompt = load_prompt(plan.prompts[request.prompt].path);
                        if (h.prompt_max > 0 && prompt.size() > h.prompt_max)
                            prompt.resize(static_cast<std::size_t>(h.prompt_max));
                        if (h.capacity > 0 && prompt.size() + request.max_tokens > h.capacity)
                            throw PlanError("batch request exceeds the session capacity " + std::to_string(h.capacity));
                    }
                } else {
                    toks = load_prompt(ppath);
                    if (h.prompt_max > 0 && toks.size() > h.prompt_max) toks.resize(static_cast<std::size_t>(h.prompt_max));
                    if (h.capacity > 0 && toks.size() + arm.steps > h.capacity)
                        local_error = "prompt " + std::to_string(toks.size()) + " + steps " + std::to_string(arm.steps) +
                                      " exceeds the session capacity " + std::to_string(h.capacity);
                }
            } catch (const std::exception& e) {
                local_error = e.what();
            }
        }
        auto f = agreement_fields(kTagSlot, plan.hash, static_cast<std::uint32_t>(slot.arm), arm_digest(plan, slot.arm),
                                  static_cast<std::uint32_t>(slot.seq), static_cast<std::uint32_t>(fnv1a64(ppath)));
        f[0] = local_error.empty() ? 1U : 0U;
        const bool all_ok = agree_all(h, f);
        if (!all_ok) {
            rec.result.error = local_error.empty() ? std::string("a peer rank could not run this slot") : local_error;
        } else {
            const auto agree_input = [&](const std::vector<std::uint32_t>& prompt, std::size_t request) {
                const auto count = static_cast<std::uint64_t>(prompt.size());
                const auto digest = token_digest(prompt);
                const auto fields = agreement_fields(kTagInput, digest, static_cast<std::uint32_t>(count),
                    static_cast<std::uint32_t>(count >> 32U), static_cast<std::uint32_t>(request),
                    static_cast<std::uint32_t>(slot.seq));
                if (!agree_all(h, fields)) throw std::runtime_error("resident: prompt input agreement refused");
                Json input = Json::object();
                input.set("request", Json(static_cast<std::uint64_t>(request)));
                input.set("n_tokens", Json(count));
                input.set("tokens_fnv", Json(hex64(digest)));
                rec.result.prompt_inputs.push(std::move(input));
            };
            if (plan.concurrency) {
                for (std::size_t i = 0; i < spec.requests.size(); ++i)
                    agree_input(batch_prompts[spec.requests[i].prompt], i);
            } else {
                agree_input(toks, 0);
            }
            if (applied_arm != slot.arm) {
                env_dirty = true;  // even a partially applied arm must roll back on failure
                apply_env(application(arm, h.levers, base));
                applied_arm = slot.arm;
                after_switch = true;
            }
            rec.result.cold_run = after_switch;
            after_switch = false;
            if (h.reset) h.reset();
            if (h.barrier) h.barrier(slot.seq);
            if (plan.concurrency) {
                std::uint64_t total_tokens = 0;
                for (const auto& request : spec.requests)
                    total_tokens += static_cast<std::uint64_t>(batch_prompts[request.prompt].size()) + request.max_tokens;
                auto bf = agreement_fields(kTagBatch, total_tokens, *plan.concurrency,
                                           static_cast<std::uint32_t>(spec.mode), spec.mtp.depth, spec.mtp.rows);
                bf[0] = h.run_batch ? 1U : 0U;
                BatchIo io;
                const auto t_batch = Clock::now();
                if (agree_all(h, bf)) io = h.run_batch(spec, batch_prompts);
                else io.unsupported = "engine has no batched path";
                longest = std::max(longest, detail::secs(t_batch, Clock::now()));
                std::vector<std::optional<RunIo>> solos(spec.requests.size());
                if (plan.solo_control.value_or(true)) {
                    for (std::size_t i = 0; i < spec.requests.size(); ++i) {
                        bool solo_go = true;
                        if (budget_s > 0 && detail::secs(t_plan, Clock::now()) + longest > budget_s) {
                            solo_go = false;
                            budget_hit = true;
                        }
                        auto vote = agreement_fields(kTagCtl, plan.hash, static_cast<std::uint32_t>(slot.seq),
                                                     static_cast<std::uint32_t>(i + 1), 0U, 0U);
                        vote[0] = (h.rank != 0 || solo_go) ? 1U : 0U;
                        if (!agree_all(h, vote)) {
                            if (h.rank != 0) oc.state = "stopped-by-rank0";
                            break;
                        }
                        const auto t_solo = Clock::now();
                        if (h.reset) h.reset();
                        if (h.barrier) h.barrier(slot.seq);
                        const auto& r = spec.requests[i];
                        solos[i] = h.run(batch_prompts[r.prompt], static_cast<unsigned>(r.max_tokens), nullptr, false);
                        longest = std::max(longest, detail::secs(t_solo, Clock::now()));
                    }
                }
                batch = batch_receipt(spec, io, solos);
                if (io.unsupported.empty()) {
                    rec.result.prefill_ms = io.prefill_s * 1000;
                    rec.result.ttft_ms = rec.result.prefill_ms;
                    rec.result.decode_ms = io.decode_s * 1000;
                    rec.result.decode_tok_s = batch.find("decode_tok_s")->as_double();
                }
                std::printf("rank %u RESIDENT_BATCH n=%zu mode=%s agg_tok_s=%.3f match=%s%s\n", h.rank,
                            spec.requests.size(), batch_mode_name(spec.mode),
                            io.unsupported.empty() ? batch.find("agg_tok_s")->as_double() : 0.0,
                            io.unsupported.empty() && batch.find("all_match_solo")->as_bool() ? "true" : "false",
                            slot.warmup ? " warmup" : "");
            } else {
                const bool want_rows = (arm.capture & kCaptureLogits) != 0 && plan.logits_last_n > 0;
                std::deque<std::vector<float>> ring;
                std::uint64_t row0_fnv = 0;
                std::string row0_sha;
                bool have_row0 = false;
                const LogitsSink sink = [&](const float* row, std::size_t n) {
                    if (!have_row0) {
                        row0_fnv = fnv1a64_bytes(row, n * sizeof(float));
                        row0_sha = sha256_hex(row, n * sizeof(float));
                        have_row0 = true;
                    }
                    if (!want_rows) return;
                    ring.emplace_back(row, row + n);
                    if (ring.size() > plan.logits_last_n) ring.pop_front();
                };
                const unsigned steps = std::max(1U, static_cast<unsigned>(arm.steps));
                const RunIo io = h.run(toks, steps, &sink, want_rows);
                RunResult& res = rec.result;
                if (arm.capture & kCaptureTiming) {
                    res.prefill_ms = 1000.0 * io.prefill_s;
                    res.ttft_ms = res.prefill_ms;
                    res.prefill_tok_s = io.prefill_s > 0 ? static_cast<double>(toks.size()) / io.prefill_s : 0.0;
                    res.decode_ms = 1000.0 * io.decode_s;
                    if (io.tokens.size() > 1 && io.decode_s > 0)
                        res.decode_tok_s = static_cast<double>(io.tokens.size() - 1) / io.decode_s;
                }
                for (const auto t : io.tokens) res.tokens.push_back(static_cast<std::int32_t>(t));
                if (have_row0) {
                    res.logits_fnv = hex64(row0_fnv);
                    res.logits_sha256 = row0_sha;
                }
                if ((arm.capture & kCaptureStateDigest) && h.state_digest) res.state_digest = h.state_digest(toks.size());
                if (want_rows) {
                    const std::string name = logits_name(plan, slot, h.rank);
                    std::ofstream lf(out / name, std::ios::binary);
                    for (const auto& lg : ring)
                        lf.write(reinterpret_cast<const char*>(lg.data()), static_cast<std::streamsize>(lg.size() * 4));
                    if (!lf) throw std::runtime_error("write " + name);
                    res.logits_path = name;
                }
                std::printf("rank %u RESIDENT_RUN seq %llu arm %s prompt %s pass %u%s%s: %zu tok prefill %.3f s, %zu tokens "
                            "decode %.3f s, tokens fnv %s, logits0 fnv %s%s%s\n",
                            h.rank, static_cast<unsigned long long>(slot.seq), arm.name.c_str(),
                            plan.prompts[slot.prompt].name.c_str(), slot.pass, slot.warmup ? " warmup" : "",
                            res.cold_run ? " cold" : "", toks.size(),
                            io.prefill_s, io.tokens.size(), io.decode_s, hex64(token_digest(res.tokens)).c_str(),
                            have_row0 ? res.logits_fnv.c_str() : "-", io.note.empty() ? "" : ", ", io.note.c_str());
            }
        }
        Json receipt = receipt_json(plan, rec);
        if (plan.concurrency && !batch.is_null()) receipt.set("batch", std::move(batch));
        write_text_atomic(out / (receipt_name(plan, slot) + suffix), receipt.dump());
        ++oc.runs;
        if (!rec.result.error.empty()) ++oc.errors;
        records.push_back(std::move(rec));
        const double slot_s = detail::secs(t_slot, Clock::now());
        longest = std::max(longest, slot_s);
        oc.arm_s[slot.arm].second += slot_s;
        if (h.rank == 0 && !stop.stop) stop = decide_early_stop(plan, compare_all(plan, groups, records));
        std::fflush(stdout);
    }
    if (h.rank == 0) {
        const auto comps = compare_all(plan, groups, records);
        write_text_atomic(out / "summary.json", summary_json(plan, groups, records, comps, stop).dump());
    }
    restore();
    if (h.reset) h.reset();
    if (stop.stop) oc.state = "stopped-early";
    else if (budget_hit) oc.state = "over-budget";  // rank 0 only; see above for the other ranks
    oc.wall_s = detail::secs(t_plan, Clock::now());
    std::printf("rank %u RESIDENT_DONE runs %zu/%zu state %s%s%s wall %.1f s\n", h.rank, oc.runs, oc.slots,
                oc.state.c_str(), stop.stop ? " " : "", stop.stop ? stop.reason.c_str() : "", oc.wall_s);
    std::fflush(stdout);
    return oc;
    } catch (...) {
        try {
            restore();  // restore both environment and partially resolved object members
        } catch (...) {
            std::throw_with_nested(std::runtime_error("resident: baseline restoration failed; session must end"));
        }
        std::throw_with_nested(std::runtime_error("resident: execution failed; session must end"));
    }
}

// ---------------------------------------------------------------------------------------------------
// Session (a stream of jobs)

struct Job {
    std::string id;
    std::string plan_text;  // compact plan JSON
    std::filesystem::path out;
    double budget_s = 0.0;  // parse_job: kDefaultBudgetS when absent, at most kMaxBudgetS
};

[[nodiscard]] inline Job parse_job(std::string_view text) {
    if (text.size() > kMaxPlanBytes) throw PlanError("resident job: input exceeds 1 MiB");
    Json root;
    try {
        root = parse_json_unique(text);
    } catch (const JsonError& e) {
        throw PlanError(std::string("resident job: invalid JSON: ") + e.what());
    }
    detail::object_at(root, "job");
    detail::check_keys(root, "job", {"id", "plan", "out", "budget_s"});
    Job j;
    const Json* id = root.find("id");
    const Json* plan = root.find("plan");
    const Json* out = root.find("out");
    if (id == nullptr || plan == nullptr || out == nullptr) detail::fail("job", "needs \"id\", \"plan\" and \"out\"");
    j.id = detail::string_value(*id, "job.id");
    if (j.id.empty() || j.id.size() > 128) detail::fail("job.id", "1..128 characters");
    detail::object_at(*plan, "job.plan");
    j.plan_text = plan->dump();
    j.out = detail::string_value(*out, "job.out");
    if (!j.out.is_absolute() && !j.out.has_root_directory()) detail::fail("job.out", "must be an absolute path");  // POSIX: the same test
    j.budget_s = kDefaultBudgetS;
    if (const Json* b = root.find("budget_s")) {
        const double v = detail::number_value(*b, "job.budget_s", 0.0, 1.0e6);
        if (v > 0.0) j.budget_s = v;  // 0 = the default
    }
    j.budget_s = std::min(j.budget_s, kMaxBudgetS);  // over the hard cap: clamped, never refused
    return j;
}

struct SessionOptions {
    double poll_s = 0.1;
    double idle_s = 120.0;  // no job and no stop file for this long: the session ends by itself
    double load_s = 0.0;    // reported in ready.r<R>
};

// Serves jobs job-1.json, job-2.json, ... from `dir` until stop-<N> (or idle_s without either). Returns the number
// of jobs served. Engine exceptions during a run propagate (the process ends; the driver handles the crash).
inline std::size_t run_session(const std::filesystem::path& dir, Hooks& h, const SessionOptions& so) {
    using detail::Clock;
    const std::string rs = ".r" + std::to_string(h.rank);
    {
        Json r = Json::object();
        r.set("rank", Json(static_cast<std::uint64_t>(h.rank)));
        r.set("pid", Json(static_cast<std::int64_t>(detail::process_id())));
        r.set("load_s", Json(so.load_s));
        write_text_atomic(dir / ("ready" + rs), r.dump());
    }
    std::printf("rank %u RESIDENT_SESSION ready %s (load %.1f s)\n", h.rank, dir.string().c_str(), so.load_s);
    std::fflush(stdout);
    std::size_t served = 0;
    for (std::uint32_t n = 1;; ++n) {
        const std::filesystem::path job_path = dir / ("job-" + std::to_string(n) + ".json");
        const std::filesystem::path stop_path = dir / ("stop-" + std::to_string(n));
        const auto t_wait = Clock::now();
        bool have_job = false;
        for (;;) {
            if (std::filesystem::exists(job_path)) {
                have_job = true;
                break;
            }
            if (std::filesystem::exists(stop_path)) break;
            if (detail::secs(t_wait, Clock::now()) > so.idle_s) break;
            std::this_thread::sleep_for(std::chrono::duration<double>(so.poll_s));
        }
        std::string bytes;
        if (have_job) bytes = read_text_file(job_path);
        // Kind, size and fnv must match on every rank (a mismatch throws on both and ends the session).
        const std::uint64_t jh = have_job ? fnv1a64(bytes) : 0U;
        if (!agree_all(h, agreement_fields(kTagJob, jh, n, have_job ? 1U : 0U, static_cast<std::uint32_t>(bytes.size()), 0U)))
            throw std::runtime_error("resident session: job agreement refused");
        if (!have_job) break;
        PlanOutcome oc;
        std::string id = "job-" + std::to_string(n);
        std::filesystem::path out;
        try {
            const Job job = parse_job(bytes);
            id = job.id;
            out = job.out;
            const Plan plan = parse_plan(job.plan_text);
            oc = run_plan(plan, job.out, job.budget_s, h);
        } catch (const PlanError& e) {
            oc.state = "error";
            oc.error = e.what();
        }
        const std::string status = outcome_json(oc, id, h.rank).dump();
        if (!out.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(out, ec);
            if (!ec) write_text_atomic(out / ("status" + rank_suffix(h.rank) + ".json"), status);
        }
        write_text_atomic(dir / ("done-" + std::to_string(n) + rs), status);
        std::printf("rank %u RESIDENT_JOB %u %s state %s%s%s\n", h.rank, n, id.c_str(), oc.state.c_str(),
                    oc.error.empty() ? "" : ": ", oc.error.c_str());
        std::fflush(stdout);
        ++served;
    }
    std::printf("rank %u RESIDENT_SESSION end, %zu jobs\n", h.rank, served);
    std::fflush(stdout);
    return served;
}

}  // namespace strix::serve::resident
